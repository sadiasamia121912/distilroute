"""Do the students break on messy input more than the LLM does? (roadmap 6.6)

    python scripts/robustness.py                      # every student, clean + 5 kinds of noise
    python scripts/label.py --split test --perturb typo3 --limit 300 --top-k 3   # then, per kind
    python scripts/robustness.py                      # picks up the teacher's noisy labels

Support tickets are typed on phones: typos, no capitals, "pls", a greeting stuck on the front.
The test split is clean, edited text, so every accuracy so far is a best case. This re-scores
each student on rule-based noisy copies of the test split (`distilroute.perturb`), through the
exact code path the service runs (`distilroute.students`, the int8 ONNX graphs), and reports:

- accuracy per kind of noise, and the drop from clean;
- the same on only the queries the noise actually changed (`slang` leaves a third untouched);
- what the noise does to the cascade: the share of queries escalated at a calibrated 0.8
  threshold, and the accuracy on the ones kept. A student that gets *less sure* on noisy input
  degrades gracefully (the LLM picks it up); one that stays confident and wrong does not.

The teacher is scored on whichever noisy label files exist (`test.perturb_<kind>.jsonl`, a
random subset: labelling all of test five times would cost ~5 days of free tier), against the
students on the same rows.

Writes results/robustness.json and docs/robustness.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distilroute import students  # noqa: E402
from distilroute.data import DOCS, LABELS, RESULTS, load_labels, load_split, rel  # noqa: E402
from distilroute.perturb import KINDS, perturb  # noqa: E402

MODELS = [
    "tfidf_lr_teacher",
    "tinybert_ft_teacher",
    "minilm_ft_teacher",
    "minilm_frozen_teacher",
    "tfidf_lr_aug_teacher",  # trained with typo1,typo3 copies: the fix (skipped if absent)
    "minilm_frozen_aug_teacher",
    "minilm_ft_aug_teacher",
    "distilbert_ft_teacher",
    "minilm_ft_gold",
    "tfidf_lr_gold",
]
THRESHOLD = 0.8  # the cascade threshold of 1b.2 / 6.5


def predict(router: students.Router, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    routed = [router.route(t) for t in texts]
    return np.array([r.intent for r in routed]), np.array([r.confidence for r in routed])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default=",".join(MODELS))
    args = ap.parse_args()

    test = load_split("test")
    gold = test.category.values
    texts = {"clean": test.text.tolist()} | {k: [perturb(t, k) for t in test.text] for k in KINDS}
    changed = {k: np.array(texts[k]) != np.array(texts["clean"]) for k in KINDS}

    # Students: every model on every variant, through the serving code path.
    rows, preds = [], {}
    for name in args.models.split(","):
        if name not in students.available():
            print(f"skip {name}: not under models/")
            continue
        router = students.load(name)
        t0 = time.time()
        row = {"model": name, "kinds": {}}
        for kind, xs in texts.items():
            pred, conf = predict(router, xs)
            preds[(name, kind)] = pred
            hit = pred == gold
            kept = conf >= THRESHOLD
            row["kinds"][kind] = {
                "acc": float(hit.mean()),
                "acc_changed": float(hit[changed[kind]].mean()) if kind != "clean" else None,
                "escalated": float((~kept).mean()),
                "kept_acc": float(hit[kept].mean()) if kept.any() else None,
            }
        rows.append(row)
        k = row["kinds"]
        print(
            f"{name:24} clean {k['clean']['acc']:.3f}  "
            + "  ".join(f"{kind} {k[kind]['acc'] - k['clean']['acc']:+.3f}" for kind in KINDS)
            + f"  ({time.time() - t0:.0f}s)"
        )

    # Teacher: on whatever noisy label files exist, vs the students on the same rows.
    teacher = {}
    clean_t = load_labels("test").teacher if (LABELS / "test.jsonl").exists() else None
    for kind in KINDS:
        if not (LABELS / f"test.perturb_{kind}.jsonl").exists() or clean_t is None:
            continue
        noisy = load_labels(f"test.perturb_{kind}").teacher
        idx = noisy.index.values
        entry = {
            "n": len(idx),
            "teacher_clean": float((clean_t.reindex(idx).values == gold[idx]).mean()),
            "teacher_noisy": float((noisy.values == gold[idx]).mean()),
            "students": {},
        }
        for r in rows:
            m = r["model"]
            entry["students"][m] = {
                "clean": float((preds[(m, "clean")][idx] == gold[idx]).mean()),
                "noisy": float((preds[(m, kind)][idx] == gold[idx]).mean()),
            }
        teacher[kind] = entry
        print(
            f"teacher on {kind} ({len(idx)} rows): "
            f"{entry['teacher_clean']:.3f} -> {entry['teacher_noisy']:.3f}"
        )

    out = {
        "threshold": THRESHOLD,
        "changed": {k: float(v.mean()) for k, v in changed.items()},
        "examples": {k: texts[k][:3] for k in texts},
        "students": rows,
        "teacher": teacher,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "robustness.json").write_text(json.dumps(out, indent=2))
    write_doc(out)
    print(f"-> {rel(RESULTS)}/robustness.json, {rel(DOCS)}/robustness.md")


def write_doc(out: dict) -> None:
    kinds = ["clean", *KINDS]
    lines = [
        "# Robustness to messy input",
        "",
        "_Generated by `scripts/robustness.py`. Every student re-scored on rule-based noisy "
        "copies of the full test split (`distilroute/perturb.py`), through the serving code "
        "path. Accuracy vs gold; the drop from clean in brackets._",
        "",
        "| noise | example | queries changed |",
        "|---|---|---:|",
    ]
    for k in kinds:
        ex = out["examples"][k][1].replace("|", "\\|")
        share = "—" if k == "clean" else f"{out['changed'][k]:.0%}"
        lines.append(f"| {k} | {ex} | {share} |")
    lines += [
        "",
        "## Accuracy",
        "",
        "| model | " + " | ".join(kinds) + " |",
        "|---|" + "---:|" * len(kinds),
    ]
    for r in out["students"]:
        k = r["kinds"]
        cells = [f"**{k['clean']['acc']:.3f}**"] + [
            f"{k[kind]['acc']:.3f} ({k[kind]['acc'] - k['clean']['acc']:+.3f})" for kind in KINDS
        ]
        lines.append(f"| {r['model']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Only the queries the noise changed (`slang` leaves many untouched, diluting the "
        "table above):",
        "",
        "| model | " + " | ".join(KINDS) + " |",
        "|---|" + "---:|" * len(KINDS),
    ]
    for r in out["students"]:
        k = r["kinds"]
        lines.append(
            f"| {r['model']} | "
            + " | ".join(f"{k[kind]['acc_changed']:.3f}" for kind in KINDS)
            + " |"
        )
    lines += [
        "",
        f"## What noise does to the cascade (threshold {out['threshold']})",
        "",
        "Share of queries escalated to the LLM · student accuracy on the ones it kept. If "
        "escalation rises while kept accuracy holds, the student knows it is confused and the "
        "cascade absorbs the noise.",
        "",
        "| model | " + " | ".join(kinds) + " |",
        "|---|" + "---:|" * len(kinds),
    ]
    for r in out["students"]:
        k = r["kinds"]
        lines.append(
            f"| {r['model']} | "
            + " | ".join(
                f"{k[kind]['escalated']:.0%} · {k[kind]['kept_acc']:.3f}" for kind in kinds
            )
            + " |"
        )
    if out["teacher"]:
        lines += [
            "",
            "## Student vs teacher on the same noisy rows",
            "",
            "A random subset per kind of noise was labelled by the teacher (same config as the "
            "clean labels). Clean → noisy accuracy on exactly those rows.",
            "",
            "| noise | rows | teacher | " + " | ".join(r["model"] for r in out["students"]) + " |",
            "|---|---:|---:|" + "---:|" * len(out["students"]),
        ]
        for kind, e in out["teacher"].items():
            st = e["students"]
            cells = [
                f"{st[r['model']]['clean']:.3f} → {st[r['model']]['noisy']:.3f}"
                for r in out["students"]
            ]
            lines.append(
                f"| {kind} | {e['n']} | {e['teacher_clean']:.3f} → {e['teacher_noisy']:.3f} | "
                + " | ".join(cells)
                + " |"
            )
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "robustness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
