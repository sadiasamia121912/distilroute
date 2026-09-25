"""How sure are the headline numbers? Paired bootstrap over the test split (roadmap 6.10).

    python scripts/bootstrap.py              # every headline system, 10,000 resamples

Every accuracy in this repo is measured on the same 3,080 test queries, so it carries sampling
noise: another 3,080 queries from the same inbox would give a slightly different number. This
resamples the test split with replacement and reports, per system:

- accuracy with a 95 % percentile interval;
- the difference to the teacher, **paired**: each resample scores both on the same queries,
  which cancels most of the noise the two share, so the interval on the difference is much
  narrower than the two intervals side by side would suggest;
- the share of resamples in which the system beats the teacher.

It covers test-set sampling only. Training randomness (seeds) is separate: ±0.3 pt for the
fine-tuned students (roadmap 1b.4), ± half-range over seeds in docs/correction.md.

Writes results/bootstrap.json and docs/bootstrap.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distilroute.calibration import apply_temperature  # noqa: E402
from distilroute.data import DOCS, RESULTS, load_labels, load_split, rel  # noqa: E402

STUDENTS = {  # run -> label, in README table order
    "minilm_ft_teacher": "MiniLM-L6 fine-tuned (fp32)",
    "minilm_frozen_teacher": "MiniLM-L6 frozen + LR",
    "distilbert_ft_teacher": "DistilBERT fine-tuned",
    "tinybert_ft_teacher": "TinyBERT fine-tuned",
    "tfidf_lr_teacher": "TF-IDF + LR",
    "minilm_frozen_aug_teacher": "MiniLM-L6 frozen, typo-augmented",
    "tfidf_lr_aug_teacher": "TF-IDF + LR, typo-augmented",
    "minilm_ft_gold": "MiniLM-L6 fine-tuned, human labels",
    "tfidf_lr_gold": "TF-IDF + LR, human labels",
}
CASCADE = ("minilm_ft_teacher", 0.8)  # README: MiniLM, escalate below 0.8 calibrated confidence
CORRECTION = {  # scripts/correct.py runs, seed 0
    "base": "frozen MiniLM, 0 human labels",
    "random_500": "+ 500 human checks, random rows",
    "add_new_500": "+ 500 new human-labelled rows",
    "disagreement_300": "+ 300 human checks, by disagreement",
    "disagreement_500": "+ 500 human checks, by disagreement",
    "disagreement_1000": "+ 1,000 human checks, by disagreement",
}


PAIRS = [  # (a, b, the README claim it checks)
    (
        "minilm_ft_teacher",
        "distilbert_ft_teacher",
        "MiniLM (22M) is at least as good as DistilBERT (67M)",
    ),
    ("minilm_ft_teacher", "minilm_frozen_teacher", "fine-tuned and frozen MiniLM are a tie"),
    ("minilm_ft_teacher_int8", "minilm_ft_teacher", "int8 costs about half a point"),
    (
        "minilm_frozen_aug_teacher",
        "minilm_frozen_teacher",
        "typo augmentation is free on clean text",
    ),
    ("cascade", "minilm_ft_teacher", "the cascade is better than its student alone"),
    ("correction_disagreement_500", "correction_random_500", "targeted checks beat random ones"),
    ("correction_disagreement_500", "correction_add_new_500", "fixing labels beats adding rows"),
]


def student(run: str) -> tuple[np.ndarray, np.ndarray]:
    """(predicted labels, calibrated confidence) from a run's saved test probabilities."""
    z = np.load(RESULTS / f"{run}_test_probs.npz", allow_pickle=False)
    meta = json.loads((RESULTS / f"{run}.json").read_text())
    proba = z["proba"]
    if meta.get("temperature"):
        proba = apply_temperature(proba, meta["temperature"])
    return z["classes"][proba.argmax(1)], proba.max(1)


def served_int8(texts: list[str]) -> np.ndarray | None:
    """The served model's predictions through the serving code path (int8 ONNX)."""
    from distilroute import students

    if "minilm_ft_teacher" not in students.available():
        return None
    router = students.load("minilm_ft_teacher")
    return np.array([router.route(t).intent for t in texts])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--no-int8", action="store_true", help="skip the served model (needs models/)")
    args = ap.parse_args()

    test = load_split("test")
    gold = test.category.values
    teacher = load_labels("test").teacher.reindex(test.index).values
    hits = {"teacher": ("LLM teacher (gpt-oss-120b)", teacher == gold)}

    for run, label in STUDENTS.items():
        if (RESULTS / f"{run}_test_probs.npz").exists():
            hits[run] = (label, student(run)[0] == gold)
    if not args.no_int8 and (pred := served_int8(test.text.tolist())) is not None:
        hits["minilm_ft_teacher_int8"] = ("MiniLM-L6 fine-tuned, int8 as served", pred == gold)
    run, thr = CASCADE
    pred, conf = student(run)
    mixed = np.where(conf < thr, teacher, pred)
    hits["cascade"] = (f"Cascade: MiniLM, escalate below {thr}", mixed == gold)
    corr_path = RESULTS / "correction_test_preds.npz"
    if corr_path.exists():
        z = np.load(corr_path, allow_pickle=False)
        for key, label in CORRECTION.items():
            if key in z:
                hits[f"correction_{key}"] = (label, z["classes"][z[key]] == gold)

    names = list(hits)
    h = np.stack([hits[k][1] for k in names]).astype("float64")  # (systems, queries)
    t = h[names.index("teacher")]
    n = h.shape[1]
    rng = np.random.default_rng(0)
    accs = []
    for start in range(0, args.resamples, 500):
        idx = rng.integers(0, n, (min(500, args.resamples - start), n))
        counts = np.stack([np.bincount(row, minlength=n) for row in idx]).astype("float64")
        accs.append(counts @ h.T / n)  # (resamples, systems); integer counts, so ties are exact
    accs = np.vstack(accs)
    # Every system is scored on the same resamples, so a column difference is a paired one.
    diffs = accs - accs[:, [names.index("teacher")]]

    rows = []
    for i, k in enumerate(names):
        lo, hi = np.percentile(accs[:, i], [2.5, 97.5])
        dlo, dhi = np.percentile(diffs[:, i], [2.5, 97.5])
        rows.append(
            {
                "system": k,
                "label": hits[k][0],
                "acc": float(h[i].mean()),
                "ci": [float(lo), float(hi)],
                "vs_teacher": float(h[i].mean() - t.mean()),
                "vs_teacher_ci": [float(dlo), float(dhi)],
                "p_beats_teacher": float((diffs[:, i] > 0).mean()),
            }
        )
        print(
            f"{hits[k][0]:44} {h[i].mean():.3f} [{lo:.3f}, {hi:.3f}]   "
            f"vs teacher {(h[i].mean() - t.mean()) * 100:+.1f} pt "
            f"[{dlo * 100:+.1f}, {dhi * 100:+.1f}]  beats it in {(diffs[:, i] > 0).mean():.1%}"
        )

    pairs = []
    for a, b, claim in PAIRS:
        if a not in names or b not in names:
            continue
        d = accs[:, names.index(a)] - accs[:, names.index(b)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        pairs.append(
            {
                "a": a,
                "b": b,
                "claim": claim,
                "diff": float(h[names.index(a)].mean() - h[names.index(b)].mean()),
                "ci": [float(lo), float(hi)],
                "p_a_better": float((d > 0).mean()),
            }
        )
        print(f"  {claim:58} {pairs[-1]['diff'] * 100:+.1f} pt [{lo * 100:+.1f}, {hi * 100:+.1f}]")

    out = {"n_test": int(n), "resamples": args.resamples, "rows": rows, "pairs": pairs}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "bootstrap.json").write_text(json.dumps(out, indent=2) + "\n")
    write_doc(out)
    print(f"-> {rel(RESULTS)}/bootstrap.json, {rel(DOCS)}/bootstrap.md")


def write_doc(out: dict) -> None:
    pt = lambda x: f"{x * 100:+.1f}"  # noqa: E731
    lines = [
        "# How sure are the numbers? Paired bootstrap on the test split",
        "",
        f"_Generated by `scripts/bootstrap.py`: {out['resamples']:,} resamples of the "
        f"{out['n_test']:,} test queries, with replacement. Intervals are 95 % percentile "
        "intervals. The difference to the teacher is paired (both scored on the same resample), "
        "so it is far tighter than the two accuracy intervals side by side. Test-set sampling "
        "only: seed-to-seed training noise is separate (±0.3 pt for the fine-tuned students)._",
        "",
        "| system | accuracy | 95 % interval | vs teacher, pt | 95 % interval, pt "
        "| resamples where it beats the teacher |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in out["rows"]:
        same = r["system"] == "teacher"
        lines.append(
            f"| {r['label']} | {r['acc']:.3f} | {r['ci'][0]:.3f} – {r['ci'][1]:.3f} | "
            + (
                "— | — | —"
                if same
                else f"{pt(r['vs_teacher'])} | "
                f"{pt(r['vs_teacher_ci'][0])} to {pt(r['vs_teacher_ci'][1])} | "
                f"{r['p_beats_teacher']:.1%}"
            )
            + " |"
        )
    labels = {r["system"]: r["label"] for r in out["rows"]}
    lines += [
        "",
        "## Head to head: the README's other comparisons",
        "",
        "Paired the same way. A claim of *better* holds when the whole interval is above zero; "
        "a claim of *a tie* holds when the interval is narrow and straddles zero.",
        "",
        "| claim | A | B | A − B, pt | 95 % interval, pt | resamples where A is better |",
        "|---|---|---|---:|---:|---:|",
    ]
    for p in out["pairs"]:
        lines.append(
            f"| {p['claim']} | {labels[p['a']]} | {labels[p['b']]} | {pt(p['diff'])} | "
            f"{pt(p['ci'][0])} to {pt(p['ci'][1])} | {p['p_a_better']:.1%} |"
        )
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "bootstrap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
