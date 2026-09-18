"""Build docs/results.md from every run in results/.

    python scripts/evaluate.py

A run is `results/<name>.json` (metrics) + `results/<name>_test_probs.npz` (test-split
probabilities, see `save_run` in baseline.py). This script never retrains anything; it
recomputes what it can from the probabilities so all runs are scored the same way:

- accuracy / macro-F1 vs gold, recomputed from the probabilities
- agreement with the teacher, on the test rows the teacher has labelled so far
- expected calibration error (ECE, 10 bins) — whether "90 % confident" means 90 % right
- a cascade preview: route the least-confident X % of queries to the teacher instead, and
  report the accuracy of the mixed system. This is the deployment pattern (roadmap 1b.2).
- the data-efficiency curves from `scripts/data_curve.py` (results/curves/), if any.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distilroute.data import DOCS, LABELS, RESULTS, load_labels, load_split  # noqa: E402

ESCALATE = [0.0, 0.05, 0.10, 0.20, 0.30]


def ece(conf: np.ndarray, hit: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(total)


def cascade(pred, conf, gold, teacher: pd.Series | None) -> dict[float, float | None]:
    """Accuracy when the least-confident fraction is answered by the teacher instead."""
    out: dict[float, float | None] = {}
    order = np.argsort(conf)  # least confident first
    for frac in ESCALATE:
        k = int(round(frac * len(conf)))
        mixed = pred.copy()
        if k:
            if teacher is None:
                out[frac] = None
                continue
            idx = order[:k]
            t = teacher.reindex(idx)
            if t.isna().any():  # teacher has not labelled those rows yet
                out[frac] = None
                continue
            mixed[idx] = t.values
        out[frac] = float((mixed == gold).mean())
    return out


def curve_lines() -> list[str]:
    """Data-efficiency table: accuracy at each training-set size, mean ± half-range over seeds."""
    curves = [json.loads(p.read_text()) for p in sorted((RESULTS / "curves").glob("*.json"))]
    if not curves:
        return []
    sizes = sorted({r["n"] for c in curves for r in c["rows"]})
    lines = [
        "",
        "## Data efficiency — accuracy vs. number of training labels",
        "",
        "How many labelled tickets does a student need? Each cell is accuracy vs gold on the "
        "full test split, training on N random rows of the pool, mean ± half-range over "
        "seeds (`scripts/data_curve.py`). With teacher labels, N is the number of LLM calls' "
        "worth of data.",
        "",
        "| student | trained on | " + " | ".join(f"{n:,}" for n in sizes) + " |",
        "|---|---|" + "---:|" * len(sizes),
    ]
    for c in curves:
        cells = []
        for n in sizes:
            accs = [r["accuracy"] for r in c["rows"] if r["n"] == n]
            if not accs:
                cells.append("—")
            elif len(accs) == 1:
                cells.append(f"{accs[0]:.3f}")
            else:
                cells.append(f"{np.mean(accs):.3f} ± {(max(accs) - min(accs)) / 2:.3f}")
        lines.append(f"| {c['student']} | {c['trained_on']} | " + " | ".join(cells) + " |")
    return lines


def main() -> None:
    test = load_split("test")
    gold = test.category.values
    teacher = None
    if (LABELS / "test.jsonl").exists():
        t = load_labels("test")
        teacher = t.teacher
        t_cov = len(t) / len(test)

    rows = []
    for meta_path in sorted(RESULTS.glob("*.json")):
        name = meta_path.stem
        probs_path = RESULTS / f"{name}_test_probs.npz"
        meta = json.loads(meta_path.read_text())
        if not probs_path.exists():
            continue
        z = np.load(probs_path, allow_pickle=False)
        classes, proba = z["classes"], z["proba"]
        pred = classes[proba.argmax(axis=1)]
        conf = proba.max(axis=1)
        hit = pred == gold
        row = {
            "name": name,
            "model": meta["model"],
            "params": meta.get("params"),
            "trained_on": meta["trained_on"],
            "n_train": meta["n_train"],
            "accuracy": accuracy_score(gold, pred),
            "macro_f1": f1_score(gold, pred, average="macro"),
            "p50_ms": meta.get("p50_ms"),
            "p95_ms": meta.get("p95_ms"),
            "ece": ece(conf, hit),
            "agree": None,
            "cascade": cascade(pred, conf, gold, teacher),
        }
        if teacher is not None:
            common = teacher.dropna()
            row["agree"] = float((pd.Series(pred).reindex(common.index) == common).mean())
        rows.append(row)

    lines = [
        "# Results",
        "",
        "_Generated by `scripts/evaluate.py` from `results/`. All scores are against human "
        "labels on the 3,080-query test split. **gold**-trained rows are the supervised "
        "reference; **teacher**-trained rows are the distilled models._",
        "",
    ]
    if teacher is not None:
        t_acc = float((t.teacher == t.gold).mean())
        lines += [
            f"Teacher (`gpt-oss-120b`, zero-shot): accuracy **{t_acc:.3f}** on the "
            f"{len(t):,} test queries labelled so far ({t_cov:.0%} — intent-sorted, so partial "
            "coverage is not a random sample). Agreement below is measured on those rows.",
            "",
        ]
    lines += [
        "| model | params | trained on | n train | acc vs gold | macro-F1 | agree w/ teacher "
        "| ECE | p50 / p95 ms |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        agree = "—" if r["agree"] is None else f"{r['agree']:.3f}"
        params = "—" if not r["params"] else f"{r['params'] / 1e6:.0f}M"
        lat = "—" if r["p50_ms"] is None else f"{r['p50_ms']:.1f} / {r['p95_ms']:.1f}"
        lines.append(
            f"| {r['model']} | {params} | {r['trained_on']} | {r['n_train']:,} "
            f"| **{r['accuracy']:.3f}** | {r['macro_f1']:.3f} | {agree} | {r['ece']:.3f} | {lat} |"
        )

    lines += [
        "",
        "## Cascade preview — escalate the least-confident queries to the teacher",
        "",
        "Accuracy of the mixed system when the student's least-confident X % of test queries "
        "are answered by the teacher instead. Needs teacher labels for those rows; blank until "
        "the test split is fully labelled.",
        "",
        "| model | trained on | " + " | ".join(f"{int(f * 100)} %" for f in ESCALATE) + " |",
        "|---|---|" + "---:|" * len(ESCALATE),
    ]
    for r in rows:
        cells = ["—" if v is None else f"{v:.3f}" for v in r["cascade"].values()]
        lines.append(f"| {r['model']} | {r['trained_on']} | " + " | ".join(cells) + " |")

    lines += curve_lines()

    DOCS.mkdir(exist_ok=True)
    out = DOCS / "results.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(lines[4:]))
    print(f"\n-> {out.relative_to(out.parents[1])}")


if __name__ == "__main__":
    main()
