"""Build docs/results.md from every run in results/.

    python scripts/evaluate.py

A run is `results/<name>.json` (metrics) + `results/<name>_test_probs.npz` (test-split
probabilities, see `save_run` in baseline.py). This script never retrains anything; it
recomputes what it can from the probabilities so all runs are scored the same way:

- accuracy / macro-F1 vs gold, recomputed from the probabilities
- agreement with the teacher, on the test rows the teacher has labelled so far
- expected calibration error (ECE, 10 bins) — whether "90 % confident" means 90 % right —
  raw and after temperature scaling (T fit by the training script on held-out training-pool
  rows, see `distilroute.calibration`); everything below uses the calibrated confidence
- the cascade (roadmap 1b.2), two ways: escalate the least-confident X % of queries to the
  teacher (a budget), or escalate everything below a confidence threshold (a policy). The
  threshold table also gives coverage and the student's accuracy on what it kept, which
  needs no teacher labels.
- the data-efficiency curves from `scripts/data_curve.py` (results/curves/), if any.
- cost per 1M requests from `scripts/cost.py` (results/cost.json), if present.

Latency comes from results/latency.json (`scripts/bench_latency.py`, this laptop's CPU) when
a run has an entry there, else from what the training script recorded (Colab's CPU for the
fine-tuned models).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distilroute.calibration import apply_temperature, ece  # noqa: E402
from distilroute.data import DOCS, LABELS, RESULTS, load_labels, load_split  # noqa: E402

ESCALATE = [0.0, 0.05, 0.10, 0.20, 0.30]
THRESHOLDS = [0.5, 0.7, 0.8, 0.9, 0.95]


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


def cost_lines() -> list[str]:
    path = RESULTS / "cost.json"
    if not path.exists():
        return []
    c = json.loads(path.read_text())
    vm, teacher = c["prices"]["vm"], c["prices"]["teacher"]
    lines = [
        "",
        "## Cost per 1M requests",
        "",
        f"Teacher at the provider's **paid** list price ({teacher['model']}: "
        f"${teacher['usd_per_1m_in']} in / ${teacher['usd_per_1m_out']} out per 1M tokens) — "
        "the free tier we labelled with is rate-capped and not a production option. Tokens per "
        f"query measured on the final config (~{c['tokens']['prompt_fixed']:,} fixed prompt + "
        f"{c['tokens']['prompt_per_query']} per query in, {c['tokens']['output_per_query']} out). "
        f"Students: mean in-process latency × a {vm['name']} at ${vm['usd_per_hour']}/h, one "
        "request at a time on one core — an upper bound; on hardware you already own it is $0. "
        "Prices checked 2026-09-18 (`scripts/cost.py`).",
        "",
        "| model | mode | per query | $ per 1M requests |",
        "|---|---|---:|---:|",
    ]
    for r in c["rows"].values():
        if "tokens_in_per_query" in r:
            per = f"{r['tokens_in_per_query']:,.0f} + {r['tokens_out_per_query']} tokens"
            mode = r["mode"]
        else:
            per = f"{r['cpu_seconds_per_query'] * 1000:.1f} ms CPU"
            mode = vm["name"].split(" (")[0]
        lines.append(f"| {r['model']} | {mode} | {per} | **{r['usd_per_1m']:,.2f}** |")
    return lines


def threshold_cascade(pred, conf, gold, teacher: pd.Series | None) -> list[dict]:
    """For each confidence threshold: how much is escalated, how good the student is on the
    rest, and the accuracy of the mixed system (None until the teacher has labelled the
    escalated rows)."""
    out = []
    for thr in THRESHOLDS:
        esc = conf < thr
        kept_acc = float((pred[~esc] == gold[~esc]).mean()) if (~esc).any() else None
        mixed = None
        if teacher is not None and not esc.any():
            mixed = float((pred == gold).mean())
        elif teacher is not None:
            t = teacher.reindex(np.flatnonzero(esc))
            if not t.isna().any():
                m = pred.copy()
                m[esc] = t.values
                mixed = float((m == gold).mean())
        out.append(
            {"thr": thr, "escalated": float(esc.mean()), "kept_acc": kept_acc, "mixed": mixed}
        )
    return out


def main() -> None:
    test = load_split("test")
    gold = test.category.values
    teacher = None
    if (LABELS / "test.jsonl").exists():
        t = load_labels("test")
        teacher = t.teacher
        t_cov = len(t) / len(test)

    lat_path = RESULTS / "latency.json"
    latency = json.loads(lat_path.read_text()) if lat_path.exists() else {}

    rows = []
    for meta_path in sorted(RESULTS.glob("*.json")):
        name = meta_path.stem
        probs_path = RESULTS / f"{name}_test_probs.npz"
        meta = json.loads(meta_path.read_text())
        if not probs_path.exists():
            continue
        lat = latency.get(name, meta)  # bench on this laptop beats whatever the trainer saw
        z = np.load(probs_path, allow_pickle=False)
        classes, proba = z["classes"], z["proba"]
        pred = classes[proba.argmax(axis=1)]
        hit = pred == gold
        conf_raw = proba.max(axis=1)
        temp = meta.get("temperature")
        conf = apply_temperature(proba, temp).max(axis=1) if temp else conf_raw
        row = {
            "name": name,
            "model": meta["model"],
            "params": meta.get("params"),
            "trained_on": meta["trained_on"],
            "n_train": meta["n_train"],
            "accuracy": accuracy_score(gold, pred),
            "macro_f1": f1_score(gold, pred, average="macro"),
            "p50_ms": lat.get("p50_ms"),
            "p95_ms": lat.get("p95_ms"),
            "lat_here": name in latency,
            "ece_raw": ece(conf_raw, hit),
            "ece": ece(conf, hit),
            "temperature": temp,
            "agree": None,
            "cascade": cascade(pred, conf, gold, teacher),
            "thresholds": threshold_cascade(pred, conf, gold, teacher),
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
            f"Teacher (`gpt-oss-120b`, zero-shot): accuracy **{t_acc:.3f}** on "
            + (
                f"the full {len(t):,}-query test split."
                if t_cov >= 1
                else f"the {len(t):,} test queries labelled so far ({t_cov:.0%} — intent-sorted, "
                "so partial coverage is not a random sample). Agreement below is measured on "
                "those rows."
            ),
            "",
        ]
    lines += [
        "| model | params | trained on | n train | acc vs gold | macro-F1 | agree w/ teacher "
        "| ECE raw → calibrated (T) | p50 / p95 ms |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        agree = "—" if r["agree"] is None else f"{r['agree']:.3f}"
        params = "—" if not r["params"] else f"{r['params'] / 1e6:.0f}M"
        lat = "—" if r["p50_ms"] is None else f"{r['p50_ms']:.1f} / {r['p95_ms']:.1f}"
        lat += "" if r["lat_here"] or r["p50_ms"] is None else " †"
        cal = (
            f"{r['ece_raw']:.3f} → {r['ece']:.3f} (T={r['temperature']:.2f})"
            if r["temperature"]
            else f"{r['ece_raw']:.3f} (uncalibrated)"
        )
        lines.append(
            f"| {r['model']} | {params} | {r['trained_on']} | {r['n_train']:,} "
            f"| **{r['accuracy']:.3f}** | {r['macro_f1']:.3f} | {agree} | {cal} | {lat} |"
        )

    if latency:
        hosts = sorted({v["host"] for v in latency.values()})
        lines += [
            "",
            f"Latency: single query, in-process, CPU of `{', '.join(hosts)}` "
            "(`scripts/bench_latency.py`); † = as recorded by the training script instead "
            "(possibly another machine).",
        ]
        if "teacher" in latency:
            t = latency["teacher"]
            lines.append(
                f"Teacher through the API, end to end: p50 **{t['p50_ms']:,.0f} ms** / "
                f"p95 {t['p95_ms']:,.0f} ms over {t['n']} queries."
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

    lines += [
        "",
        "## Cascade by confidence threshold — the policy a service would actually run",
        "",
        "Escalate a query when the student's *calibrated* confidence is below the threshold. "
        "Per cell: share of queries escalated · student accuracy on the ones it kept · accuracy "
        "of the mixed system (— until the teacher has labelled the escalated rows). Calibration "
        "is what makes the threshold mean something: without it 0.9 is just a number.",
        "",
        "| model | trained on | " + " | ".join(f"< {t}" for t in THRESHOLDS) + " |",
        "|---|---|" + "---:|" * len(THRESHOLDS),
    ]
    for r in rows:
        cells = []
        for t in r["thresholds"]:
            kept = "—" if t["kept_acc"] is None else f"{t['kept_acc']:.3f}"
            mixed = "—" if t["mixed"] is None else f"{t['mixed']:.3f}"
            cells.append(f"{t['escalated']:.0%} · {kept} · {mixed}")
        lines.append(f"| {r['model']} | {r['trained_on']} | " + " | ".join(cells) + " |")

    lines += curve_lines()
    lines += cost_lines()

    DOCS.mkdir(exist_ok=True)
    out = DOCS / "results.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(lines[4:]))
    print(f"\n-> {out.relative_to(out.parents[1])}")


if __name__ == "__main__":
    main()
