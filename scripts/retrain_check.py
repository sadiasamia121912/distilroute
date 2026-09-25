"""Roadmap 1.6 checkpoint: do more teacher labels close the student's gap to its teacher?

    python scripts/retrain_check.py --rows 5000     # a checkpoint while labelling continues
    python scripts/retrain_check.py                 # every labelled row (10,003 when done)

Steps 1-2 of the 1.6 retraining plan in one command, at N teacher-labelled rows:

1. Check the new labels (no training): unparsed rate and teacher accuracy vs gold on rows
   3,001..N against the first 3,000. Gold on train is read for this report only.
2. CPU students with DISTILROUTE_TEACHER_ROWS=N: TF-IDF, frozen MiniLM, the denoise variants
   (base, soft, filter), the teacher data curve, then evaluate.py.

Everything goes to results/teacher<N>/, docs/teacher<N>/ and models/teacher<N>/, never over the
published 3,000-row runs. The summary compares N with 3,000 and with the teacher (0.867).
Fine-tuned students (Colab) and the serve decision are steps 3-4 of the plan, after 10,003.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from distilroute.data import DEFAULT_TEACHER_ROWS, LABELS, load_labels, rel  # noqa: E402

RESULTS = ROOT / "results"  # the published runs, whatever DISTILROUTE_TEACHER_ROWS says
TEACHER_ACC = 0.8672  # gpt-oss-120b on the shuffled test split (docs/teacher.md)
STUDENTS = ["tfidf_lr_teacher", "minilm_frozen_teacher"]
DENOISE = ["base", "soft", "filter"]


def label_order() -> list[int]:
    """Row ids in train.jsonl order: the labeller's seed-0 shuffle, so any prefix is random."""
    path = LABELS / "train.jsonl"
    return [json.loads(line)["idx"] for line in path.open(encoding="utf-8") if line.strip()]


def check_labels(n: int) -> dict:
    lab = load_labels("train")
    order = label_order()
    parts = {
        "first 3,000": order[:DEFAULT_TEACHER_ROWS],
        f"3,001-{n:,}": order[DEFAULT_TEACHER_ROWS:n],
    }
    out = {}
    for name, ids in parts.items():
        if not ids:
            continue
        d = lab.loc[ids]
        parsed = d[d.teacher.notna()]
        out[name] = {
            "rows": len(d),
            "unparsed": round(float(d.teacher.isna().mean()), 4),
            "teacher_vs_gold": round(float((parsed.teacher == parsed.gold).mean()), 4),
            "top3_hit": round(
                float(sum(g in r for g, r in zip(d.gold, d.ranked, strict=True))) / len(d), 4
            ),
        }
    return out


def run(cmd: list[str], n: int) -> None:
    print(f"\n$ DISTILROUTE_TEACHER_ROWS={n} python {' '.join(cmd)}", flush=True)
    env = {**os.environ, "DISTILROUTE_TEACHER_ROWS": str(n)}
    subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env, check=True)


def acc(path: Path) -> float | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))["accuracy"]
    except (OSError, KeyError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--rows", type=int, default=None, help="teacher rows to train on (default: all)"
    )
    ap.add_argument("--skip-train", action="store_true", help="step 1 and the summary only")
    args = ap.parse_args()
    have = len(label_order())
    n = args.rows or have
    if n > have:
        sys.exit(f"train.jsonl has {have:,} labelled rows; --rows {n:,} is more than that")
    if n == DEFAULT_TEACHER_ROWS:
        sys.exit(f"{n:,} rows is the published run; nothing to check")

    print(f"1.6 checkpoint at {n:,} of {have:,} labelled rows")
    labels = check_labels(n)
    for name, r in labels.items():
        print(
            f"  labels {name:>14}: {r['rows']:>6,} rows, unparsed {r['unparsed']:.1%}, "
            f"teacher vs gold {r['teacher_vs_gold']:.3f}, top-3 {r['top3_hit']:.3f}"
        )

    out_dir = RESULTS / f"teacher{n}"
    if not args.skip_train:
        run(["scripts/baseline.py", "--labels", "teacher"], n)
        run(["scripts/setfit_student.py", "--mode", "frozen", "--labels", "teacher"], n)
        run(["scripts/denoise.py", "--variants", ",".join(DENOISE)], n)
        run(["scripts/data_curve.py", "--labels", "teacher"], n)
        run(["scripts/evaluate.py"], n)

    rows = []
    for run_name in STUDENTS + [f"minilm_denoise_{v}" for v in DENOISE]:
        before, after = acc(RESULTS / f"{run_name}.json"), acc(out_dir / f"{run_name}.json")
        rows.append({"run": run_name, "at_3000": before, f"at_{n}": after})
    summary = {"rows": n, "teacher": TEACHER_ACC, "labels": labels, "students": rows}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoint.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    f3 = lambda x: "  —  " if x is None else f"{x:.3f}"  # noqa: E731
    print(f"\n{'student':32} {'3,000':>7} {f'{n:,}':>7}  change  gap to teacher {TEACHER_ACC:.3f}")
    for r in rows:
        a, b = r["at_3000"], r[f"at_{n}"]
        delta = "" if a is None or b is None else f"{(b - a) * 100:+.1f} pt"
        gap = "" if b is None else f"{(b - TEACHER_ACC) * 100:+.1f} pt"
        print(f"{r['run']:32} {f3(a):>7} {f3(b):>7}  {delta:>7}  {gap:>7}")
    print(f"\n-> {rel(out_dir / 'checkpoint.json')}")


if __name__ == "__main__":
    main()
