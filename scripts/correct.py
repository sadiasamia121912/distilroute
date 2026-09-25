"""Where should a small budget of human labels go? (roadmap 6.9)

    python scripts/correct.py                          # budgets 100..1,000, all strategies

A team that distils an LLM can usually afford *some* human labelling: an afternoon of an
annotator's time, a few hundred rows. The student inherits the teacher's mistakes (0.848 on
3,000 teacher labels against the teacher's 0.867), so the question is how to spend those rows.
A human is simulated with the gold label; each strategy gets the same budget:

- **random**       — check random teacher-labelled rows, fix the ones that are wrong.
- **disagreement** — check the rows whose teacher label the student believes least
                     (cross-validated, out-of-fold probability of the teacher's label: the
                     confident-learning score behind 6.1's filter, as a ranking, not a cut).
- **unsure**       — check the rows the student is least sure about (lowest out-of-fold top
                     probability): classic uncertainty sampling.
- **add_new**      — leave the teacher labels alone; label that many *new* rows by hand and add
                     them to the pool. The alternative use of the same hours.

Checking a row costs one human label whether or not it turns out to be wrong, so a strategy is
good when it points the human at teacher errors. Reported per budget: test accuracy (vs gold,
as everywhere), the share of checked rows that were really wrong, and the teacher errors fixed.
Same frozen MiniLM embeddings and logistic head as `denoise.py`, so budget 0 is its `base` run.

Writes results/correction.json and docs/correction.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distilroute.data import (  # noqa: E402
    DOCS,
    RESULTS,
    categories,
    load_labels,
    load_split,
    rel,
    teacher_train_labels,
)
from distilroute.runs import CALIB_FRAC  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_curve import minilm_embeddings  # noqa: E402
from denoise import fit_head, oof_proba, proba, soft_targets  # noqa: E402

BUDGETS = [100, 200, 300, 500, 1000]
STRATEGIES = ["random", "disagreement", "unsure", "add_new"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--budgets", default=",".join(map(str, BUDGETS)))
    ap.add_argument("--seeds", type=int, default=3, help="for the random strategies")
    args = ap.parse_args()
    budgets = [int(b) for b in args.budgets.split(",")]

    classes = categories()
    pos = {c: i for i, c in enumerate(classes)}
    full = load_split("train")
    train = teacher_train_labels()
    test = load_split("test")
    gold_test = test.category.values
    teacher_acc = float(
        (load_labels("test").teacher.reindex(test.index).values == gold_test).mean()
    )
    x_all = minilm_embeddings("train", full.text.tolist())
    x_test = minilm_embeddings("test", test.text.tolist())

    idx = train.index.values
    teacher_y = np.array([pos[y] for y in train.y])
    gold_y = np.array([pos[y] for y in full.category.values])
    wrong = teacher_y != gold_y[idx]
    unlabelled = np.setdiff1d(np.arange(len(full)), idx)
    print(
        f"{len(idx):,} teacher-labelled rows ({wrong.mean():.1%} wrong), "
        f"{len(unlabelled):,} unlabelled for add_new"
    )

    # The same calibration hold-out as denoise.py's runs, so budget 0 reproduces `base`.
    rng = np.random.default_rng(0)
    cal = rng.choice(len(idx), max(77, int(CALIB_FRAC * len(idx))), replace=False)
    fit = np.setdiff1d(np.arange(len(idx)), cal)

    def score(rows: np.ndarray, y: np.ndarray) -> float:
        head = fit_head(x_all[rows], np.eye(len(classes), dtype="float32")[y])
        return float(accuracy_score(gold_test, np.array(classes)[proba(head, x_test).argmax(1)]))

    base = score(idx[fit], teacher_y[fit])
    print(f"budget 0: {base:.4f} (teacher {teacher_acc:.4f})")

    # Rankings over the fit rows, from out-of-fold predictions: no row is scored by a head
    # that saw it. Lower = check first.
    oof = oof_proba(
        x_all[idx[fit]],
        soft_targets(train.ranked.iloc[fit], train.y.iloc[fit], classes, 0.0),
        teacher_y[fit],
    )
    ranking = {
        "disagreement": np.argsort(oof[np.arange(len(fit)), teacher_y[fit]], kind="stable"),
        "unsure": np.argsort(oof.max(1), kind="stable"),
    }

    def corrected(checked: np.ndarray) -> float:
        """Accuracy after a human checks `checked` (positions in `fit`) and fixes what's wrong."""
        y = teacher_y[fit].copy()
        y[checked] = gold_y[idx[fit][checked]]
        return score(idx[fit], y)

    runs = []
    for b in budgets:
        for strat in STRATEGIES:
            accs, hit_rates, fixed = [], [], []
            for seed in range(args.seeds if strat in ("random", "add_new") else 1):
                r = np.random.default_rng(100 + seed)
                if strat == "add_new":
                    new = r.choice(unlabelled, b, replace=False)
                    rows = np.concatenate([idx[fit], new])
                    accs.append(score(rows, np.concatenate([teacher_y[fit], gold_y[new]])))
                    continue
                checked = (
                    r.choice(len(fit), b, replace=False)
                    if strat == "random"
                    else ranking[strat][:b]
                )
                w = wrong[fit][checked]
                accs.append(corrected(checked))
                hit_rates.append(float(w.mean()))
                fixed.append(int(w.sum()))
            run = {
                "budget": b,
                "strategy": strat,
                "acc": float(np.mean(accs)),
                "acc_pm": float((max(accs) - min(accs)) / 2),
                "hit_rate": float(np.mean(hit_rates)) if hit_rates else None,
                "errors_fixed": float(np.mean(fixed)) if fixed else None,
            }
            runs.append(run)
            hr = "" if run["hit_rate"] is None else f"  {run['hit_rate']:.0%} were wrong"
            print(f"  {b:>5,} {strat:13} {run['acc']:.4f} ± {run['acc_pm']:.4f}{hr}")

    out = {
        "pool": int(len(idx)),
        "teacher_error_rate": float(wrong.mean()),
        "teacher_errors_in_fit": int(wrong[fit].sum()),
        "teacher": teacher_acc,
        "base": base,
        "seeds": args.seeds,
        "runs": runs,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "correction.json").write_text(json.dumps(out, indent=2) + "\n")
    write_doc(out)
    print(f"-> {rel(RESULTS)}/correction.json, {rel(DOCS)}/correction.md")


def write_doc(out: dict) -> None:
    budgets = sorted({r["budget"] for r in out["runs"]})
    get = {(r["budget"], r["strategy"]): r for r in out["runs"]}

    def cell(r: dict) -> str:
        s = f"{r['acc']:.3f}"
        if r["acc_pm"]:
            s += f" ± {r['acc_pm']:.3f}"
        return f"**{s}**" if r["acc"] > out["teacher"] else s

    lines = [
        "# Human corrections: where do a few human labels go?",
        "",
        "_Generated by `scripts/correct.py`. Frozen MiniLM student on "
        f"{out['pool']:,} teacher labels ({out['teacher_error_rate']:.1%} of them wrong); a "
        "human is simulated with the gold label. Accuracy vs gold on the full test split; "
        f"random strategies are the mean ± half-range over {out['seeds']} seeds. **Bold** beats "
        f"the teacher ({out['teacher']:.3f})._",
        "",
        f"Budget 0 (teacher labels only): **{out['base']:.3f}**.",
        "",
        "## Accuracy by budget",
        "",
        "| human labels | " + " | ".join(STRATEGIES) + " |",
        "|---:|" + "---:|" * len(STRATEGIES),
    ]
    for b in budgets:
        lines.append(f"| {b:,} | " + " | ".join(cell(get[(b, s)]) for s in STRATEGIES) + " |")
    lines += [
        "",
        "## How well each strategy finds teacher errors",
        "",
        "Share of checked rows whose teacher label was really wrong (the base rate is "
        f"{out['teacher_error_rate']:.1%}), and teacher errors fixed, of "
        f"{out['teacher_errors_in_fit']:,} in the pool.",
        "",
        "| human labels | " + " | ".join(s for s in STRATEGIES if s != "add_new") + " |",
        "|---:|" + "---:|" * (len(STRATEGIES) - 1),
    ]
    for b in budgets:
        cells = [
            f"{get[(b, s)]['hit_rate']:.0%} · {get[(b, s)]['errors_fixed']:.0f}"
            for s in STRATEGIES
            if s != "add_new"
        ]
        lines.append(f"| {b:,} | " + " | ".join(cells) + " |")
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "correction.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
