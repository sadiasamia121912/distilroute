"""Can the student beat the teacher that labelled it? (roadmap 6.1)

    python scripts/denoise.py                      # every variant, frozen MiniLM
    python scripts/denoise.py --variants base,all  # just two of them

The distilled student reproduces its teacher, noise included (0.848 on 3,000 teacher labels,
teacher 0.867). Three levers try to beat that **without ever seeing a human label**:

- **soft**  — train on the teacher's ranked top-3 (weights 1, 1/2, 1/3, blended with the hard
  label by `--soft-alpha`) instead of its top-1. The second guess is right often enough to be
  worth something, and it says which intents the teacher found confusable.
- **filter** — cross-validated out-of-fold predictions on the training pool; where the student
  confidently disagrees with the teacher (teacher's label given < `--filter-p`, student's own
  pick above `--filter-q`), the label is probably wrong. Drop those rows. Confident-learning
  in miniature (roadmap 1b.5, the tabaudit idea).
- **filter_self** — the two that help, combined; **all** adds the soft target on top.
- **self**  — pseudo-label the 7,003 train rows the teacher never saw, keep those above
  `--self-p`, retrain on teacher labels + pseudo-labels. Free extra data, and the standard way
  to exceed a noisy teacher. The threshold is applied to *calibrated* probabilities (the head
  is under-confident, T ≈ 0.7; on raw probabilities a 0.9 cut keeps 379 rows instead of ~3,000).

Every variant uses the same frozen MiniLM embeddings (cached) and the same logistic head as
`setfit_student.py --mode frozen`, so the only thing that changes is the training target and
these numbers sit beside the rest of the table. A soft target reaches that head as one weighted
copy of the row per non-zero class, which is what a soft cross-entropy is.

Accuracy is always vs **gold on test**. Gold on train is read only to report what the filter
actually caught — a diagnostic, printed, never fed to the model; `--no-diagnostics` turns even
that off.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distilroute.calibration import apply_temperature, fit_temperature  # noqa: E402
from distilroute.data import categories, load_split, teacher_train_labels  # noqa: E402
from distilroute.runs import CALIB_FRAC, save_run  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_curve import minilm_embeddings  # noqa: E402

RANK_W = np.array([1.0, 0.5, 1 / 3])
VARIANTS = ["base", "soft", "filter", "self", "filter_self", "all"]


def soft_targets(ranked: pd.Series, y: pd.Series, classes: list[str], alpha: float) -> np.ndarray:
    """(n, 77): (1-alpha) on the teacher's top-1, alpha spread over its ranked top-3."""
    pos = {c: i for i, c in enumerate(classes)}
    t = np.zeros((len(y), len(classes)), dtype="float32")
    for row, (top1, rank) in enumerate(zip(y, ranked, strict=True)):
        t[row, pos[top1]] = 1.0
        known = [r for r in list(rank)[:3] if r in pos]
        if alpha and len(known) > 1:
            w = RANK_W[: len(known)] / RANK_W[: len(known)].sum()
            t[row] *= 1 - alpha
            for r, wi in zip(known, w, strict=True):
                t[row, pos[r]] += alpha * wi
    return t


def fit_head(x: np.ndarray, target: np.ndarray, min_w: float = 1e-3) -> LogisticRegression:
    """The project's frozen-student head (LR, C=10), fitted to a target *distribution*.

    Each row is repeated once per class it puts mass on, with that mass as the sample weight —
    identical to minimising soft cross-entropy, and identical to the hard fit when the target
    is one-hot, so `base` reproduces the 0.848 in the main table exactly.
    """
    rows, ys = np.nonzero(target > min_w)
    ws = target[rows, ys]
    return LogisticRegression(C=10, max_iter=3000).fit(x[rows], ys, sample_weight=ws)


def proba(head: LogisticRegression, x: np.ndarray) -> np.ndarray:
    """Probabilities over all 77 classes, in `categories()` order (LR drops unseen classes)."""
    p = np.zeros((len(x), 77), dtype="float32")
    p[:, head.classes_] = head.predict_proba(x)
    return p


def oof_proba(x: np.ndarray, target: np.ndarray, y: np.ndarray, folds: int = 5) -> np.ndarray:
    """Out-of-fold probabilities: every row scored by a head that never saw it."""
    out = np.zeros_like(target)
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=0).split(x, y):
        out[te] = proba(fit_head(x[tr], target[tr]), x[te])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--soft-alpha", type=float, default=0.3)
    ap.add_argument("--filter-p", type=float, default=0.10, help="drop if teacher's label < p")
    ap.add_argument("--filter-q", type=float, default=0.70, help="...and student's pick > q")
    ap.add_argument("--self-p", type=float, default=0.90, help="keep pseudo-labels above p")
    ap.add_argument("--no-diagnostics", action="store_true", help="never read gold on train")
    args = ap.parse_args()

    classes = categories()
    pos = {c: i for i, c in enumerate(classes)}
    train = teacher_train_labels()
    test = load_split("test")
    gold_test = test.category.values
    x_all = minilm_embeddings("train", load_split("train").text.tolist())
    x_test = minilm_embeddings("test", test.text.tolist())
    x_train = x_all[train.index]
    y_idx = np.array([pos[y] for y in train.y])
    unlabelled = load_split("train").drop(index=train.index)
    print(f"{len(train):,} teacher-labelled rows, {len(unlabelled):,} unlabelled, 77 intents")

    gold_train = None if args.no_diagnostics else load_split("train").category[train.index].values

    # --- the filter: which teacher labels does the student confidently reject? ---
    keep = np.ones(len(train), dtype=bool)
    if {"filter", "filter_self", "all"} & set(args.variants.split(",")):
        oof = oof_proba(x_train, soft_targets(train.ranked, train.y, classes, 0.0), y_idx)
        p_teacher = oof[np.arange(len(train)), y_idx]
        p_student = oof.max(1)
        keep = ~((p_teacher < args.filter_p) & (p_student > args.filter_q))
        msg = f"filter: dropping {(~keep).sum()} of {len(train):,} rows"
        if gold_train is not None:
            wrong = train.y.values != gold_train
            msg += (
                f" — {(wrong[~keep]).mean():.0%} of them really are teacher errors "
                f"(base rate {wrong.mean():.0%}); "
                f"{(wrong[~keep]).sum()}/{wrong.sum()} of all teacher errors caught"
            )
        print(msg)

    def run(
        name: str, x: np.ndarray, target: np.ndarray, note: str, n_real: int | None = None
    ) -> None:
        """`n_real`: rows [0, n_real) carry teacher labels, the rest are pseudo-labels. Only the
        real ones may fit the temperature — calibrating on the model's own output would just
        confirm its own confidence."""
        n_real = len(x) if n_real is None else n_real
        n_cal = max(77, int(CALIB_FRAC * n_real))
        rng = np.random.default_rng(0)
        cal = rng.choice(n_real, n_cal, replace=False)
        fit = np.setdiff1d(np.arange(len(x)), cal)
        head = fit_head(x[fit], target[fit])
        p_test = proba(head, x_test)
        pred = np.array(classes)[p_test.argmax(1)]
        acc = accuracy_score(gold_test, pred)
        f1 = f1_score(gold_test, pred, average="macro")
        print(f"  {name:28} {acc:.4f} acc  {f1:.4f} macro-F1   ({note})")
        save_run(
            f"minilm_denoise_{name}",
            {
                "model": f"MiniLM-L6 frozen + {name}",
                "params": 22_713_216,
                "trained_on": "teacher",
                "n_train": int(len(fit)),
                "accuracy": acc,
                "macro_f1": f1,
                "variant": name,
                "note": note,
                "soft_alpha": args.soft_alpha if "soft" in name or name == "all" else None,
            },
            classes,
            p_test,
            calib=(proba(head, x[cal]), target[cal].argmax(1)),
        )

    want = args.variants.split(",")
    if "base" in want:
        run("base", x_train, soft_targets(train.ranked, train.y, classes, 0.0), "teacher top-1")
    if "soft" in want:
        run(
            "soft",
            x_train,
            soft_targets(train.ranked, train.y, classes, args.soft_alpha),
            f"rank-weighted top-3, alpha={args.soft_alpha}",
        )
    if "filter" in want:
        run(
            "filter",
            x_train[keep],
            soft_targets(train.ranked[keep], train.y[keep], classes, 0.0),
            f"{(~keep).sum()} noisy labels dropped",
        )

    def self_train(seed_rows: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """Pseudo-label the unlabelled rows with a head fitted on `seed_rows`, keep the
        confident ones. The threshold is applied to calibrated probabilities: the head is
        under-confident, so a raw 0.9 cut would keep a tenth of what it should."""
        idx = np.flatnonzero(seed_rows)
        rng = np.random.default_rng(1)
        cal = rng.choice(idx, max(77, int(CALIB_FRAC * len(idx))), replace=False)
        fit = np.setdiff1d(idx, cal)
        temp = fit_temperature(
            proba(fit_head(x_train[fit], target[fit]), x_train[cal]), target[cal].argmax(1)
        )
        x_un = x_all[unlabelled.index]
        p_un = apply_temperature(proba(fit_head(x_train[idx], target[idx]), x_un), temp)
        conf = p_un.max(1) > args.self_p
        print(
            f"  self-training: T={temp:.2f}, {conf.sum():,} of {len(unlabelled):,} "
            f"pseudo-labels above {args.self_p}"
        )
        onehot = np.eye(len(classes), dtype="float32")[p_un[conf].argmax(1)]
        return (
            np.vstack([x_train[idx], x_un[conf]]),
            np.vstack([target[idx], onehot]),
            int(conf.sum()),
        )

    if "self" in want:
        hard = soft_targets(train.ranked, train.y, classes, 0.0)
        x_plus, t_plus, n = self_train(np.ones(len(train), bool), hard)
        run("self", x_plus, t_plus, f"+{n:,} pseudo-labelled rows", n_real=len(train))
    if "filter_self" in want:
        hard = soft_targets(train.ranked, train.y, classes, 0.0)
        x_plus, t_plus, n = self_train(keep, hard)
        run(
            "filter_self",
            x_plus,
            t_plus,
            f"filter ({(~keep).sum()} dropped) + {n:,} pseudo-labels",
            n_real=int(keep.sum()),
        )
    if "all" in want:
        soft = soft_targets(train.ranked, train.y, classes, args.soft_alpha)
        x_plus, t_plus, n = self_train(keep, soft)
        run(
            "all",
            x_plus,
            t_plus,
            f"soft + filter ({(~keep).sum()} dropped) + {n:,} pseudo-labels",
            n_real=int(keep.sum()),
        )


if __name__ == "__main__":
    main()
