"""Which tickets are worth spending an LLM call on? (roadmap 6.3)

    python scripts/active.py                       # every strategy, 3 seeds
    python scripts/active.py --strategies random,uncertainty --seeds 1

The data curve (`data_curve.py`) labels *random* rows. A team paying per LLM call can choose
which rows to send instead, and the question is worth money: if picking well reaches the same
accuracy with half the labels, the labelling bill halves.

Simulated honestly inside the 3,000 rows the teacher has already labelled — a strategy may
only "buy" a label that exists, and never sees gold. Strategies:

- **random**        — the baseline `data_curve.py` measures.
- **uncertainty**   — label a random 500 first, train, then buy the rows the student is least
  sure about (entropy over all 77 intents, the score 6.2 found best).
- **disagreement**  — same seed round, then buy where TF-IDF and MiniLM predict different
  intents: cheap query-by-committee, no confidence needed.
- **diverse**       — no seed round: k-means over the embeddings, buy the row closest to each
  centroid. The cold-start strategy — it needs no model at all.

Writes results/curves/active_<strategy>.json, which `scripts/evaluate.py` renders in the same
data-efficiency table as the random curves.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distilroute.data import ROOT, load_split, teacher_train_labels  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline import build as build_tfidf  # noqa: E402
from data_curve import CURVES, minilm_embeddings  # noqa: E402

SEED_ROUND = 500  # labels spent before any strategy can be clever
SIZES = [500, 1000, 1500, 2000, 2500]
STRATEGIES = ["random", "uncertainty", "disagreement", "diverse"]


def head(x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    return LogisticRegression(C=10, max_iter=3000).fit(x, y)


def rank_uncertainty(x_pool: np.ndarray, seed_idx: np.ndarray, rest: np.ndarray, y) -> np.ndarray:
    """`rest`, most-uncertain first, according to a student trained on the seed round."""
    p = head(x_pool[seed_idx], y[seed_idx]).predict_proba(x_pool[rest])
    entropy = -(p * np.log(p + 1e-12)).sum(1)
    return rest[np.argsort(-entropy)]


def rank_disagreement(
    x_pool: np.ndarray, texts: np.ndarray, seed_idx: np.ndarray, rest: np.ndarray, y
) -> np.ndarray:
    """`rest` with the rows where two different students disagree first (ties broken randomly)."""
    minilm = head(x_pool[seed_idx], y[seed_idx]).predict(x_pool[rest])
    tfidf = build_tfidf().fit(texts[seed_idx], y[seed_idx]).predict(texts[rest])
    disagree = minilm != tfidf
    return np.r_[rest[disagree], rest[~disagree]]


def rank_diverse(x_pool: np.ndarray, n: int, seed: int) -> np.ndarray:
    """One row per k-means centroid: coverage of the input space, no model, no seed round."""
    km = KMeans(n_clusters=n, n_init=3, random_state=seed).fit(x_pool)
    # km.transform gives the (rows x centroids) distance matrix directly; building it by
    # broadcasting would be a (3000, 2500, 384) intermediate, i.e. 8 GB.
    return np.unique(km.transform(x_pool).argmin(0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--sizes", default=",".join(map(str, SIZES)))
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]

    pool = teacher_train_labels()
    test = load_split("test")
    gold = test.category.values
    x_pool = minilm_embeddings("train", load_split("train").text.tolist())[pool.index]
    x_test = minilm_embeddings("test", test.text.tolist())
    y = pool.y.values
    texts = pool.text.values
    # Diagnostic only, printed and stored, never used to select or train: a strategy that
    # prefers ambiguous rows also prefers rows the teacher got wrong, and that shows up here.
    gold_train = load_split("train").category[pool.index].values
    noisy = y != gold_train
    print(
        f"pool {len(pool):,} teacher-labelled rows ({noisy.mean():.1%} of them mislabelled by "
        f"the teacher); sizes {sizes}; {args.seeds} seeds"
    )

    for strategy in args.strategies.split(","):
        rows = []
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(pool))
            seed_idx = order[:SEED_ROUND]
            rest = order[SEED_ROUND:]
            if strategy == "random":
                ranked = rest
            elif strategy == "uncertainty":
                ranked = rank_uncertainty(x_pool, seed_idx, rest, y)
            elif strategy == "disagreement":
                ranked = rank_disagreement(x_pool, texts, seed_idx, rest, y)
            else:
                ranked = None  # diverse picks per size, below
            for n in sizes:
                t0 = time.time()
                if strategy == "diverse":
                    pick = rank_diverse(x_pool, n, seed)
                    if len(pick) < n:  # duplicate centroids: top up at random
                        extra = np.setdiff1d(order, pick)[: n - len(pick)]
                        pick = np.r_[pick, extra]
                else:
                    pick = np.r_[seed_idx, ranked[: max(0, n - SEED_ROUND)]][:n]
                pred = head(x_pool[pick], y[pick]).predict(x_test)
                row = {
                    "n": int(n),
                    "seed": seed,
                    "n_classes_seen": int(len(set(y[pick]))),
                    "accuracy": float(accuracy_score(gold, pred)),
                    "macro_f1": float(f1_score(gold, pred, average="macro")),
                    "teacher_error_rate": float(noisy[pick].mean()),
                    "picked": [int(i) for i in pick],
                    "fit_seconds": time.time() - t0,
                }
                rows.append(row)
                print(
                    f"  {strategy:13} n={n:>5} seed={seed}  acc {row['accuracy']:.4f}  "
                    f"F1 {row['macro_f1']:.4f}  ({row['n_classes_seen']} intents, "
                    f"{row['teacher_error_rate']:.1%} of the bought labels wrong)",
                    flush=True,
                )
        CURVES.mkdir(parents=True, exist_ok=True)
        out = CURVES / f"active_{strategy}.json"
        out.write_text(
            json.dumps(
                {
                    "student": f"MiniLM-L6 frozen — {strategy} selection",
                    "trained_on": "teacher",
                    "pool": int(len(pool)),
                    "seed_round": SEED_ROUND,
                    "rows": rows,
                },
                indent=2,
            )
        )
        print(f"  -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
