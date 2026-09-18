"""Data-efficiency curve: student accuracy vs. number of training labels (roadmap 1b.3).

    python scripts/data_curve.py --labels gold                 # both students, 3 seeds
    python scripts/data_curve.py --labels teacher --student minilm

"How many LLM calls do you actually need?" For each size N, sample N random training rows
(what labelling N random tickets would give you), train the student, score against gold on
the full test split. Repeated over seeds so the spread is visible. Sizes larger than the
pool of available labels are skipped, so the same command works while the teacher labels
are still filling in.

Students are the two CPU-trainable ones — TF-IDF + LR and frozen MiniLM-L6 + LR — with the
same hyper-parameters as their full runs. MiniLM embeddings are computed once per split and
cached under models/ (gitignored) because the encoder never changes, only the head does.

Writes results/curves/<student>_<labels>.json; `scripts/evaluate.py` renders the table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distilroute.data import RESULTS, ROOT, load_split, teacher_train_labels  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline import build as build_tfidf  # noqa: E402
from setfit_student import ENCODER  # noqa: E402

CURVES = RESULTS / "curves"
CACHE = ROOT / "models" / "cache"
SIZES = [250, 500, 1000, 2000, 5000]  # plus the full pool, whatever its size


def load_pool(labels: str) -> pd.DataFrame:
    if labels == "gold":
        return load_split("train").rename(columns={"category": "y"})
    df = teacher_train_labels()
    print(f"teacher labels: {len(df):,} usable")
    return df


def minilm_embeddings(split: str, texts: list[str]) -> np.ndarray:
    """Frozen-encoder embeddings for a whole split, cached: the encoder is fixed, so they are."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"minilm_{split}.npy"
    if path.exists():
        emb = np.load(path)
        if len(emb) == len(texts):
            return emb
    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    emb = SentenceTransformer(ENCODER, device="cpu").encode(
        texts, batch_size=64, show_progress_bar=False
    )
    np.save(path, emb)
    print(f"  encoded {split} ({len(texts):,}) in {time.time() - t0:.0f}s -> {path.name}")
    return emb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", choices=["gold", "teacher"], default="gold")
    ap.add_argument("--student", choices=["tfidf", "minilm", "both"], default="both")
    ap.add_argument("--sizes", default=",".join(map(str, SIZES)))
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    students = ["tfidf", "minilm"] if args.student == "both" else [args.student]

    pool = load_pool(args.labels)
    test = load_split("test")
    gold = test.category.values
    sizes = sorted({n for n in sizes if n < len(pool)} | {len(pool)})  # always end on the full pool
    print(f"pool {len(pool):,} rows ({args.labels}); sizes {sizes}; {args.seeds} seeds")

    for student in students:
        if student == "minilm":
            x_pool = minilm_embeddings("train", load_split("train").text.tolist())[pool.index]
            x_test = minilm_embeddings("test", test.text.tolist())
        rows = []
        for n in sizes:
            for seed in range(args.seeds):
                sub = pool.sample(n=n, random_state=seed) if n < len(pool) else pool
                pos = pool.index.get_indexer(sub.index)
                t0 = time.time()
                if student == "tfidf":
                    model = build_tfidf().fit(sub.text, sub.y)
                    pred = model.predict(test.text)
                else:
                    head = LogisticRegression(C=10, max_iter=3000).fit(x_pool[pos], sub.y)
                    pred = head.predict(x_test)
                row = {
                    "n": int(n),
                    "seed": seed,
                    "n_classes_seen": int(sub.y.nunique()),
                    "accuracy": float(accuracy_score(gold, pred)),
                    "macro_f1": float(f1_score(gold, pred, average="macro")),
                    "fit_seconds": time.time() - t0,
                }
                rows.append(row)
                print(
                    f"  {student:6} n={n:>6,} seed={seed}  acc {row['accuracy']:.4f}  "
                    f"F1 {row['macro_f1']:.4f}  ({row['n_classes_seen']} intents seen, "
                    f"{row['fit_seconds']:.0f}s)",
                    flush=True,
                )
                if n == len(pool):
                    break  # the full pool has no sampling variance
        CURVES.mkdir(parents=True, exist_ok=True)
        out = CURVES / f"{student}_{args.labels}.json"
        out.write_text(
            json.dumps(
                {
                    "student": {"tfidf": "tfidf+lr", "minilm": "MiniLM-L6 frozen"}[student],
                    "trained_on": args.labels,
                    "pool": int(len(pool)),
                    "rows": rows,
                },
                indent=2,
            )
        )
        print(f"  -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
