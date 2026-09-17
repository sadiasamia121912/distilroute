"""TF-IDF + logistic regression: the student everyone should beat, and the first row of the table.

    python scripts/baseline.py --labels gold      # supervised reference (human labels)
    python scripts/baseline.py --labels teacher   # distilled (needs data/labels/train.jsonl)

Either way the score is against **gold** on the untouched test split. Word 1-2-grams plus
character 2-5-grams: the char n-grams are what make this competitive on short, typo-ridden
queries. Also reports p50/p95 latency for a single query, since that column is the point of
the project.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
LABELS = ROOT / "data" / "labels"
RESULTS = ROOT / "results"


def load_train(labels: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / "train.csv")
    if labels == "gold":
        return df.rename(columns={"category": "y"})
    recs = [json.loads(line) for line in (LABELS / "train.jsonl").open(encoding="utf-8")]
    lab = pd.DataFrame(recs).set_index("idx")["teacher"]
    df = df.join(lab, how="inner")  # only the queries the teacher has labelled so far
    n_unparsed = df.teacher.isna().sum()
    df = df.dropna(subset=["teacher"]).rename(columns={"teacher": "y"})
    print(f"teacher labels: {len(df):,} usable, {n_unparsed} unparsed dropped")
    return df


def build() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                FeatureUnion(
                    [
                        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2
                            ),
                        ),
                    ]
                ),
            ),
            ("clf", LogisticRegression(C=10, max_iter=2000)),
        ]
    )


def latency(model: Pipeline, texts: list[str], n: int = 500) -> tuple[float, float]:
    times = []
    for t in texts[:n]:
        t0 = time.perf_counter()
        model.predict([t])
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(times, 50)), float(np.percentile(times, 95))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", choices=["gold", "teacher"], default="gold")
    args = ap.parse_args()

    train = load_train(args.labels)
    test = pd.read_csv(RAW / "test.csv")

    model = build()
    t0 = time.time()
    model.fit(train.text, train.y)
    fit_s = time.time() - t0

    pred = model.predict(test.text)
    acc = accuracy_score(test.category, pred)
    f1 = f1_score(test.category, pred, average="macro")
    p50, p95 = latency(model, test.text.tolist())

    print(f"tfidf+lr trained on {args.labels} ({len(train):,} rows, {fit_s:.0f}s)")
    print(f"  test accuracy vs gold  {acc:.4f}")
    print(f"  test macro-F1 vs gold  {f1:.4f}")
    print(f"  latency per query      p50 {p50:.2f} ms  p95 {p95:.2f} ms")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"tfidf_lr_{args.labels}.json"
    out.write_text(
        json.dumps(
            {
                "model": "tfidf+lr",
                "trained_on": args.labels,
                "n_train": int(len(train)),
                "accuracy": acc,
                "macro_f1": f1,
                "p50_ms": p50,
                "p95_ms": p95,
                "fit_seconds": fit_s,
            },
            indent=2,
        )
    )
    print(f"  -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
