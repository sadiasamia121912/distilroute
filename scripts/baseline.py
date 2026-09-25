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
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distilroute.data import (  # noqa: E402
    MODELS,
    RAW,
    RESULTS,
    TEACHER_ROWS,
    rel,
    teacher_train_labels,
)
from distilroute.perturb import augment  # noqa: E402
from distilroute.runs import latency_ms, model_dir, save_run, split_calib  # noqa: E402


def load_train(labels: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / "train.csv")
    if labels == "gold":
        return df.rename(columns={"category": "y"})
    df = teacher_train_labels()  # the first TEACHER_ROWS labelled rows, unparsed dropped
    print(f"teacher labels: {len(df):,} usable of the first {TEACHER_ROWS:,} labelled")
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", choices=["gold", "teacher"], default="gold")
    ap.add_argument(
        "--augment",
        default="",
        help="add noisy copies of the training rows, e.g. typo1,typo3 (roadmap 6.6)",
    )
    args = ap.parse_args()
    kinds = [k for k in args.augment.split(",") if k]

    train, calib = split_calib(load_train(args.labels))
    if kinds:
        train = augment(train, kinds)
    test = pd.read_csv(RAW / "test.csv")

    model = build()
    t0 = time.time()
    model.fit(train.text, train.y)
    fit_s = time.time() - t0

    proba = model.predict_proba(test.text)
    classes = list(model.classes_)
    p_calib = model.predict_proba(calib.text)
    y_calib = np.array([classes.index(y) for y in calib.y])
    pred = np.array(classes)[proba.argmax(axis=1)]
    acc = accuracy_score(test.category, pred)
    f1 = f1_score(test.category, pred, average="macro")
    p50, p95 = latency_ms(lambda t: model.predict([t]), test.text.tolist())

    print(
        f"tfidf+lr trained on {args.labels} ({len(train):,} rows, {len(calib)} held out for "
        f"calibration, {fit_s:.0f}s)"
    )
    print(f"  test accuracy vs gold  {acc:.4f}")
    print(f"  test macro-F1 vs gold  {f1:.4f}")
    print(f"  latency per query      p50 {p50:.2f} ms  p95 {p95:.2f} ms")

    name = "tfidf_lr" + ("_aug" if kinds else "") + f"_{args.labels}"
    metrics = save_run(
        name,
        {
            "model": "tfidf+lr",
            "params": 0,
            "trained_on": args.labels,
            "n_train": int(len(train)),
            "accuracy": acc,
            "macro_f1": f1,
            "p50_ms": p50,
            "p95_ms": p95,
            "fit_seconds": fit_s,
            "augment": kinds or None,
        },
        classes,
        proba,
        calib=(p_calib, y_calib),
    )
    d = model_dir(name, "tfidf", temperature=metrics["temperature"])
    joblib.dump(model, d / "model.joblib")
    print(f"  -> {rel(RESULTS)}/{name}.json + _test_probs.npz, {rel(MODELS)}/{name}/")


if __name__ == "__main__":
    sys.exit(main())
