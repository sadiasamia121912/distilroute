"""Sentence-embedding students (MiniLM, 22M params), CPU-only.

    python scripts/setfit_student.py --mode frozen  --labels gold      # embeddings + LR head
    python scripts/setfit_student.py --mode setfit  --labels teacher --per-class 16

`frozen`: encode every training query with a pretrained sentence-transformer, fit a logistic
head. No fine-tuning, so it is fast (a couple of minutes on this laptop) and shows what a
pretrained encoder gives for free.

`setfit`: SetFit contrastive fine-tuning of the encoder on `--per-class` examples per intent,
then the same head. This is the few-shot pattern — how far do 16 labels per intent go? On CPU
the contrastive phase is the cost, so it is capped with `--max-steps`.

Both write results under the shared contract (metrics JSON + test probabilities), and both
report per-query latency for encode + head, which is what a service would pay.

Requires `requirements-train.txt` (torch CPU, sentence-transformers, setfit).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distilroute.data import load_split, teacher_train_labels  # noqa: E402
from distilroute.runs import latency_ms, model_dir, save_run  # noqa: E402

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
PARAMS = 22_713_216


def load_train(labels: str, per_class: int | None, seed: int):
    if labels == "gold":
        df = load_split("train").rename(columns={"category": "y"})
    else:
        df = teacher_train_labels()
        print(f"teacher labels: {len(df):,} usable")
    if per_class:
        # Same N for every intent (the smallest Banking77 intent has 35 train rows).
        n = min(per_class, int(df.y.value_counts().min()))
        df = df.groupby("y").sample(n=n, random_state=seed)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["frozen", "setfit"], default="frozen")
    ap.add_argument("--labels", choices=["gold", "teacher"], default="gold")
    ap.add_argument("--per-class", type=int, default=None, help="subsample N per intent")
    ap.add_argument("--max-steps", type=int, default=400, help="setfit: contrastive steps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.mode == "setfit" and args.per_class is None:
        args.per_class = 16

    from sentence_transformers import SentenceTransformer

    train = load_train(args.labels, args.per_class, args.seed)
    test = load_split("test")
    print(f"{args.mode}: {len(train):,} training rows, labels={args.labels}")

    t0 = time.time()
    if args.mode == "frozen":
        encoder = SentenceTransformer(ENCODER, device="cpu")
    else:
        from datasets import Dataset
        from setfit import SetFitModel, Trainer, TrainingArguments

        model = SetFitModel.from_pretrained(ENCODER, device="cpu")
        ds = Dataset.from_dict({"text": train.text.tolist(), "label": train.y.tolist()})
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                batch_size=16, max_steps=args.max_steps, seed=args.seed, report_to="none"
            ),
            train_dataset=ds,
        )
        trainer.train()  # contrastive phase only; we fit our own head below for the probas
        encoder = model.model_body

    x_train = encoder.encode(train.text.tolist(), batch_size=64, show_progress_bar=False)
    head = LogisticRegression(C=10, max_iter=3000).fit(x_train, train.y)
    fit_s = time.time() - t0

    x_test = encoder.encode(test.text.tolist(), batch_size=64, show_progress_bar=False)
    proba = head.predict_proba(x_test)
    classes = list(head.classes_)
    pred = np.array(classes)[proba.argmax(axis=1)]
    acc = accuracy_score(test.category, pred)
    f1 = f1_score(test.category, pred, average="macro")

    def predict_one(t: str):
        return head.predict(encoder.encode([t], show_progress_bar=False))

    p50, p95 = latency_ms(predict_one, test.text.tolist(), n=200)

    tag = f"minilm_{args.mode}" + (f"_{args.per_class}pc" if args.per_class else "")
    name = f"{tag}_{args.labels}"
    print(f"  fit {fit_s:.0f}s   acc {acc:.4f}   macro-F1 {f1:.4f}")
    print(f"  latency p50 {p50:.1f} ms  p95 {p95:.1f} ms")
    save_run(
        name,
        {
            "model": f"MiniLM-L6 {args.mode}"
            + (f" ({args.per_class}/intent)" if args.per_class else ""),
            "params": PARAMS,
            "trained_on": args.labels,
            "n_train": int(len(train)),
            "accuracy": acc,
            "macro_f1": f1,
            "p50_ms": p50,
            "p95_ms": p95,
            "fit_seconds": fit_s,
            "encoder": ENCODER,
            "max_steps": args.max_steps if args.mode == "setfit" else None,
        },
        classes,
        proba,
    )
    # Frozen mode only needs the head; setfit mode also saves the fine-tuned encoder.
    if args.mode == "frozen":
        d = model_dir(name, "minilm", encoder=ENCODER)
    else:
        d = model_dir(name, "minilm", encoder="encoder")
        encoder.save(str(d / "encoder"))
    joblib.dump(head, d / "head.joblib")
    print(f"  -> results/{name}.json, models/{name}/")


if __name__ == "__main__":
    main()
