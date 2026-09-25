"""Fine-tune a small transformer as the student (roadmap 2.3 / 1b.4), on Colab's free GPU.

    python scripts/finetune.py --model distilbert --labels gold
    python scripts/finetune.py --model distilbert --labels teacher --soft      # top-3 targets
    python scripts/finetune.py --model tinybert   --labels teacher --export-onnx

Full fine-tuning with a classification head, plain PyTorch loop (AdamW, linear warm-up and
decay, fp16 autocast on CUDA). `--soft` trains on the teacher's ranked top-3 instead of the
hard top-1: the target is (1 - alpha) * one-hot(top-1) + alpha * rank-weighted(1, 1/2, 1/3),
i.e. Hinton-style distillation with the ranks standing in for the logits we cannot get from
the API. 5 % of the training pool is held out to print validation accuracy per epoch, scored
against the *training* labels (teacher or gold) — gold on the test split is never looked at
until the end.

`--export-onnx` writes models/<run>/model.onnx and model.int8.onnx (dynamic quantisation),
which is what the laptop serves and benchmarks with onnxruntime alone (no torch). Accuracy of
the int8 graph on the test split is recorded next to the fp32 number.

On this laptop (CPU) only a smoke test is sensible: `--limit 256 --epochs 1`.
Requires requirements-train.txt (+ onnx/onnxruntime for the export).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distilroute.data import (  # noqa: E402
    RESULTS,
    ROOT,
    categories,
    load_split,
    rel,
    teacher_train_labels,
)
from distilroute.perturb import augment  # noqa: E402
from distilroute.runs import latency_ms, model_dir, save_run  # noqa: E402

# name -> (hub id, params, learning rate). Small encoders need a higher LR: at 5e-5 MiniLM and
# TinyBERT were still at 0.5 / 0.4 after 4 epochs on 9k rows (Colab, 2026-09-21).
MODELS = {
    "distilbert": ("distilbert-base-uncased", 66_955_085, 5e-5),
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", 22_713_216, 1e-4),
    "tinybert": ("huawei-noah/TinyBERT_General_4L_312D", 14_350_874, 3e-4),
}
MIN_STEPS = 2000  # a 77-way head needs this many updates whatever the pool size
RANK_WEIGHTS = np.array([1.0, 1 / 2, 1 / 3])
MAX_LEN = 64  # p95 query is 151 chars ~ 40 tokens


def load_train(labels: str, limit: int | None, seed: int) -> pd.DataFrame:
    if labels == "gold":
        df = load_split("train").rename(columns={"category": "y"})
        df["ranked"] = [[y] for y in df.y]
    else:
        df = teacher_train_labels()
        print(f"teacher labels: {len(df):,} usable")
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=seed)
    return df


def targets(df: pd.DataFrame, classes: list[str], soft: bool, alpha: float) -> np.ndarray:
    """(n, 77) target distribution: one-hot, or blended with the rank-weighted top-k."""
    pos = {c: i for i, c in enumerate(classes)}
    t = np.zeros((len(df), len(classes)), dtype="float32")
    for row, (y, ranked) in enumerate(zip(df.y, df.ranked, strict=True)):
        t[row, pos[y]] = 1.0
        if soft and len(ranked) > 1:
            known = [r for r in ranked[: len(RANK_WEIGHTS)] if r in pos]
            w = RANK_WEIGHTS[: len(known)] / RANK_WEIGHTS[: len(known)].sum()
            t[row] *= 1 - alpha
            for r, wi in zip(known, w, strict=True):
                t[row, pos[r]] += alpha * wi
    return t


def encode(tokenizer, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    enc = tokenizer(
        texts, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt"
    )
    return enc["input_ids"], enc["attention_mask"]


@torch.no_grad()
def predict_proba(model, ids, mask, device, batch_size: int = 128) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(ids), batch_size):
        logits = model(
            input_ids=ids[i : i + batch_size].to(device),
            attention_mask=mask[i : i + batch_size].to(device),
        ).logits
        out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def export_onnx(model, tokenizer, out_dir: Path) -> tuple[Path, Path]:
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out_dir.mkdir(parents=True, exist_ok=True)
    fp32, int8 = out_dir / "model.onnx", out_dir / "model.int8.onnx"
    model = model.cpu().eval()
    ids, mask = encode(tokenizer, ["export sample"])
    torch.onnx.export(
        model,
        (ids, mask),
        str(fp32),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(str(fp32))
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    tokenizer.save_pretrained(out_dir)
    return fp32, int8


def onnx_inputs(enc) -> dict[str, np.ndarray]:
    return {k: np.asarray(enc[k]).astype("int64") for k in ("input_ids", "attention_mask")}


def onnx_predict_proba(path: Path, tokenizer, texts: list[str], batch_size: int = 128):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[i : i + batch_size],
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="np",
        )
        logits = sess.run(["logits"], onnx_inputs(enc))[0]
        z = logits - logits.max(axis=1, keepdims=True)
        out.append(np.exp(z) / np.exp(z).sum(axis=1, keepdims=True))
    return np.concatenate(out), sess


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="distilbert")
    ap.add_argument("--labels", choices=["gold", "teacher"], default="gold")
    ap.add_argument("--soft", action="store_true", help="train on the teacher's ranked top-3")
    ap.add_argument("--soft-alpha", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=None, help="default: enough for MIN_STEPS")
    ap.add_argument("--lr", type=float, default=None, help="default: per model, see MODELS")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="subsample the training pool")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--export-onnx", action="store_true")
    ap.add_argument(
        "--augment",
        default="",
        help="add noisy copies of the training rows, e.g. typo1,typo3 (roadmap 6.6)",
    )
    args = ap.parse_args()
    kinds = [k for k in args.augment.split(",") if k]
    if args.soft and args.labels == "gold":
        sys.exit("--soft needs teacher labels (gold has no ranking)")

    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_name, n_params, default_lr = MODELS[args.model]
    args.lr = args.lr or default_lr
    classes = categories()
    train = load_train(args.labels, args.limit, args.seed)
    # Held out from training: per-epoch validation and, at the end, the calibration set.
    val = train.groupby("y", group_keys=False).sample(frac=0.1, random_state=args.seed)
    train = train.drop(val.index)
    if kinds:
        train = augment(train, kinds)
    test = load_split("test")
    print(
        f"{args.model} ({hf_name}) on {device}: {len(train):,} train / {len(val):,} val rows, "
        f"labels={args.labels}{' soft' if args.soft else ''}"
    )

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForSequenceClassification.from_pretrained(hf_name, num_labels=len(classes)).to(
        device
    )
    ids, mask = encode(tokenizer, train.text.tolist())
    y = torch.tensor(targets(train, classes, args.soft, args.soft_alpha))
    loader = DataLoader(TensorDataset(ids, mask, y), batch_size=args.batch_size, shuffle=True)
    if args.epochs is None:
        args.epochs = max(4, -(-MIN_STEPS // len(loader)))  # ceil
    print(f"  lr {args.lr:g}, {args.epochs} epochs x {len(loader)} steps")
    val_ids, val_mask = encode(tokenizer, val.text.tolist())
    val_y = val.y.values

    steps = args.epochs * len(loader)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for b_ids, b_mask, b_y in loader:
            b_ids, b_mask, b_y = b_ids.to(device), b_mask.to(device), b_y.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids=b_ids, attention_mask=b_mask).logits
            loss = -(b_y * torch.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(b_ids)
        val_pred = np.array(classes)[predict_proba(model, val_ids, val_mask, device).argmax(1)]
        print(
            f"  epoch {epoch + 1}/{args.epochs}  loss {total / len(train):.4f}  "
            f"val acc (vs {args.labels}) {accuracy_score(val_y, val_pred):.4f}  "
            f"{time.time() - t0:.0f}s",
            flush=True,
        )
    fit_s = time.time() - t0

    test_ids, test_mask = encode(tokenizer, test.text.tolist())
    proba = predict_proba(model, test_ids, test_mask, device)
    pred = np.array(classes)[proba.argmax(1)]
    acc = accuracy_score(test.category, pred)
    f1 = f1_score(test.category, pred, average="macro")
    print(f"  test acc {acc:.4f}   macro-F1 {f1:.4f}   ({fit_s:.0f}s training)")

    tag = (
        f"{args.model}_ft"
        + ("_soft" if args.soft else "")
        + (f"_{args.limit}" if args.limit else "")
        + ("_aug" if kinds else "")
    )
    name = f"{tag}_{args.labels}"
    metrics = {
        "model": f"{args.model} fine-tuned" + (" (soft top-3)" if args.soft else ""),
        "hf_model": hf_name,
        "params": n_params,
        "trained_on": args.labels,
        "n_train": int(len(train)),
        "accuracy": acc,
        "macro_f1": f1,
        "fit_seconds": fit_s,
        "device": device,
        "epochs": args.epochs,
        "lr": args.lr,
        "soft_alpha": args.soft_alpha if args.soft else None,
        "augment": kinds or None,
        "latency_host": platform.node(),
    }

    # Latency of the torch model on this host's CPU, single query, as a service would see it.
    model_cpu = model.to("cpu").eval()

    def predict_one(t: str):
        i, m = encode(tokenizer, [t])
        return model_cpu(input_ids=i, attention_mask=m).logits.argmax(-1)

    with torch.no_grad():
        metrics["p50_ms"], metrics["p95_ms"] = latency_ms(predict_one, test.text.tolist(), n=200)
    print(f"  torch CPU latency p50 {metrics['p50_ms']:.1f} ms  p95 {metrics['p95_ms']:.1f} ms")

    if args.export_onnx:
        out_dir = model_dir(name, "onnx", hf_model=hf_name)  # temperature added below
        fp32, int8 = export_onnx(model_cpu, tokenizer, out_dir)
        q_proba, sess = onnx_predict_proba(int8, tokenizer, test.text.tolist())
        q_pred = np.array(classes)[q_proba.argmax(1)]
        metrics["accuracy_int8"] = accuracy_score(test.category, q_pred)

        def predict_one_int8(t: str):
            enc = tokenizer([t], truncation=True, max_length=MAX_LEN, return_tensors="np")
            return sess.run(["logits"], onnx_inputs(enc))[0].argmax(-1)

        metrics["p50_ms_int8"], metrics["p95_ms_int8"] = latency_ms(
            predict_one_int8, test.text.tolist(), n=200
        )
        metrics["onnx_mb"] = round(fp32.stat().st_size / 1e6, 1)
        metrics["onnx_int8_mb"] = round(int8.stat().st_size / 1e6, 1)
        print(
            f"  onnx int8: acc {metrics['accuracy_int8']:.4f}  "
            f"p50 {metrics['p50_ms_int8']:.1f} ms  {metrics['onnx_int8_mb']} MB "
            f"(fp32 {metrics['onnx_mb']} MB) -> {out_dir.relative_to(ROOT)}"
        )
        (out_dir / "classes.json").write_text(json.dumps(classes))

    y_calib = np.array([classes.index(y) for y in val_y])
    metrics = save_run(
        name,
        metrics,
        classes,
        proba,
        calib=(predict_proba(model_cpu, val_ids, val_mask, "cpu"), y_calib),
    )
    if args.export_onnx:
        model_dir(name, "onnx", hf_model=hf_name, temperature=metrics["temperature"])
    print(f"  -> {rel(RESULTS)}/{name}.json")


if __name__ == "__main__":
    main()
