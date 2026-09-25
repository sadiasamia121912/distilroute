"""Build (and optionally publish) the Hugging Face release: model, dataset, Space (roadmap 4.3).

    python scripts/build_hf_release.py --user NAME              # build/hf/{model,dataset,space}
    python scripts/build_hf_release.py --user NAME --publish    # ...and upload (hf auth login first)

- **model**   — the served student (MiniLM-L6 fine-tuned on teacher labels, int8 ONNX), its
  tokenizer, calibration temperature, and `route.py`: inference with only onnxruntime +
  tokenizers, identical to the service (`distilroute.students.OnnxRouter`).
- **dataset** — every teacher label so far (train and test), next to the human label, so the
  LLM's noise can be studied without re-buying it. CC-BY-4.0, like Banking77.
- **space**   — a Gradio app on the model: intent, calibrated confidence, top 3, and whether the
  cascade would send the message to the LLM.

Every number on the cards is read from results/ or data/labels/ at build time, like the case
study, so a re-run after an experiment updates the release instead of drifting from it.
"""

# ruff: noqa: E501 — the cards are Markdown tables; wrapping their rows would break them
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distilroute.data import LABELS, MODELS, RESULTS, ROOT, load_labels, load_split  # noqa: E402

SERVED = "minilm_ft_teacher"
EXAMPLE = "I still have not received my new card"
THRESHOLD = 0.8  # the cascade threshold of 1b.2 / 6.5
MAX_LENGTH = 64
GITHUB = "https://github.com/sadiasamia121912/distilroute"
OUT = ROOT / "build" / "hf"

ROUTE_PY = '''"""Route a support message with the distilroute student: onnxruntime + tokenizers only.

    pip install onnxruntime tokenizers numpy
    python route.py "I still have not received my new card"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class Router:
    def __init__(self, path: str | Path = "."):
        path = Path(path)
        cfg = json.loads((path / "config.json").read_text())
        self.temperature = cfg["temperature"]
        self.threshold = cfg["escalate_below"]
        self.classes = json.loads((path / "classes.json").read_text())
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.tokenizer.enable_truncation(cfg["max_length"])
        self.tokenizer.no_padding()
        self.session = ort.InferenceSession(
            str(path / "model.int8.onnx"), providers=["CPUExecutionProvider"]
        )

    def probabilities(self, text: str) -> np.ndarray:
        """Calibrated probabilities over the 77 intents, in `classes` order."""
        enc = self.tokenizer.encode(text)
        feed = {
            "input_ids": np.array([enc.ids], dtype="int64"),
            "attention_mask": np.array([enc.attention_mask], dtype="int64"),
        }
        logits = self.session.run(["logits"], feed)[0][0] / self.temperature
        z = np.exp(logits - logits.max())
        return z / z.sum()

    def route(self, text: str) -> dict:
        p = self.probabilities(text)
        top = np.argsort(-p)[:3]
        entropy = float(-(p * np.log(np.clip(p, 1e-12, None))).sum())
        return {
            "intent": self.classes[top[0]],
            "confidence": float(p[top[0]]),
            "top3": [(self.classes[i], float(p[i])) for i in top],
            "escalate": bool(p[top[0]] < self.threshold),
            "entropy": entropy,
        }


if __name__ == "__main__":
    print(json.dumps(Router(Path(__file__).parent).route(" ".join(sys.argv[1:])), indent=2))
'''

APP_PY = '''"""distilroute: a 22M-parameter student of a 120B LLM, routing banking support messages."""

import gradio as gr
from huggingface_hub import snapshot_download

from route import Router

router = Router(snapshot_download("{model_repo}"))

EXAMPLES = [
    "I still have not received my new card, I ordered over a week ago.",
    "Why was I charged a fee for topping up by card?",
    "my card got declined at the shop but I have money in my acct",
    "Someone used my card in another country, I did not make that payment!",
    "Can I get a physical card if I live in Canada?",
    "what's the weather like tomorrow",
]


def route(text: str):
    if not text.strip():
        return {{}}, ""
    r = router.route(text)
    verdict = (
        f"**Send to the LLM**: confidence {{r['confidence']:.2f}} is below {{router.threshold}}."
        if r["escalate"]
        else f"**Answered by the student**: confidence {{r['confidence']:.2f}}."
    )
    return {{name: p for name, p in r["top3"]}}, verdict


demo = gr.Interface(
    fn=route,
    inputs=gr.Textbox(label="Support message", lines=2),
    outputs=[gr.Label(label="Intent (calibrated)", num_top_classes=3), gr.Markdown()],
    examples=EXAMPLES,
    title="distilroute: a 120B LLM distilled into 22M parameters",
    description=(
        "Routes a banking support message to one of 77 intents in about 3 ms on a CPU. The "
        "student was trained only on labels from gpt-oss-120b, keeps {keep} of its accuracy "
        "({acc} vs {teacher_acc} on Banking77 test), and sends messages it is unsure about "
        "(confidence < {threshold}) to the LLM. [Model]({model_url}) · [Code]({github})"
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
'''


def j(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def numbers() -> dict:
    boot = {r["system"]: r for r in j(RESULTS / "bootstrap.json")["rows"]}
    lat = j(RESULTS / "latency.json")[SERVED]
    cost = j(RESULTS / "cost.json")["rows"]
    rob = next(r for r in j(RESULTS / "robustness.json")["students"] if r["model"] == SERVED)
    k = rob["kinds"]
    served, teacher, cascade = boot[f"{SERVED}_int8"], boot["teacher"], boot["cascade"]
    return {
        "acc": served["acc"],
        "ci": served["ci"],
        "teacher_acc": teacher["acc"],
        "gap": served["vs_teacher"],
        "gap_ci": served["vs_teacher_ci"],
        "cascade": cascade["acc"],
        "cascade_ci": cascade["vs_teacher_ci"],
        "gold_ref": boot["minilm_ft_gold"]["acc"],
        "p50": lat["p50_ms"],
        "p95": lat["p95_ms"],
        "usd": cost[SERVED]["usd_per_1m"],
        "teacher_usd": cost["teacher_single"]["usd_per_1m"],
        "typo1": k["typo1"]["acc"] - k["clean"]["acc"],
        "typo3": k["typo3"]["acc"] - k["clean"]["acc"],
        "esc_clean": k["clean"]["escalated"],
        "esc_typo3": k["typo3"]["escalated"],
        "n_train": j(RESULTS / f"{SERVED}.json")["n_train"],
    }


def model_card(n: dict, repos: dict) -> str:
    pt = lambda x: f"{x * 100:+.1f}"  # noqa: E731
    return f"""---
license: apache-2.0
language: en
library_name: onnx
pipeline_tag: text-classification
base_model: sentence-transformers/all-MiniLM-L6-v2
datasets:
  - PolyAI/banking77
  - {repos["dataset"]}
tags:
  - intent-classification
  - knowledge-distillation
  - llm-distillation
  - onnx
  - int8
model-index:
  - name: distilroute-minilm-banking77
    results:
      - task:
          type: text-classification
        dataset:
          name: Banking77 (test, 3,080 queries)
          type: PolyAI/banking77
          split: test
        metrics:
          - type: accuracy
            value: {n["acc"]:.4f}
---

# distilroute: MiniLM-L6 distilled from gpt-oss-120b for banking intent routing

A 22M-parameter student that routes banking support messages to one of Banking77's 77 intents.
It **never saw a human label**: it was fine-tuned on {n["n_train"]:,} messages labelled zero-shot by
`gpt-oss-120b` (free tier on Groq), then quantised to int8 ONNX. This is the model the
[distilroute]({GITHUB}) service and demo run.

| | accuracy on Banking77 test (human labels) | latency p50 / p95, laptop CPU | $ per 1M messages |
|---|---:|---:|---:|
| **this model** (int8 ONNX) | **{n["acc"]:.3f}** (95 % CI {n["ci"][0]:.3f}–{n["ci"][1]:.3f}) | {n["p50"]:.1f} / {n["p95"]:.1f} ms | {n["usd"]:.2f} (AWS t3.small; $0 on your own hardware) |
| the LLM teacher, zero-shot | {n["teacher_acc"]:.3f} | API call | {n["teacher_usd"]:.2f} (one message per call) |
| this model + escalate to the LLM below {THRESHOLD} confidence | {n["cascade"]:.3f} | 3 ms for most messages | about a seventh of the LLM's |

It keeps {n["acc"] / n["teacher_acc"]:.0%} of its teacher's accuracy ({pt(n["gap"])} pt, paired
95 % CI {pt(n["gap_ci"][0])} to {pt(n["gap_ci"][1])}). The same architecture trained on all 10,003
human labels reaches {n["gold_ref"]:.3f}: the gap is the teacher's own error rate, passed on.

## Use

```python
import sys
from huggingface_hub import snapshot_download

path = snapshot_download("{repos["model"]}")
sys.path.insert(0, path)
from route import Router  # route.py in this repo: onnxruntime + tokenizers, no torch

router = Router(path)
router.route("{EXAMPLE}")
# {n["example"]}
```

`pip install onnxruntime tokenizers numpy huggingface_hub`. The tokenizer truncates at
{MAX_LENGTH} tokens, as in training.

**Confidence is calibrated** (temperature scaling, T = {j(MODELS / SERVED / "meta.json")["temperature"]:.3f} in
`config.json`, fitted on a held-out slice of the teacher labels), so a threshold means what it
says. The intended policy is a cascade: answer when confidence ≥ {THRESHOLD}, otherwise send the
message to an LLM. On Banking77 that is {n["cascade"]:.3f} overall, level with or above the LLM
alone (paired 95 % CI {pt(n["cascade_ci"][0])} to {pt(n["cascade_ci"][1])} pt) with {n["esc_clean"]:.0%} of messages
escalated.

**Off-topic messages:** `entropy` (spread of the prediction) separates out-of-scope messages
better than confidence does; see the repo's `docs/oos.md` for thresholds measured on CLINC150's
out-of-scope queries.

## Limitations

- **Typos hurt.** One typo costs {abs(n["typo1"]) * 100:.0f} pt, three typos {abs(n["typo3"]) * 100:.0f} pt: WordPiece
  splits a misspelt word into unfamiliar pieces. The model gets less sure too, so with three
  typos the cascade escalates {n["esc_typo3"]:.0%} of messages instead of {n["esc_clean"]:.0%}: noise turns into LLM
  cost more than into silent errors. Training on typo'd copies halves the loss (repo, roadmap
  6.6); this release is the un-augmented model.
- **Banking77 only**, English, 77 fixed intents. It will force every message into one of them;
  use the escalation threshold and entropy for anything else.
- **Teacher noise.** About 15 % of its training labels are wrong (the teacher is
  {n["teacher_acc"]:.1%} accurate). Checking 500 of them, chosen by where the student disagrees, lifts a
  student above the teacher (repo, `docs/correction.md`).

## Training

- Base: `sentence-transformers/all-MiniLM-L6-v2`, fine-tuned for classification (Colab T4),
  exported to ONNX and dynamically quantised to int8 (−0.5 pt vs fp32).
- Labels: `gpt-oss-120b`, zero-shot, intent names plus one-line descriptions, top-3 ranked,
  batches of 20 **shuffled** messages (sorted batches leak the answer: +8 pt, the repo explains).
  The labels are in [{repos["dataset"]}](https://huggingface.co/datasets/{repos["dataset"]}).
- Evaluation: accuracy against the human labels of the untouched Banking77 test split; intervals
  from a paired bootstrap (10,000 resamples).

Everything, including every number above, is reproducible from [the repository]({GITHUB}).

## Citation

Banking77: Casanueva et al., *Efficient Intent Detection with Dual Sentence Encoders*, NLP4ConvAI
2020 (CC-BY-4.0).
"""


def dataset_card(stats: dict, repos: dict) -> str:
    return f"""---
license: cc-by-4.0
language: en
task_categories:
  - text-classification
tags:
  - banking77
  - llm-labels
  - label-noise
  - knowledge-distillation
size_categories:
  - 1K<n<10K
configs:
  - config_name: default
    data_files:
      - split: train
        path: train.jsonl
      - split: test
        path: test.jsonl
---

# Banking77 labelled by gpt-oss-120b (zero-shot)

Banking77 messages with **two labels each**: the original human intent and the intent chosen by
`gpt-oss-120b` (Groq, zero-shot, reasoning effort low), with its ranked top 3. Made for
[distilroute]({GITHUB}), which trains small students on the LLM labels only; the human label is
there to measure them. Useful for studying LLM label noise without paying for the labels again.

| split | rows labelled | LLM agrees with the human label | human label in LLM's top 3 | unparsed |
|---|---:|---:|---:|---:|
| train | {stats["train"]["n"]:,} of 10,003 | {stats["train"]["acc"]:.3f} | {stats["train"]["top3"]:.3f} | {stats["train"]["unparsed"]:.1%} |
| test | {stats["test"]["n"]:,} of 3,080 | {stats["test"]["acc"]:.3f} | {stats["test"]["top3"]:.3f} | {stats["test"]["unparsed"]:.1%} |

The train split grows as the free tier allows (the rest of the 10,003 is being labelled); rows
are in the labeller's seed-0 shuffled order, so any prefix is a random sample.

## Fields

- `idx` — row number in the original Banking77 CSV of that split
- `text` — the message
- `label_human` — Banking77's label
- `label_llm` — the LLM's top-1 intent (`null` if its answer could not be parsed)
- `llm_top3` — the LLM's ranked top 3
- `model`, `reasoning`, `descriptions` — the labelling config (`descriptions`: version of the
  one-line intent descriptions in the prompt, from the repo)

## How the labels were made

Zero-shot prompt with the 77 intent names and a one-line description of each, 20 messages per
call, **shuffled**: the source files are sorted by intent, and batches of same-intent messages
let the LLM read the answer off its neighbours (0.948 sorted vs 0.867 shuffled on test).

## License and attribution

CC-BY-4.0, as Banking77: Casanueva et al., *Efficient Intent Detection with Dual Sentence
Encoders*, NLP4ConvAI 2020. The LLM labels are added under the same license.
"""


def space_readme(repos: dict) -> str:
    return f"""---
title: distilroute
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
license: apache-2.0
models:
  - {repos["model"]}
short_description: A 120B LLM distilled into 22M params for banking routing
---

A 22M-parameter student of `gpt-oss-120b` routing banking support messages to 77 intents, with
calibrated confidence and the cascade decision (send to the LLM when unsure).
Model: [{repos["model"]}](https://huggingface.co/{repos["model"]}) · Code: {GITHUB}
"""


def example_output(model: Path) -> str:
    """The card's usage example, run for real on the built files: nothing on it is typed in."""
    import importlib.util

    sys.dont_write_bytecode = True  # no __pycache__ in the folder that gets uploaded
    spec = importlib.util.spec_from_file_location("route", model / "route.py")
    route = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(route)
    r = route.Router(model).route(EXAMPLE)
    r["confidence"] = round(r["confidence"], 3)
    r["entropy"] = round(r["entropy"], 3)
    r["top3"] = [(c, round(p, 3)) for c, p in r["top3"]]
    return repr(r)


def build_dataset(out: Path) -> dict:
    out.mkdir(parents=True)
    stats = {}
    for split in ("train", "test"):
        df = load_labels(split)
        order = [
            json.loads(line)["idx"]
            for line in (LABELS / f"{split}.jsonl").open(encoding="utf-8")
            if line.strip()
        ]
        df = df.loc[order]
        gold = load_split(split).category
        rows = [
            {
                "idx": int(i),
                "text": r.text,
                "label_human": gold[i],
                "label_llm": r.teacher,
                "llm_top3": list(r.ranked)[:3],
                "model": r.model,
                "reasoning": r.reasoning,
                "descriptions": r.descriptions,
            }
            for i, r in df.iterrows()
        ]
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n = len(rows)
        stats[split] = {
            "n": n,
            "acc": sum(r["label_llm"] == r["label_human"] for r in rows) / n,
            "top3": sum(r["label_human"] in r["llm_top3"] for r in rows) / n,
            "unparsed": sum(r["label_llm"] is None for r in rows) / n,
        }
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--user", required=True, help="Hugging Face username or organisation")
    ap.add_argument("--publish", action="store_true", help="upload the three repos (public)")
    args = ap.parse_args()
    repos = {
        "model": f"{args.user}/distilroute-minilm-banking77",
        "dataset": f"{args.user}/banking77-gpt-oss-labels",
        "space": f"{args.user}/distilroute",
    }

    if OUT.exists():
        shutil.rmtree(OUT)
    n = numbers()

    model = OUT / "model"
    model.mkdir(parents=True)
    src = MODELS / SERVED
    for f in ("model.int8.onnx", "tokenizer.json", "tokenizer_config.json", "classes.json"):
        shutil.copy(src / f, model / f)
    meta = j(src / "meta.json")
    config = {
        "base_model": meta["hf_model"],
        "temperature": meta["temperature"],
        "max_length": MAX_LENGTH,
        "escalate_below": THRESHOLD,
        "labels": "gpt-oss-120b zero-shot (Banking77 train)",
    }
    (model / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (model / "route.py").write_text(ROUTE_PY, encoding="utf-8")
    n["example"] = example_output(model)
    (model / "README.md").write_text(model_card(n, repos), encoding="utf-8")
    shutil.copy(ROOT / "LICENSE", model / "LICENSE")

    stats = build_dataset(OUT / "dataset")
    (OUT / "dataset" / "README.md").write_text(dataset_card(stats, repos), encoding="utf-8")

    space = OUT / "space"
    space.mkdir()
    (space / "route.py").write_text(ROUTE_PY, encoding="utf-8")
    (space / "app.py").write_text(
        APP_PY.format(
            model_repo=repos["model"],
            model_url=f"https://huggingface.co/{repos['model']}",
            github=GITHUB,
            keep=f"{n['acc'] / n['teacher_acc']:.0%}",
            acc=f"{n['acc']:.3f}",
            teacher_acc=f"{n['teacher_acc']:.3f}",
            threshold=THRESHOLD,
        ),
        encoding="utf-8",
    )
    (space / "requirements.txt").write_text(
        "onnxruntime\ntokenizers\nnumpy\nhuggingface_hub\n", encoding="utf-8"
    )
    (space / "README.md").write_text(space_readme(repos), encoding="utf-8")
    print(f"built {OUT.relative_to(ROOT).as_posix()}/{{model,dataset,space}} for {args.user}")
    for split, s in stats.items():
        print(f"  dataset {split}: {s['n']:,} rows, LLM vs human {s['acc']:.3f}")

    if args.publish:
        from huggingface_hub import HfApi

        api = HfApi()
        for kind, folder in (("model", model), ("dataset", OUT / "dataset"), ("space", space)):
            repo_type = None if kind == "model" else kind
            extra = {"space_sdk": "gradio"} if kind == "space" else {}
            api.create_repo(repos[kind], repo_type=repo_type, exist_ok=True, **extra)
            api.upload_folder(repo_id=repos[kind], repo_type=repo_type, folder_path=folder)
            prefix = {"model": "", "dataset": "datasets/", "space": "spaces/"}[kind]
            print(f"  published https://huggingface.co/{prefix}{repos[kind]}")


if __name__ == "__main__":
    main()
