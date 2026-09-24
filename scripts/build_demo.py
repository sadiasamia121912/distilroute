"""Build the data files for the in-browser demo page, `demo/index.html` (roadmap 6.7).

    python scripts/build_demo.py            # -> demo/*.json, demo/*.txt, demo/*.wasm

The page runs the served student (fine-tuned MiniLM, int8 ONNX) in the browser with
onnxruntime-web, so it needs next to it:

- `ort-wasm-simd-threaded.wasm` — the WebAssembly runtime, downloaded from npm at the pinned
  version the page imports (an artifact page may load scripts from a CDN but not fetch files
  from one, so the binary is published alongside the page);
- `model.part{1,2,3}.txt` — the ONNX graph as base64, in parts under the 15 MB per-file limit
  (artifact pages serve text, not arbitrary binaries);
- `vocab.json` — the WordPiece vocabulary, id-ordered; the page tokenizes in JavaScript;
- `demo.json` — classes, the fitted temperature, the escalation threshold, the out-of-scope
  entropy threshold (measured here, on this exact model), cost per 1M requests, and reference
  outputs from the Python service that the page re-computes as a self-check.

The generated files are gitignored (45 MB); the page itself is committed.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distilroute import students  # noqa: E402
from distilroute.data import MODELS, RAW, RESULTS, ROOT, load_split, rel  # noqa: E402

MODEL = "minilm_ft_teacher"
ORT_VERSION = "1.30.0"  # must match the import in demo/index.html
WASM = "ort-wasm-simd-threaded.wasm"
PARTS = 3
ESCALATE_BELOW = 0.8
OOS_BUDGET = 0.10  # share of in-scope messages the out-of-scope flag may catch
CHECKS = [
    "My card still hasn't arrived, it's been 2 weeks!",
    "why was i charged a fee for topping up",
    "Can I get a refund for this item?",
    "whats the weather like in paris tomorrow",
    "I think someone has my card details, pls freeze it",
    "ATM swallowed my card",
]
OUT = ROOT / "demo"


def entropy(p: np.ndarray) -> float:
    return float(-(p * np.log(p + 1e-12)).sum())


def main() -> None:
    path = MODELS / MODEL
    router = students.load(MODEL)
    meta = json.loads((path / "meta.json").read_text())
    OUT.mkdir(exist_ok=True)
    assets = []

    vocab_map = json.loads((path / "tokenizer.json").read_text(encoding="utf-8"))["model"]["vocab"]
    vocab = [None] * len(vocab_map)
    for tok, i in vocab_map.items():
        vocab[i] = tok
    (OUT / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    assets.append(["vocab.json", (OUT / "vocab.json").stat().st_size])

    if not (OUT / WASM).exists():
        url = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ORT_VERSION}/dist/{WASM}"
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        (OUT / WASM).write_bytes(r.content)
    assets.append([WASM, (OUT / WASM).stat().st_size])

    graph = (path / "model.int8.onnx").read_bytes()
    size = -(-len(graph) // PARTS)
    size += (-size) % 3  # a multiple of 3 bytes, so each part decodes on its own
    for i in range(PARTS):
        name = f"model.part{i + 1}.txt"
        (OUT / name).write_bytes(base64.b64encode(graph[i * size : (i + 1) * size]))
        assets.append([name, (OUT / name).stat().st_size])

    # Out-of-scope threshold: the entropy that flags OOS_BUDGET of real banking messages.
    raw = lambda t: router.proba(t)[1]  # noqa: E731 — uncalibrated, as in scripts/oos.py
    e_in = np.array([entropy(raw(t)) for t in load_split("test").text])
    threshold = float(np.quantile(e_in, 1 - OOS_BUDGET))
    e_oos = np.array([entropy(raw(t)) for t in pd.read_csv(RAW / "clinc_oos.csv").text])
    caught = float((e_oos > threshold).mean())

    checks = []
    for text in CHECKS:
        routed = router.route(text)
        ids = router.tokenizer([text], truncation=True, max_length=64)["input_ids"][0]
        checks.append(
            {"text": text, "ids": ids, "intent": routed.intent, "conf": round(routed.confidence, 4)}
        )

    cost = json.loads((RESULTS / "cost.json").read_text())
    vm_per_s = cost["prices"]["vm"]["usd_per_hour"] / 3600
    cfg = {
        "model": MODEL,
        "classes": router.classes,
        "temperature": meta["temperature"],
        "escalate_below": ESCALATE_BELOW,
        "oos_entropy": threshold,
        "oos_caught": caught,
        "usd_per_1m": {
            "llm": cost["rows"]["teacher_single"]["usd_per_1m"],
            "student": 1e6 * cost["rows"][MODEL]["cpu_seconds_per_query"] * vm_per_s,
        },
        "checks": checks,
        "assets": assets,
        "model_bytes": len(graph),
    }
    (OUT / "demo.json").write_text(json.dumps(cfg), encoding="utf-8")
    total = sum(a[1] for a in assets) / 1e6
    print(f"OOS entropy threshold {threshold:.3f} catches {caught:.1%} of CLINC150 out-of-scope")
    print(f"-> {rel(OUT)}/ ({total:.1f} MB of assets + demo.json)")


if __name__ == "__main__":
    main()
