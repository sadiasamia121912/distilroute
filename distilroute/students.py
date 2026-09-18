"""Load any trained router from models/<name>/ behind one interface, for serving and benchmarks.

Every student script leaves a `meta.json` (see `runs.model_dir`) whose `kind` picks the loader:

- `tfidf`  — a pickled sklearn pipeline (`model.joblib`)
- `minilm` — a sentence-transformer encoder (hub name, or a fine-tuned copy in `encoder/`)
             plus a logistic head (`head.joblib`)
- `onnx`   — a tokenizer + `model.int8.onnx` from `scripts/finetune.py`; needs only onnxruntime

plus `teacher`, the LLM itself through the same interface (one query per call, so its latency
is what a request would actually cost). Heavy imports happen inside the loaders so the service
only pays for the models it is asked for.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from distilroute.calibration import apply_temperature
from distilroute.data import MODELS, ROOT, categories

TEACHER = "teacher"


@dataclass
class Routed:
    intent: str | None
    confidence: float | None  # probability of `intent`; None for the teacher (no probabilities)
    ranked: list[str]  # top-3, best first


class Router:
    name: str
    kind: str
    temperature: float | None = None  # from meta.json; confidence is calibrated when set

    def proba(self, text: str) -> tuple[list[str], np.ndarray]:  # pragma: no cover - abstract
        raise NotImplementedError

    def route(self, text: str) -> Routed:
        classes, p = self.proba(text)
        if self.temperature:
            p = apply_temperature(p[None, :], self.temperature)[0]
        top = np.argsort(-p)[:3]
        return Routed(classes[top[0]], float(p[top[0]]), [classes[i] for i in top])


class TfidfRouter(Router):
    kind = "tfidf"

    def __init__(self, path: Path, meta: dict):
        import joblib

        self.name, self.temperature = meta["name"], meta.get("temperature")
        self.model = joblib.load(path / "model.joblib")
        self.classes = list(self.model.classes_)

    def proba(self, text: str):
        return self.classes, self.model.predict_proba([text])[0]


class MiniLMRouter(Router):
    kind = "minilm"

    def __init__(self, path: Path, meta: dict):
        import joblib
        from sentence_transformers import SentenceTransformer

        self.name, self.temperature = meta["name"], meta.get("temperature")
        enc = meta["encoder"]
        self.encoder = SentenceTransformer(
            str(path / enc) if (path / enc).is_dir() else enc, device="cpu"
        )
        self.head = joblib.load(path / "head.joblib")
        self.classes = list(self.head.classes_)

    def proba(self, text: str):
        x = self.encoder.encode([text], show_progress_bar=False)
        return self.classes, self.head.predict_proba(x)[0]


class OnnxRouter(Router):
    kind = "onnx"

    def __init__(self, path: Path, meta: dict):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.name, self.temperature = meta["name"], meta.get("temperature")
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.session = ort.InferenceSession(
            str(path / "model.int8.onnx"), providers=["CPUExecutionProvider"]
        )
        self.classes = json.loads((path / "classes.json").read_text())

    def proba(self, text: str):
        enc = self.tokenizer([text], truncation=True, max_length=64, return_tensors="np")
        feed = {k: np.asarray(enc[k]).astype("int64") for k in ("input_ids", "attention_mask")}
        logits = self.session.run(["logits"], feed)[0][0]
        z = np.exp(logits - logits.max())
        return self.classes, z / z.sum()


class TeacherRouter(Router):
    """The LLM through the same interface: the final config (v2 descriptions, top-3, low)."""

    kind = name = TEACHER

    def __init__(self, provider: str = "groq"):
        from dotenv import load_dotenv

        from distilroute.teacher import Teacher

        load_dotenv(ROOT / ".env")
        raw = json.loads((ROOT / "data" / "intent_descriptions.json").read_text(encoding="utf-8"))
        self.teacher = Teacher(
            labels=categories(),
            provider=provider,
            descriptions={k: v for k, v in raw.items() if not k.startswith("_")},
            reasoning_effort="low",
            top_k=3,
        )

    def route(self, text: str) -> Routed:
        r = self.teacher.label_batch([text])[0]
        return Routed(r.label, None, r.ranked)


LOADERS = {"tfidf": TfidfRouter, "minilm": MiniLMRouter, "onnx": OnnxRouter}


def available() -> dict[str, dict]:
    """name -> meta for every loadable model under models/, plus the teacher if a key is set."""
    out = {}
    for meta_path in sorted(MODELS.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("kind") in LOADERS:
            out[meta["name"]] = meta
    if os.environ.get("GROQ_API_KEY") or (ROOT / ".env").exists():
        out[TEACHER] = {"kind": TEACHER, "name": TEACHER}
    return out


def load(name: str) -> Router:
    if name == TEACHER:
        return TeacherRouter()
    path = MODELS / name
    meta = json.loads((path / "meta.json").read_text())
    return LOADERS[meta["kind"]](path, meta)
