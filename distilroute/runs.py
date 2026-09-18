"""The contract every student follows, so `scripts/evaluate.py` can score them all alike."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from distilroute.data import MODELS, RESULTS


def save_run(name: str, metrics: dict, classes: list[str], proba: np.ndarray) -> None:
    """Metrics JSON + test-split probabilities (rows in test order, columns in `classes` order).

    The probabilities are what the cascade and calibration analysis need; nothing downstream
    retrains a model to get them.
    """
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(
        RESULTS / f"{name}_test_probs.npz",
        classes=np.array(classes),
        proba=proba.astype("float32"),
    )


def model_dir(name: str, kind: str, **meta) -> Path:
    """models/<name>/ with a meta.json saying how to load it (see `distilroute.students`).

    `kind` is one of tfidf | minilm | onnx; the rest of `meta` is whatever the loader needs
    (encoder name, classes). models/ is gitignored — these are rebuilt by the scripts.
    """
    d = MODELS / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps({"kind": kind, "name": name, **meta}, indent=2))
    return d


def latency_ms(
    predict_one: Callable[[str], object], texts: list[str], n: int = 500
) -> tuple[float, float]:
    """p50 / p95 wall-clock per single-query call, the way a service would see it."""
    times = []
    for t in texts[:n]:
        t0 = time.perf_counter()
        predict_one(t)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(times, 50)), float(np.percentile(times, 95))
