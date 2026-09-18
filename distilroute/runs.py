"""The contract every student follows, so `scripts/evaluate.py` can score them all alike."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from distilroute.calibration import fit_temperature
from distilroute.data import MODELS, RESULTS

CALIB_FRAC = 0.1


def split_calib(train: pd.DataFrame, frac: float = CALIB_FRAC, seed: int = 0):
    """Hold out a stratified `frac` of the training pool to fit the temperature on.

    The student is fit on the rest. The held-out rows carry the same labels the student trains
    on (teacher or gold), so calibration never touches the test split.
    """
    calib = train.groupby("y", group_keys=False).sample(frac=frac, random_state=seed)
    return train.drop(calib.index), calib


def save_run(
    name: str,
    metrics: dict,
    classes: list[str],
    proba: np.ndarray,
    calib: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Metrics JSON + test-split probabilities (rows in test order, columns in `classes` order).

    The probabilities are what the cascade and calibration analysis need; nothing downstream
    retrains a model to get them. `calib` = (probabilities on the held-out calibration rows,
    their label indices in `classes` order); when given, the fitted temperature goes into the
    metrics and `scripts/evaluate.py` applies it to the raw test probabilities. Returns the
    metrics as written, so the caller can put the temperature into the model's meta.json.
    """
    if calib is not None:
        p_cal, y_cal = calib
        metrics = {**metrics, "temperature": fit_temperature(p_cal, y_cal), "n_calib": len(y_cal)}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(
        RESULTS / f"{name}_test_probs.npz",
        classes=np.array(classes),
        proba=proba.astype("float32"),
    )
    return metrics


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
