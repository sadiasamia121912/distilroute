"""Temperature scaling and ECE, so a student's "90 % confident" can be trusted (roadmap 1b.2).

The cascade escalates a query to the LLM when the student's confidence is below a threshold.
That only works if confidence means what it says, and logistic heads and fine-tuned
transformers are typically over-confident. Temperature scaling (Guo et al., 2017) fixes the
worst of it with one scalar T fit on a held-out calibration set: p_cal = softmax(log p / T).
T is fit on *training-pool* labels (teacher or gold, whatever the student trained on) held
out from fitting — never on the test split.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def apply_temperature(proba: np.ndarray, t: float) -> np.ndarray:
    z = np.log(np.clip(proba, 1e-12, None)) / t
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(proba: np.ndarray, y_idx: np.ndarray) -> float:
    """T minimising negative log-likelihood of the calibration labels; T > 1 softens."""

    def nll(t: float) -> float:
        p = apply_temperature(proba, t)
        return float(-np.log(p[np.arange(len(y_idx)), y_idx] + 1e-12).mean())

    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)


def ece(conf: np.ndarray, hit: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: |accuracy - confidence| averaged over equal-width bins."""
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(total)
