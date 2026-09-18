import numpy as np
import pandas as pd

from distilroute.calibration import apply_temperature, ece, fit_temperature
from distilroute.runs import split_calib


def overconfident(n=4000, k=5, sharpen=3.0, seed=0):
    """Softmax of calibrated logits multiplied by `sharpen`: right as often, far more sure."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 1, (n, k))
    logits[:, 0] += 1.5
    p_true = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(k, p=row) for row in p_true])  # labels drawn from the true p
    z = logits * sharpen
    p = np.exp(z - z.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True), y


def test_temperature_identity_and_argmax_preserved():
    p, _ = overconfident()
    assert np.allclose(apply_temperature(p, 1.0), p)
    assert (apply_temperature(p, 3.0).argmax(1) == p.argmax(1)).all()


def test_fit_temperature_softens_overconfident_model_and_lowers_ece():
    p, y = overconfident()
    t = fit_temperature(p, y)
    assert 2.5 < t < 3.5  # recovers the sharpening factor
    hit = p.argmax(1) == y
    assert ece(apply_temperature(p, t).max(1), hit) < ece(p.max(1), hit) / 2


def test_fit_temperature_leaves_a_calibrated_model_alone():
    p, y = overconfident(sharpen=1.0, seed=1)
    assert 0.85 < fit_temperature(p, y) < 1.15


def test_split_calib_is_stratified_and_disjoint():
    df = pd.DataFrame({"text": [str(i) for i in range(300)], "y": ["a"] * 200 + ["b"] * 100})
    train, calib = split_calib(df, frac=0.1, seed=0)
    assert len(calib) == 30 and calib.y.value_counts().to_dict() == {"a": 20, "b": 10}
    assert set(train.index).isdisjoint(calib.index) and len(train) + len(calib) == 300
