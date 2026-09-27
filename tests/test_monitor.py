"""The traffic monitor: flags an intent whose share jumps, and stays quiet otherwise."""

from __future__ import annotations

import numpy as np

from distilroute.monitor import IntentMonitor, shift_z_array

CLASSES = [f"intent_{i}" for i in range(20)]


def stream(rng, n, shares):
    return rng.choice(CLASSES, size=n, p=shares).tolist()


def test_stable_traffic_raises_no_flag():
    rng = np.random.default_rng(0)
    uniform = np.full(len(CLASSES), 1 / len(CLASSES))
    mon = IntentMonitor(CLASSES, reference=5000, window=2000)
    for intent in stream(rng, 7000, uniform):
        mon.add(intent)
    report = mon.report()
    assert report["state"] == "watching"
    assert report["flagged"] == []


def test_a_surge_into_one_intent_is_flagged_by_name():
    rng = np.random.default_rng(1)
    uniform = np.full(len(CLASSES), 1 / len(CLASSES))
    surge = uniform * 0.85
    surge[3] += 0.15  # a new intent's traffic lands on intent_3
    mon = IntentMonitor(CLASSES, reference=5000, window=2000)
    for intent in stream(rng, 5000, uniform) + stream(rng, 2000, surge):
        mon.add(intent)
    flagged = mon.report()["flagged"]
    assert flagged and flagged[0]["intent"] == "intent_3"
    assert flagged[0]["window_share"] > flagged[0]["reference_share"]


def test_phases_before_watching():
    mon = IntentMonitor(CLASSES, reference=3, window=2)
    assert mon.report()["state"].startswith("collecting the reference")
    for intent in CLASSES[:4]:
        mon.add(intent)
    assert mon.report()["state"].startswith("filling the window")
    mon.add(None)  # an unparsed answer is not traffic
    assert mon.report()["window"] == 1


def test_z_is_zero_for_identical_shares_and_positive_for_a_rise():
    ref = np.array([100.0, 100.0, 100.0])
    assert np.allclose(shift_z_array(ref, ref), 0)
    z = shift_z_array(ref, np.array([60.0, 20.0, 20.0]))
    assert z[0] > 0 > z[1]
