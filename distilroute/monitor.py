"""Notice a new intent the router was never taught, from the mix of intents it routes.

A new banking intent (a product launch, a new fee) is hard to catch message by message: its
queries look like banking, and the router files most of them under one neighbouring intent with
ordinary confidence (docs/new_intents.md: 53 % caught at a 10 % escalation budget, and no
per-message score does much better). What does change is the *mix*: that neighbour's share of
traffic jumps. This watches for exactly that.

The first `reference` routed messages set each intent's share. After that, the last `window`
messages are compared with them intent by intent, with a two-sample test for proportions:
z = (p_window - p_reference) / sqrt(p (1 - p) (1/n_window + 1/n_reference)), p pooled. Both
samples are noisy (5,000 messages is ~65 per intent), and a test that treated the reference as
exact raised 11-20 % false alarms in simulation. An intent is flagged when z exceeds `z`, by
default the one-sided Bonferroni threshold for 77 intents at 1 % (3.65); measured on stable
traffic that gives 1.6 % false alarms per check with the default 2,000-message window (3.9 % at
250, where counts are small). The report names the flagged intents: where to look.
"""

from __future__ import annotations

import threading
from collections import Counter, deque

import numpy as np

Z_DEFAULT = 3.65  # one-sided, alpha 0.01 / 77 intents


def shift_z_array(reference: np.ndarray, window: np.ndarray) -> np.ndarray:
    """Per-intent two-sample z-scores: is the intent's share higher in the window?

    `reference` and `window` are (..., C) counts, so many simulated streams can be scored at
    once. The service and the experiment in scripts/new_intents.py both use this. A half count
    is added to both sides so an intent unseen in either still gets a finite score.
    """
    ref, win = reference + 0.5, window + 0.5
    n_ref, n_win = ref.sum(-1, keepdims=True), win.sum(-1, keepdims=True)
    pooled = (ref + win) / (n_ref + n_win)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n_ref + 1 / n_win))
    return (win / n_win - ref / n_ref) / se


def shift_z(reference: Counter, window: Counter, classes: list[str]) -> dict[str, float]:
    ref = np.array([reference.get(c, 0) for c in classes], dtype=float)
    win = np.array([window.get(c, 0) for c in classes], dtype=float)
    return dict(zip(classes, shift_z_array(ref, win).tolist(), strict=True))


class IntentMonitor:
    def __init__(
        self, classes: list[str], reference: int = 5000, window: int = 2000, z: float = Z_DEFAULT
    ):
        self.classes, self.reference_size, self.z = list(classes), reference, z
        self.reference: Counter = Counter()
        self.recent: deque = deque(maxlen=window)
        self.lock = threading.Lock()

    def add(self, intent: str | None) -> None:
        if intent is None:
            return
        with self.lock:
            if sum(self.reference.values()) < self.reference_size:
                self.reference[intent] += 1
            else:
                self.recent.append(intent)

    def report(self) -> dict:
        with self.lock:
            n_ref, recent = sum(self.reference.values()), list(self.recent)
            reference = Counter(self.reference)
        out = {
            "reference": n_ref,
            "window": len(recent),
            "window_size": self.recent.maxlen,
            "z_threshold": self.z,
            "flagged": [],
        }
        if n_ref < self.reference_size:
            out["state"] = f"collecting the reference ({n_ref} of {self.reference_size})"
            return out
        if len(recent) < self.recent.maxlen:
            out["state"] = f"filling the window ({len(recent)} of {self.recent.maxlen})"
            return out
        window = Counter(recent)
        zs = shift_z(reference, window, self.classes)
        out["state"] = "watching"
        out["flagged"] = [
            {
                "intent": c,
                "z": round(z, 2),
                "window_share": round(window.get(c, 0) / len(recent), 4),
                "reference_share": round(reference.get(c, 0) / n_ref, 4),
            }
            for c, z in sorted(zs.items(), key=lambda kv: -kv[1])
            if z > self.z
        ]
        return out
