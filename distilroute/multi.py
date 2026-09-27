"""A second request in the same message: the `also` field of `POST /route`.

The router answers one intent per message. When a message has several sentences, each is
routed on its own as well, and any other intent a sentence is confident about comes back as
`also`, best first. The main `intent` is still the whole-message answer. Single-sentence
messages cost nothing extra.

The threshold trades finding the second request against inventing one: small talk ("hi there,
hope you are well") can get a confident wrong intent from a fine-tuned student. scripts/stress.py
measures both, per threshold, on two-question messages and on single-question long tickets
(docs/stress.md), and ALSO_THRESHOLD is set from that.
"""

from __future__ import annotations

import re

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
ALSO_THRESHOLD = 0.95  # calibrated confidence a sentence needs; chosen in docs/stress.md


def sentences(text: str) -> list[str]:
    return [s for s in SENTENCE_END.split(text.strip()) if s]


def also_intents(
    route, text: str, main: str | None, threshold: float = ALSO_THRESHOLD
) -> list[str]:
    """Other intents, one per sentence at most, whose sentence's confidence reaches `threshold`.

    `route` is a router's `route` method (anything returning `.intent` and `.confidence`).
    """
    parts = sentences(text)
    if len(parts) < 2:
        return []
    found: dict[str, float] = {}
    for part in parts:
        r = route(part)
        if r.intent and r.intent != main and (r.confidence or 0) >= threshold:
            found[r.intent] = max(found.get(r.intent, 0.0), r.confidence)
    return sorted(found, key=found.get, reverse=True)
