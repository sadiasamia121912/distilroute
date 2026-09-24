"""Rule-based noise for robustness tests (roadmap 6.6): real tickets are typed in a hurry.

Every perturbation is deterministic in (kind, text, seed), so the students and the teacher see
the exact same noisy query, and a re-run reproduces it. No model is involved, so nothing here
can leak a label.

- `typo1`  — one typo in one word: adjacent swap, dropped letter, doubled letter, or a
             neighbouring key on a QWERTY keyboard.
- `typo3`  — three typos, in three different words where the query has them.
- `chat`   — lowercase, no punctuation (apostrophes included: "dont").
- `slang`  — texting abbreviations: you → u, please → pls, account → acct, ...
- `wrap`   — a greeting in front and a sign-off after: "hi team, <query> thanks!".
"""

from __future__ import annotations

import random
import re

KINDS = ["typo1", "typo3", "chat", "slang", "wrap"]

_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
_NEIGHBOURS: dict[str, str] = {}
for r, row in enumerate(_ROWS):
    for c, ch in enumerate(row):
        near = [row[i] for i in (c - 1, c + 1) if 0 <= i < len(row)]
        for rr in (r - 1, r + 1):  # the keys above and below, roughly
            if 0 <= rr < len(_ROWS):
                near += [_ROWS[rr][i] for i in (c - 1, c) if 0 <= i < len(_ROWS[rr])]
        _NEIGHBOURS[ch] = "".join(near)

SLANG = {
    "you": "u",
    "your": "ur",
    "you're": "ur",
    "are": "r",
    "please": "pls",
    "thanks": "thx",
    "because": "cuz",
    "account": "acct",
    "money": "$",
    "tomorrow": "tmrw",
    "today": "2day",
    "to": "2",
    "for": "4",
    "why": "y",
    "okay": "ok",
    "information": "info",
    "message": "msg",
    "transaction": "txn",
    "people": "ppl",
}
GREETINGS = ["hi,", "hello!", "hey team,", "good morning,", "hi there,"]
SIGN_OFFS = ["thanks!", "thank you", "cheers", "pls help asap", "thx in advance"]


def _typo(word: str, rng: random.Random) -> str:
    """One typo in `word`, never touching its first letter."""
    i = rng.randrange(1, len(word))
    op = rng.choice(["swap", "drop", "double", "neighbour"])
    if op == "swap" and i < len(word) - 1:
        return word[:i] + word[i + 1] + word[i] + word[i + 2 :]
    if op == "drop":
        return word[:i] + word[i + 1 :]
    if op == "double":
        return word[:i] + word[i] + word[i:]
    near = _NEIGHBOURS.get(word[i].lower())
    if near:
        return word[:i] + rng.choice(near) + word[i + 1 :]
    return word[:i] + word[i + 1 :]  # not a letter: drop it


def _typos(text: str, n: int, rng: random.Random) -> str:
    """Typos in `n` distinct words of 3+ letters (punctuation around a word is kept as is)."""
    spans = [m.span() for m in re.finditer(r"[A-Za-z]{3,}", text)]
    for start, end in sorted(rng.sample(spans, min(n, len(spans))), reverse=True):
        text = text[:start] + _typo(text[start:end], rng) + text[end:]
    return text


def _slang(text: str) -> str:
    def sub(m: re.Match) -> str:
        return SLANG.get(m.group(0).lower(), m.group(0))

    return re.sub(r"[A-Za-z']+", sub, text)


def perturb(text: str, kind: str, seed: int = 0) -> str:
    rng = random.Random(f"{seed}:{kind}:{text}")
    if kind == "typo1":
        return _typos(text, 1, rng)
    if kind == "typo3":
        return _typos(text, 3, rng)
    if kind == "chat":
        return " ".join(re.sub(r"[^\w\s]", "", text.lower()).split())
    if kind == "slang":
        return _slang(text)
    if kind == "wrap":
        return f"{rng.choice(GREETINGS)} {text} {rng.choice(SIGN_OFFS)}"
    raise ValueError(f"unknown perturbation {kind!r}; choose from {KINDS}")
