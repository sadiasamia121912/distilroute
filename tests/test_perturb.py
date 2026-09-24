"""Perturbations are deterministic, change what they claim to, and nothing else."""

from __future__ import annotations

import pytest

from distilroute.perturb import KINDS, perturb

QUERY = "Why is my card payment still pending? Please help, it's for my account."


@pytest.mark.parametrize("kind", KINDS)
def test_deterministic_and_different(kind):
    assert perturb(QUERY, kind) == perturb(QUERY, kind)
    assert perturb(QUERY, kind) != QUERY
    assert perturb(QUERY, kind, seed=1) == perturb(QUERY, kind, seed=1)


def test_typo_counts_and_first_letters():
    for kind, n in (("typo1", 1), ("typo3", 3)):
        a, b = QUERY.split(), perturb(QUERY, kind).split()
        assert len(a) == len(b)
        changed = [(x, y) for x, y in zip(a, b, strict=True) if x != y]
        assert 1 <= len(changed) <= n
        assert all(x[0] == y[0] for x, y in changed)


def test_chat_slang_wrap():
    assert perturb(QUERY, "chat") == (
        "why is my card payment still pending please help its for my account"
    )
    assert perturb(QUERY, "slang") == (
        "y is my card payment still pending? pls help, it's 4 my acct."
    )
    wrapped = perturb(QUERY, "wrap")
    assert QUERY in wrapped and not wrapped.startswith(QUERY) and not wrapped.endswith(QUERY)


def test_short_words_untouched_by_typos():
    assert perturb("to be or", "typo3") == "to be or"
