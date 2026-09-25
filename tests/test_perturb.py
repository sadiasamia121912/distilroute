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


def test_augment_adds_changed_copies_with_the_same_label():
    import pandas as pd

    from distilroute.perturb import augment

    train = pd.DataFrame({"text": ["I still have not received my new card", "hi"], "y": ["a", "b"]})
    out = augment(train, ["typo1", "typo3"])
    copies = out.iloc[len(train) :]
    assert len(copies) == 2  # "hi" has no 3-letter word, so the noise leaves it alone
    assert (copies.y == "a").all() and (copies.text != train.text[0]).all()
    # training copies never share a draw with the test copies (seed 0)
    assert perturb(train.text[0], "typo1", 0) not in set(copies.text)
