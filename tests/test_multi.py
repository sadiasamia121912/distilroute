"""Sentence splitting and the `also` field's selection rule."""

from __future__ import annotations

from types import SimpleNamespace

from distilroute.multi import also_intents, sentences

ANSWERS = {
    "My card is late.": ("card_arrival", 0.97),
    "Also, what is the exchange rate?": ("exchange_rate", 0.96),
    "Thanks!": ("card_arrival", 0.99),  # same as the main intent: not repeated
    "Hi there.": ("top_up_failed", 0.60),  # small talk, unsure: below the threshold
}


def route(text):
    intent, conf = ANSWERS[text]
    return SimpleNamespace(intent=intent, confidence=conf)


def test_sentences_split_on_end_punctuation():
    assert sentences("Hi there. My card is late!  Why?") == [
        "Hi there.",
        "My card is late!",
        "Why?",
    ]
    assert sentences("no punctuation at all") == ["no punctuation at all"]


def test_also_keeps_confident_other_intents_only():
    text = "Hi there. My card is late. Also, what is the exchange rate? Thanks!"
    assert also_intents(route, text, "card_arrival", 0.95) == ["exchange_rate"]
    assert also_intents(route, text, "card_arrival", 0.5) == ["exchange_rate", "top_up_failed"]
    assert also_intents(route, "My card is late.", "card_arrival", 0.5) == []
