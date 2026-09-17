"""Teacher tests run against a fake transport — no key, no network."""

from __future__ import annotations

import json

import pytest

from distilroute.teacher import Teacher

LABELS = ["card_arrival", "lost_or_stolen_card", "top_up_failed", "exchange_rate"]


def fake(answers: dict[str, str] | str, calls: list | None = None):
    """A transport that answers every request with the same JSON (or raw text) and logs payloads."""

    def transport(payload: dict) -> tuple[str, dict]:
        if calls is not None:
            calls.append(payload)
        text = answers if isinstance(answers, str) else json.dumps(answers)
        return text, {"prompt_tokens": 100, "completion_tokens": 10}

    return transport


def make(answers, calls=None) -> Teacher:
    return Teacher(labels=LABELS, provider="groq", transport=fake(answers, calls))


def test_prompt_lists_names_only_and_numbers_queries():
    t = make({})
    system, user = t.build_messages(["my card never came", "rate for euros?"])
    for name in LABELS:
        assert f"- {name}" in system
    assert user == "1. my card never came\n2. rate for euros?"
    # No gold example queries anywhere in the prompt — the zero-shot promise.
    assert "never came" not in system


def test_parse_happy_path_and_normalisation():
    t = make({})
    parsed = t.parse('{"1": "card_arrival", "2": " Lost or stolen card "}', 2)
    assert parsed[1] == ("card_arrival", "card_arrival", ["card_arrival"])
    assert parsed[2][0] == "lost_or_stolen_card"


def test_parse_returns_canonical_case_for_mixed_case_labels():
    t = Teacher(
        labels=["Refund_not_showing_up", "card_arrival"], provider="groq", transport=fake({})
    )
    assert (
        t.parse('{"1": "refund_not_showing_up", "2": "Card Arrival"}', 2)[1][0]
        == "Refund_not_showing_up"
    )
    assert t.parse('{"1": "Refund_not_showing_up"}', 1)[1][0] == "Refund_not_showing_up"


def test_parse_rejects_unknown_labels_and_keeps_raw():
    t = make({})
    parsed = t.parse('{"1": "card_delivery_delay"}', 1)
    assert parsed[1] == (None, "card_delivery_delay", [])


def test_parse_survives_fences_and_garbage():
    t = make({})
    fenced = '```json\n{"1": "top_up_failed"}\n```'
    assert t.parse(fenced, 1)[1][0] == "top_up_failed"
    assert t.parse("Sure! Here you go.", 2) == {1: (None, "", []), 2: (None, "", [])}
    assert t.parse('{"1": "top_up_failed"', 1)[1][0] is None  # truncated JSON


def test_label_batch_retries_missing_items_individually():
    calls: list[dict] = []
    # Batch answer covers only item 1; item 2 is retried alone and gets the same answer key "1".
    t = make({"1": "exchange_rate"}, calls)
    res = t.label_batch(["a", "b"])
    assert [r.label for r in res] == ["exchange_rate", "exchange_rate"]
    assert t.calls == 2 and len(calls) == 2
    assert calls[1]["messages"][1]["content"] == "1. b"
    assert t.tokens_in == 200 and t.tokens_out == 20


def test_single_item_batch_does_not_retry_forever():
    t = make("nonsense")
    res = t.label_batch(["a"])
    assert res[0].label is None and t.calls == 1


def test_payload_shapes():
    t = make({})
    p = t._payload("sys", "usr")
    assert p["response_format"] == {"type": "json_object"} and p["temperature"] == 0.0
    g = Teacher(labels=LABELS, provider="gemini", transport=fake({}))
    gp = g._payload("sys", "usr")
    assert gp["generationConfig"]["responseMimeType"] == "application/json"
    assert g.model == "gemini-2.0-flash"


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        Teacher(labels=LABELS, provider="groq")


def test_descriptions_go_in_prompt_but_stay_optional():
    t = Teacher(
        labels=LABELS,
        provider="groq",
        transport=fake({}),
        descriptions={"card_arrival": "card not here yet"},
    )
    system, _ = t.build_messages(["x"])
    assert "- card_arrival: card not here yet" in system
    assert "- top_up_failed" in system  # a label without a description is still listed


def test_all_banking77_intents_have_a_description():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    desc = json.loads((root / "data" / "intent_descriptions.json").read_text(encoding="utf-8"))
    cats = json.loads((root / "data" / "raw" / "categories.json").read_text(encoding="utf-8"))
    assert set(cats) <= set(desc)
    # No dataset query may appear verbatim in a description (zero-shot promise).
    import pandas as pd

    train = pd.read_csv(root / "data" / "raw" / "train.csv").text.str.lower()
    for d in desc.values():
        assert d.lower() not in set(train)


def test_top_k_prompt_and_ranked_parsing():
    t = Teacher(labels=LABELS, provider="groq", transport=fake({}), top_k=3)
    system, _ = t.build_messages(["x"])
    assert "list of the 3 most likely intent names" in system
    parsed = t.parse(
        '{"1": ["exchange_rate", "bogus", "Exchange Rate", "top_up_failed", "card_arrival"],'
        ' "2": "card_arrival"}',
        2,
    )
    # unknown dropped, duplicate collapsed, truncated to k, label == first valid
    assert parsed[1][2] == ["exchange_rate", "top_up_failed", "card_arrival"]
    assert parsed[1][0] == "exchange_rate"
    assert parsed[2] == ("card_arrival", "card_arrival", ["card_arrival"])


def test_label_batch_carries_ranked_list():
    t = Teacher(
        labels=LABELS,
        provider="groq",
        transport=fake({"1": ["top_up_failed", "card_arrival"]}),
        top_k=2,
    )
    res = t.label_batch(["a"])
    assert res[0].label == "top_up_failed" and res[0].ranked == ["top_up_failed", "card_arrival"]
