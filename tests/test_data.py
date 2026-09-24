"""The dataset switch: Banking77 keeps the top-level paths, others get a subdirectory."""

from __future__ import annotations

import importlib

import pytest

from distilroute import data


@pytest.fixture
def reload_data(monkeypatch):
    def load(dataset: str | None):
        if dataset is None:
            monkeypatch.delenv("DISTILROUTE_DATASET", raising=False)
        else:
            monkeypatch.setenv("DISTILROUTE_DATASET", dataset)
        return importlib.reload(data)

    yield load
    monkeypatch.undo()
    importlib.reload(data)


def test_default_dataset_keeps_top_level_paths(reload_data):
    d = reload_data(None)
    assert d.DATASET == "banking77"
    assert d.RAW == d.ROOT / "data" / "raw"
    assert d.RESULTS == d.ROOT / "results"
    assert d.DESCRIPTIONS == d.ROOT / "data" / "intent_descriptions.json"


def test_other_dataset_is_scoped_everywhere(reload_data):
    d = reload_data("clinc150")
    for path, base in [
        (d.RAW, d.ROOT / "data" / "raw"),
        (d.LABELS, d.ROOT / "data" / "labels"),
        (d.RESULTS, d.ROOT / "results"),
        (d.MODELS, d.ROOT / "models"),
        (d.DOCS, d.ROOT / "docs"),
    ]:
        assert path == base / "clinc150"
    assert d.DESCRIPTIONS == d.ROOT / "data" / "clinc150" / "intent_descriptions.json"
    assert d.rel(d.RESULTS) == "results/clinc150"
