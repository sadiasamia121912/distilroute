"""The dataset and teacher-row switches: defaults keep the top-level paths, others get their own."""

from __future__ import annotations

import importlib
import json

import pandas as pd
import pytest

from distilroute import data


@pytest.fixture
def reload_data(monkeypatch):
    def load(dataset: str | None, rows: int | None = None):
        for var, value in (("DISTILROUTE_DATASET", dataset), ("DISTILROUTE_TEACHER_ROWS", rows)):
            if value is None:
                monkeypatch.delenv(var, raising=False)
            else:
                monkeypatch.setenv(var, str(value))
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


def test_default_teacher_rows_keep_published_paths(reload_data):
    d = reload_data(None)
    assert d.TEACHER_ROWS == 3000
    assert d.RESULTS == d.ROOT / "results" and d.MODELS == d.ROOT / "models"


def test_other_teacher_rows_write_apart_but_read_the_same_labels(reload_data):
    d = reload_data(None, rows=10003)
    assert d.RESULTS == d.ROOT / "results" / "teacher10003"
    assert d.MODELS == d.ROOT / "models" / "teacher10003"
    assert d.DOCS == d.ROOT / "docs" / "teacher10003"
    assert d.LABELS == d.ROOT / "data" / "labels" and d.RAW == d.ROOT / "data" / "raw"
    d = reload_data("clinc150", rows=500)
    assert d.RESULTS == d.ROOT / "results" / "clinc150" / "teacher500"
    assert d.LABELS == d.ROOT / "data" / "labels" / "clinc150"


def test_teacher_train_labels_takes_the_first_rows_in_file_order(
    reload_data, tmp_path, monkeypatch
):
    d = reload_data(None)
    (tmp_path / "raw").mkdir()
    (tmp_path / "labels").mkdir()
    pd.DataFrame({"text": [f"q{i}" for i in range(6)], "category": list("abcabc")}).to_csv(
        tmp_path / "raw" / "train.csv", index=False
    )
    order = [4, 1, 5, 0]  # the labeller's shuffle; row 5 did not parse
    with (tmp_path / "labels" / "train.jsonl").open("w", encoding="utf-8") as f:
        for i in order:
            rec = {"idx": i, "teacher": None if i == 5 else "x", "ranked": []}
            f.write(json.dumps(rec) + "\n")
    monkeypatch.setattr(d, "RAW", tmp_path / "raw")
    monkeypatch.setattr(d, "LABELS", tmp_path / "labels")

    assert list(d.teacher_train_labels(rows=2).index) == [1, 4]
    assert list(d.teacher_train_labels(rows=3).index) == [1, 4]  # 5 unparsed, dropped
    assert list(d.teacher_train_labels(rows=4).index) == [0, 1, 4]
    assert "category" not in d.teacher_train_labels(rows=4)  # gold never reaches a student
    with pytest.raises(ValueError, match="4 labelled rows, 5 asked"):
        d.teacher_train_labels(rows=5)
