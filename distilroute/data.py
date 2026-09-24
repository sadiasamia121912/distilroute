"""Loading helpers shared by the scripts: splits, label files, and the gold/teacher join."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pandas is imported where used: the service image does not install it
    import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
LABELS = ROOT / "data" / "labels"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
DOCS = ROOT / "docs"


def load_split(split: str) -> pd.DataFrame:
    """Gold data for a split, indexed by row number; columns `text`, `category`."""
    import pandas as pd

    df = pd.read_csv(RAW / f"{split}.csv")
    df.index.name = "idx"
    return df


def categories() -> list[str]:
    return json.loads((RAW / "categories.json").read_text(encoding="utf-8"))


def load_labels(name: str) -> pd.DataFrame:
    """A label file (`test`, `test.gate_v2`, ...) as a frame indexed by `idx`.

    `teacher` is the hard label (None where unparsed), `ranked` the top-k list.
    """
    import pandas as pd

    path = LABELS / f"{name}.jsonl"
    recs = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    df = pd.DataFrame(recs).set_index("idx").sort_index()
    if "ranked" not in df:
        df["ranked"] = [[x] if x else [] for x in df.teacher]
    return df


def teacher_train_labels(name: str = "train") -> pd.DataFrame:
    """Train rows the teacher has labelled so far (unparsed dropped), with `y` = teacher label.

    Gold is intentionally not returned: students must never see it.
    """
    lab = load_labels(name)
    df = load_split("train").join(lab[["teacher", "ranked"]], how="inner")
    return df.dropna(subset=["teacher"]).rename(columns={"teacher": "y"}).drop(columns="category")
