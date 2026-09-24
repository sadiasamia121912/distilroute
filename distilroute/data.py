"""Loading helpers shared by the scripts: splits, label files, and the gold/teacher join.

`DISTILROUTE_DATASET` picks the dataset every script works on (roadmap 6.4). The default,
Banking77, keeps the original top-level paths; any other dataset gets its own subdirectory of
each (`data/raw/clinc150/`, `results/clinc150/`, ...), so the same scripts run unchanged:

    DISTILROUTE_DATASET=clinc150 python scripts/baseline.py --labels gold

`DISTILROUTE_TEACHER_ROWS` (default 3,000) is how many teacher-labelled training rows a
student may use: the first N of the label file, which is written in one fixed shuffle (seed 0),
so every N is a random sample and each smaller one is inside each larger one. The default keeps
the published numbers reproducible as the file grows (roadmap 1.6); any other N writes its
results, models and docs under their own `teacher<N>/` subdirectory, never over the default's:

    DISTILROUTE_TEACHER_ROWS=10003 python scripts/baseline.py --labels teacher
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pandas is imported where used: the service image does not install it
    import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "banking77"
DATASET = os.environ.get("DISTILROUTE_DATASET") or DEFAULT_DATASET
DEFAULT_TEACHER_ROWS = 3000
TEACHER_ROWS = int(os.environ.get("DISTILROUTE_TEACHER_ROWS") or DEFAULT_TEACHER_ROWS)


def scoped(base: Path) -> Path:
    """`base` for the default dataset, `base/<dataset>` for any other."""
    return base if DATASET == DEFAULT_DATASET else base / DATASET


def outputs(base: Path) -> Path:
    """Where runs write: scoped by dataset, and by teacher-row count when it is not the default.
    Inputs (raw data, label files) are shared by every row count and use `scoped` alone."""
    base = scoped(base)
    return base if TEACHER_ROWS == DEFAULT_TEACHER_ROWS else base / f"teacher{TEACHER_ROWS}"


RAW = scoped(ROOT / "data" / "raw")
LABELS = scoped(ROOT / "data" / "labels")
RESULTS = outputs(ROOT / "results")
MODELS = outputs(ROOT / "models")
DOCS = outputs(ROOT / "docs")
DESCRIPTIONS = scoped(ROOT / "data") / "intent_descriptions.json"


def rel(path: Path) -> str:
    """`path` relative to the repo, for the "-> wrote X" lines the scripts print."""
    return path.relative_to(ROOT).as_posix()


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


def teacher_train_labels(name: str = "train", rows: int | None = None) -> pd.DataFrame:
    """The first `rows` (default `TEACHER_ROWS`) teacher-labelled train rows in label-file order,
    unparsed dropped, with `y` = teacher label.

    File order is the labeller's seed-0 shuffle, so the first N rows are a random sample of the
    split whatever N is. Gold is intentionally not returned: students must never see it.
    """
    rows = TEACHER_ROWS if rows is None else rows
    path = LABELS / f"{name}.jsonl"
    first = [json.loads(line)["idx"] for line in path.open(encoding="utf-8") if line.strip()]
    if len(first) < rows:
        raise ValueError(
            f"{path.name} has {len(first):,} labelled rows, {rows:,} asked "
            "(DISTILROUTE_TEACHER_ROWS); label more first or ask for fewer"
        )
    lab = load_labels(name).loc[sorted(first[:rows])]
    df = load_split("train").join(lab[["teacher", "ranked"]], how="inner")
    return df.dropna(subset=["teacher"]).rename(columns={"teacher": "y"}).drop(columns="category")
