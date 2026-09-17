"""Fetch Banking77 (PolyAI) into data/raw/ and run the sanity checks recorded in ROADMAP.md.

    python scripts/download_data.py

Plain CSVs from the authors' GitHub; no `datasets` dependency.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def fetch(name: str) -> Path:
    out = RAW / name
    if not out.exists():
        r = requests.get(f"{BASE}/{name}", timeout=60)
        r.raise_for_status()
        out.write_bytes(r.content)
    return out


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(fetch("train.csv"))
    test = pd.read_csv(fetch("test.csv"))
    cats = json.loads(fetch("categories.json").read_text())

    # The checks that would embarrass us later if they were false.
    assert list(train.columns) == ["text", "category"] == list(test.columns)
    assert set(train.category) == set(test.category) == set(cats), "label sets differ"
    assert train.text.duplicated().sum() == 0, "duplicate train queries"
    assert not (set(train.text) & set(test.text)), "train/test overlap"

    vc = train.category.value_counts()
    print(f"train {len(train):,}  test {len(test):,}  intents {len(cats)}")
    print(f"per-class train count: min {vc.min()}  median {int(vc.median())}  max {vc.max()}")
    print(
        f"query chars: median {int(train.text.str.len().median())}  "
        f"p95 {int(train.text.str.len().quantile(0.95))}  max {train.text.str.len().max()}"
    )
    print(f"written to {RAW}")


if __name__ == "__main__":
    main()
