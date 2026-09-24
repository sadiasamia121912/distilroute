"""CLINC150 → data/raw/: out-of-scope queries (6.2) and a second routing dataset (6.4).

    python scripts/download_clinc.py

Banking77 has no "none of these" class, so a router trained on it will always name one of 77
intents, however irrelevant the message. CLINC150 (Larson et al., EMNLP 2019) ships 1,200
deliberately out-of-scope queries for exactly this test, plus 150 in-scope intents over 10
domains, which Phase 6.4 reuses as the second dataset.

Files:
- `clinc_oos.csv`   — the 1,200 out-of-scope queries ("how much has the dow changed"). Only ever
  scored against, never trained on.
- `clinc_full.csv`  — every in-scope query with its intent and split.
- `clinc150/{train,test}.csv` + `categories.json` — the in-scope data in Banking77's layout, so
  every script runs on it with `DISTILROUTE_DATASET=clinc150`: train 15,000 (100 per intent),
  test 4,500 (30 per intent). The 3,000-row val split is not used; the students hold out their
  own calibration slice, as on Banking77.

Source: github.com/clinc/oos-eval (CC BY 3.0).
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distilroute.data import ROOT  # noqa: E402

URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
# Explicit, not `data.RAW`: that follows DISTILROUTE_DATASET, and this script writes for both.
RAW = ROOT / "data" / "raw"
CLINC = RAW / "clinc150"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    r = requests.get(URL, timeout=120)
    r.raise_for_status()
    data = json.load(io.StringIO(r.text))

    oos = pd.DataFrame(
        [(t, split) for split in ("oos_test", "oos_val", "oos_train") for t, _ in data[split]],
        columns=["text", "split"],
    )
    oos.to_csv(RAW / "clinc_oos.csv", index=False)

    full = pd.DataFrame(
        [(t, y, split) for split in ("train", "val", "test") for t, y in data[split]],
        columns=["text", "category", "split"],
    )
    full.to_csv(RAW / "clinc_full.csv", index=False)

    CLINC.mkdir(exist_ok=True)
    for split in ("train", "test"):
        part = full[full.split == split].drop(columns="split").reset_index(drop=True)
        part.to_csv(CLINC / f"{split}.csv", index=False)
    cats = sorted(full.category.unique())
    (CLINC / "categories.json").write_text(json.dumps(cats, indent=2), encoding="utf-8")

    print(f"out-of-scope: {len(oos):,} queries ({dict(oos.split.value_counts())})")
    print(f"in-scope:     {len(full):,} queries, {full.category.nunique()} intents")
    print(f"-> {RAW / 'clinc_oos.csv'}\n-> {RAW / 'clinc_full.csv'}")
    print(f"-> {CLINC} (train / test / categories, {len(cats)} intents)")
    dup = set(oos.text) & set(pd.read_csv(RAW / "test.csv").text)
    print(f"overlap with the Banking77 test split: {len(dup)}")


if __name__ == "__main__":
    main()
