"""CLINC150 → data/raw/clinc_*.csv: the out-of-scope queries the router must refuse (6.2).

    python scripts/download_clinc.py

Banking77 has no "none of these" class, so a router trained on it will always name one of 77
intents, however irrelevant the message. CLINC150 (Larson et al., EMNLP 2019) ships 1,200
deliberately out-of-scope queries for exactly this test, plus 150 in-scope intents over 10
domains, which Phase 6.4 reuses as the second dataset.

Two files:
- `clinc_oos.csv`   — the 1,000 test out-of-scope queries ("how much has the dow changed").
- `clinc_full.csv`  — every in-scope query with its intent (for 6.4).

Source: github.com/clinc/oos-eval (CC BY 3.0). Nothing here is used for training.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distilroute.data import RAW  # noqa: E402

URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"


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

    print(f"out-of-scope: {len(oos):,} queries ({dict(oos.split.value_counts())})")
    print(f"in-scope:     {len(full):,} queries, {full.category.nunique()} intents")
    print(f"-> {RAW / 'clinc_oos.csv'}\n-> {RAW / 'clinc_full.csv'}")
    dup = set(oos.text) & set(pd.read_csv(RAW / "test.csv").text)
    print(f"overlap with the Banking77 test split: {len(dup)}")


if __name__ == "__main__":
    main()
