"""Cost per 1M requests for the teacher and every benchmarked student (roadmap 3.3).

    python scripts/cost.py

Nothing is bought: this multiplies measured usage by published list prices.

Teacher: tokens per query x the provider's *paid* per-token price. The free tier we labelled
with is rate-capped with no SLA, so it is not a production option and is not what the table
quotes. Two rows, because the way we labelled (20 queries per call, the 77 descriptions
amortised) is not how a routing service runs (one ticket per call, the full prompt every time).

Students: CPU-seconds per query (mean in-process latency from results/latency.json) x the price
of a CPU-second on a small cloud VM, assuming one request at a time on one core. That is the
naive upper bound — a 2-vCPU box serving two at once halves it — so the row for "the laptop
you already own" ($0) is also there, because that is the honest answer for this volume.

Writes results/cost.json; `scripts/evaluate.py` renders it into docs/results.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distilroute.data import RESULTS  # noqa: E402

# Published list prices, checked 2026-09-18. Update the date if you update the numbers.
PRICES = {
    "teacher": {
        "model": "openai/gpt-oss-120b on Groq (paid tier)",
        "usd_per_1m_in": 0.15,
        "usd_per_1m_out": 0.60,
        "source": "https://www.cloudzero.com/blog/groq-pricing/",
    },
    "vm": {
        "name": "AWS t3.small (2 vCPU, us-east-1, on-demand)",
        "usd_per_hour": 0.0208,
        "source": "https://instances.vantage.sh/aws/ec2/t3.small",
    },
}

# Measured on the full test-split run (final config: v2 descriptions, top-3, reasoning low),
# 2026-09-19: 251,996 prompt + 66,064 completion tokens over 3,080 queries in 135 calls of
# 20-ish -> 1,867 prompt tokens per call = ~1,330 fixed (77 names + descriptions +
# instructions) + ~27 per query, and 21 completion tokens (answer + reasoning) per query.
# Single-query mode pays the fixed prompt on every call.
TOKENS = {"prompt_fixed": 1330, "prompt_per_query": 27, "output_per_query": 21}


def teacher_cost(batch: int) -> dict:
    p = PRICES["teacher"]
    tok_in = (TOKENS["prompt_fixed"] + batch * TOKENS["prompt_per_query"]) / batch
    tok_out = TOKENS["output_per_query"]
    usd = tok_in * p["usd_per_1m_in"] + tok_out * p["usd_per_1m_out"]  # per 1M queries
    return {
        "model": p["model"],
        "mode": f"{batch} queries per call" if batch > 1 else "one query per call",
        "tokens_in_per_query": round(tok_in, 1),
        "tokens_out_per_query": tok_out,
        "usd_per_1m": round(usd, 2),
    }


def main() -> None:
    latency = json.loads((RESULTS / "latency.json").read_text())
    per_sec = PRICES["vm"]["usd_per_hour"] / 3600
    rows = {"teacher_batched": teacher_cost(20), "teacher_single": teacher_cost(1)}
    for name, lat in latency.items():
        if "@" in name or name == "teacher":
            continue
        cpu_s = lat["mean_ms"] / 1000
        rows[name] = {
            "model": name,
            "mode": f"{lat['mean_ms']:.1f} ms/query on {PRICES['vm']['name']}",
            "cpu_seconds_per_query": cpu_s,
            "usd_per_1m": round(cpu_s * per_sec * 1e6, 2),
            "usd_per_1m_own_hardware": 0.0,
        }
    out = {"prices": PRICES, "tokens": TOKENS, "rows": rows}
    (RESULTS / "cost.json").write_text(json.dumps(out, indent=2))
    for r in rows.values():
        print(f"{r['model']:50} {r['mode']:42} ${r['usd_per_1m']:>10,.2f} / 1M")
    print("-> results/cost.json")


if __name__ == "__main__":
    main()
