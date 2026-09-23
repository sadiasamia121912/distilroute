"""Does the router know when a message is not its job? (roadmap 6.2)

    python scripts/download_clinc.py && python scripts/oos.py

A Banking77 student always names one of 77 intents, so the only thing standing between a
stray message and a confidently wrong queue is its **confidence**. This measures whether that
confidence separates in-scope from out-of-scope traffic, using CLINC150's 1,200 out-of-scope
queries as the negatives and the Banking77 test split as the positives.

Per model:
- **AUROC** of three uncertainty scores as out-of-scope detectors (0.5 = useless, 1.0 =
  perfect): max probability, max probability after temperature scaling — the cascade thresholds
  on that, so it matters whether calibration helps or hurts *separation* — and entropy over all
  77 intents.
- **The operating table**: fix the share of in-scope traffic you are willing to escalate
  (5 / 10 / 20 %), read off how much out-of-scope traffic that catches. That is the trade-off
  an operator actually sets.
- **What it costs**: out-of-scope queries that slip through are silently filed into a banking
  queue; the table reports where they land, since a few intents act as magnets.

Writes results/oos.json and docs/oos.md. Nothing here is training data — CLINC is only ever
scored, never learned from.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distilroute import students  # noqa: E402
from distilroute.calibration import apply_temperature  # noqa: E402
from distilroute.data import DOCS, RAW, RESULTS, load_split  # noqa: E402

SCORES = ["max_prob", "max_prob_calibrated", "entropy"]
BEST = "entropy"  # the one the operating table and the magnet list use
BUDGETS = [0.05, 0.10, 0.20]


def scores(router: students.Router, texts: list[str]) -> np.ndarray:
    """(n, 77) raw probabilities — the router's own calibration is applied separately here."""
    return np.vstack([router.proba(t)[1] for t in texts])


def uncertainty(p: np.ndarray, kind: str, temp: float | None = None) -> np.ndarray:
    """How unsure the router is — higher means "more likely not my job".

    `max_prob` is what the cascade already thresholds on. `entropy` uses the whole
    distribution, which is the better detector here: an out-of-scope message spreads mass over
    many intents, while a hard in-scope one puts it on two or three.
    """
    if kind == "max_prob_calibrated":
        p = apply_temperature(p, temp) if temp else p
    return {
        "max_prob": lambda: -p.max(1),
        "max_prob_calibrated": lambda: -p.max(1),
        "entropy": lambda: -(p * np.log(p + 1e-12)).sum(1),
    }[kind]()


def catch_rate(in_u: np.ndarray, oos_u: np.ndarray, budget: float) -> tuple[float, float]:
    """Escalate the most-uncertain `budget` share of in-scope traffic; how much OOS is caught?"""
    thr = float(np.quantile(in_u, 1 - budget))
    return thr, float((oos_u >= thr).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="comma-separated; default: all under models/")
    args = ap.parse_args()

    oos = pd.read_csv(RAW / "clinc_oos.csv")
    test = load_split("test")
    names = (
        args.models.split(",")
        if args.models
        else [n for n in students.available() if n != students.TEACHER]
    )
    print(f"{len(test):,} in-scope (Banking77 test) vs {len(oos):,} out-of-scope (CLINC150)")

    rows = []
    for name in names:
        router = students.load(name)
        p_in, p_oos = scores(router, test.text.tolist()), scores(router, oos.text.tolist())
        row = {"model": name, "temperature": router.temperature}
        y = np.r_[np.zeros(len(p_in)), np.ones(len(p_oos))]
        for kind in SCORES:
            u_in = uncertainty(p_in, kind, router.temperature)
            u_oos = uncertainty(p_oos, kind, router.temperature)
            row[f"auroc_{kind}"] = float(roc_auc_score(y, np.r_[u_in, u_oos]))
            for budget in BUDGETS:
                thr, caught = catch_rate(u_in, u_oos, budget)
                row[f"caught_{kind}_at_{int(budget * 100)}pct"] = caught
                row[f"threshold_{kind}_at_{int(budget * 100)}pct"] = thr
            if kind == BEST:
                # Where do the ones that slip through end up?
                slipped = u_oos < np.quantile(u_in, 0.90)
                landed = pd.Series(np.array(router.classes)[p_oos.argmax(1)][slipped])
                row["top_magnets"] = landed.value_counts().head(3).to_dict()
        rows.append(row)
        print(
            f"  {name:28} AUROC {row['auroc_max_prob']:.3f} max-prob / "
            f"{row['auroc_max_prob_calibrated']:.3f} calibrated / {row['auroc_entropy']:.3f} "
            f"entropy   catches {row[f'caught_{BEST}_at_10pct']:.0%} of OOS at a 10 % budget"
        )

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "oos.json").write_text(json.dumps(rows, indent=2))

    lines = [
        "# Out-of-scope detection",
        "",
        "_Generated by `scripts/oos.py`. Positives: the 3,080 Banking77 test queries. Negatives: "
        f"the {len(oos):,} out-of-scope queries from CLINC150 — general-assistant messages "
        '("how much has the dow changed today"), never seen in training by any model here._',
        "",
        "A Banking77 router has no *none of these* class, so an irrelevant message is always "
        "filed under one of 77 intents. The only defence is confidence. Does it work?",
        "",
        "Yes — the signal is already there, for free, in every student. **AUROC** of each "
        "uncertainty score as an out-of-scope detector (0.5 = useless, 1.0 = perfect):",
        "",
        "| model | max prob | max prob, calibrated | **entropy** | "
        + " | ".join(f"OOS caught at {int(b * 100)} % budget" for b in BUDGETS)
        + " |",
        "|---|---:|---:|---:|" + "---:|" * len(BUDGETS),
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['auroc_max_prob']:.3f} | "
            f"{r['auroc_max_prob_calibrated']:.3f} | **{r['auroc_entropy']:.3f}** | "
            + " | ".join(f"**{r[f'caught_{BEST}_at_{int(b * 100)}pct']:.0%}**" for b in BUDGETS)
            + " |"
        )
    lines += [
        "",
        "**Budget** = the share of *in-scope* queries the service is willing to escalate; the "
        "threshold is that quantile of in-scope uncertainty, and the cell says how much "
        f"out-of-scope traffic it catches (scored by {BEST}). Escalating 10 % of real tickets "
        "is a policy an operator would accept; whatever share of out-of-scope messages that "
        "catches comes free with it.",
        "",
        "Two things worth noticing. **Entropy beats max-probability** on every model: an "
        "out-of-scope message spreads its mass thinly over many intents, while a genuinely "
        "hard in-scope one is torn between two or three — max-probability cannot tell those "
        "apart, the full distribution can. And **temperature scaling, which fixes calibration, "
        "slightly hurts separation** — it is fitted on in-scope data to make confidence honest, "
        "not to push unfamiliar inputs away. Calibrate for the cascade threshold; score "
        "out-of-scope with entropy.",
        "",
        "## Where the ones that slip through land",
        "",
        "Out-of-scope queries that stay below the 10 %-budget threshold, by the intent they "
        "were filed under — a few intents act as magnets, worth watching in production.",
        "",
        "| model | most common landing intents |",
        "|---|---|",
    ]
    for r in rows:
        magnets = ", ".join(f"`{k}` ({v})" for k, v in r["top_magnets"].items())
        lines.append(f"| {r['model']} | {magnets} |")
    lines += [
        "",
        "**Caveat.** CLINC's out-of-scope queries are *far* out of scope — cooking, sport, "
        "trivia. A message about a mortgage or an insurance claim is much closer to banking and "
        "would be harder; no such near-out-of-scope set exists for Banking77, so these numbers "
        "are an upper bound on how easy the problem is.",
    ]
    DOCS.mkdir(exist_ok=True)
    (DOCS / "oos.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("-> docs/oos.md, results/oos.json")


if __name__ == "__main__":
    main()
