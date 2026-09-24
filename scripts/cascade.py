"""Multi-tier cascade: cheap student → better student → LLM, a threshold per tier (roadmap 6.5).

    python scripts/cascade.py
    DISTILROUTE_DATASET=clinc150 python scripts/cascade.py

Each query goes to the first tier; if that tier's *calibrated* confidence clears its
threshold it answers, otherwise the query moves down a tier, and whatever no student
answers goes to the LLM (the teacher labels on the test split stand in for its answer, so
nothing is called). A query pays for every tier it reaches: CPU-seconds on the priced VM for
a student (`results/latency.json`), the paid per-query price for the LLM (`results/cost.json`,
one query per call, since a live cascade cannot wait to fill a batch).

The question it answers: for a given budget, which system is most accurate — and does a
1.7 ms TF-IDF first tier save anything in front of a 2.8 ms MiniLM?

Thresholds are **not** tuned on the rows they are scored on. The test split is halved at
random; thresholds are chosen on one half and scored on the other, then the halves swap, over
several halvings, and every number is the mean over the held-out halves. Choosing and scoring
on the same rows would flatter every multi-threshold system, and the three-tier ones most.

Writes results/cascade.json and docs/cascade.md.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distilroute.calibration import apply_temperature  # noqa: E402
from distilroute.data import DATASET, DOCS, RESULTS, load_labels, load_split, rel  # noqa: E402

# Systems to compare: the student tiers in order; every system ends in the LLM.
SYSTEMS = [
    [],
    ["tfidf_lr_teacher"],
    ["tinybert_ft_teacher"],
    ["minilm_ft_teacher"],
    ["minilm_frozen_teacher"],
    ["tfidf_lr_teacher", "minilm_ft_teacher"],
    ["tinybert_ft_teacher", "minilm_ft_teacher"],
    ["tfidf_lr_teacher", "minilm_frozen_teacher"],
]
# Candidate thresholds per tier; 0.0 = the tier answers everything that reaches it.
GRID = [0.0] + np.round(np.arange(0.30, 0.99, 0.02), 2).tolist()
BUDGETS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]  # max share of queries sent to the LLM
KEYS = ("acc", "llm", "usd", "ms", "first")  # what is reported per held-out half


def load_tier(name: str, classes: list[str]) -> dict:
    """Calibrated predictions and confidence of one student on the test split."""
    meta = json.loads((RESULTS / f"{name}.json").read_text())
    z = np.load(RESULTS / f"{name}_test_probs.npz", allow_pickle=True)
    proba, own = z["proba"], list(z["classes"])
    if meta.get("temperature"):
        proba = apply_temperature(proba, meta["temperature"])
    assert set(own) <= set(classes), f"{name}: unknown classes"
    return {"pred": np.array(own)[proba.argmax(axis=1)], "conf": proba.max(axis=1)}


def run(tiers: list[dict], thr: tuple[float, ...], llm: np.ndarray, rows: np.ndarray) -> dict:
    """Route `rows` through the tiers at thresholds `thr`: predictions + who answered."""
    pred = llm[rows].copy()
    open_ = np.ones(len(rows), dtype=bool)  # not answered yet
    reached = []
    for tier, t in zip(tiers, thr, strict=True):
        reached.append(float(open_.mean()))
        take = open_ & (tier["conf"][rows] >= t)
        pred[take] = tier["pred"][rows][take]
        open_ &= ~take
    return {"pred": pred, "reached": reached, "llm": float(open_.mean())}


def summarise(held: list[dict]) -> dict:
    """Mean and half-range over the held-out halves, per reported key."""
    out = {}
    for k in KEYS:
        v = [h[k] for h in held]
        out[k] = float(np.mean(v))
        out[f"{k}_pm"] = float((max(v) - min(v)) / 2)
    return out | {"folds": held}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=5, help="random halvings of the test split")
    args = ap.parse_args()

    test = load_split("test")
    gold = test.category.values
    labels = load_labels("test").teacher.reindex(test.index)
    rows_all = np.flatnonzero(labels.notna().values)  # the LLM's answer must be known
    llm = labels.fillna("").values
    classes = sorted(set(gold))
    teacher_acc = float((llm[rows_all] == gold[rows_all]).mean())

    cost = json.loads((RESULTS / "cost.json").read_text())
    llm_cost = cost["rows"]["teacher_single"]["usd_per_1m"] / 1e6
    vm_per_s = cost["prices"]["vm"]["usd_per_hour"] / 3600
    latency = json.loads((RESULTS / "latency.json").read_text())

    # Each seed halves the test rows; each half is used once to pick and once to score.
    halves, folds = [], []
    for seed in range(args.seeds):
        perm = np.random.default_rng(seed).permutation(rows_all)
        halves += [np.sort(perm[: len(perm) // 2]), np.sort(perm[len(perm) // 2 :])]
        a, b = len(halves) - 2, len(halves) - 1
        folds += [(a, b), (b, a)]

    systems = []
    for names in SYSTEMS:
        missing = [n for n in names if not (RESULTS / f"{n}.json").exists()]
        if missing:
            print(f"skip {' -> '.join(names)}: no run for {missing}")
            continue
        tiers = [load_tier(n, classes) for n in names]
        ms = [latency[n]["mean_ms"] for n in names]
        tier_cost = [m / 1000 * vm_per_s for m in ms]
        grid = list(itertools.product(GRID, repeat=len(names)))

        def score(thr, rows, tiers=tiers, ms=ms, tier_cost=tier_cost) -> dict:
            res = run(tiers, thr, llm, rows)
            reached = res["reached"]
            first = 1 - (reached[1] if len(reached) > 1 else res["llm"]) if reached else 0.0
            return {
                "acc": float((res["pred"] == gold[rows]).mean()),
                "llm": res["llm"],
                "usd": 1e6 * (np.dot(reached, tier_cost) + res["llm"] * llm_cost),
                "ms": float(np.dot(reached, ms)),
                "first": float(first),
            }

        # Every threshold combination scored once on every half.
        evals = [[score(t, h) for t in grid] for h in halves]

        def held_out(choose, evals=evals, grid=grid) -> dict | None:
            """Pick a grid point with `choose` on one half, score it on the other; all folds."""
            held = []
            for pick, score_on in folds:
                i = choose(evals[pick], pick)
                if i is None:
                    return None
                held.append(evals[score_on][i] | {"thr": list(grid[i])})
            return summarise(held)

        def most_accurate_within(budget):
            def choose(ev, _pick):
                ok = [(e["acc"], -e["usd"], i) for i, e in enumerate(ev) if e["llm"] <= budget]
                return max(ok)[2] if ok else None

            return choose

        def cheapest_matching(ev, pick):
            target = float((llm[halves[pick]] == gold[halves[pick]]).mean())
            ok = [(e["usd"], i) for i, e in enumerate(ev) if e["acc"] >= target]
            return min(ok)[1] if ok else None

        budgets = {str(b): held_out(most_accurate_within(b)) for b in BUDGETS}
        match = held_out(cheapest_matching) if names else None
        systems.append(
            {
                "tiers": names,
                "mean_ms": ms,
                "budgets": {k: v for k, v in budgets.items() if v},
                "match_llm": match,
            }
        )
        print(
            f"{' -> '.join(names + ['LLM']):50} "
            + (
                f"matches the LLM at ${match['usd']:.2f} +/- {match['usd_pm']:.2f} per 1M, "
                f"{match['llm']:.1%} to the LLM, {match['ms']:.2f} ms"
                if match
                else ""
            )
        )

    out = {
        "dataset": DATASET,
        "n_rows": len(rows_all),
        "teacher_acc": teacher_acc,
        "llm_usd_per_1m": llm_cost * 1e6,
        "seeds": args.seeds,
        "systems": systems,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "cascade.json").write_text(json.dumps(out, indent=2))
    write_doc(out)
    print(f"-> {rel(RESULTS)}/cascade.json, {rel(DOCS)}/cascade.md")


def _name(tiers: list[str]) -> str:
    return " → ".join([t.replace("_teacher", "") for t in tiers] + ["LLM"])


def paired_lines(systems: list[dict]) -> list[str]:
    """Each multi-tier system against the same chain without its first tier, fold by fold.

    Both are scored on identical halves, so the per-fold difference cancels most of the
    halving noise that dominates the ± in the table above.
    """
    by_tiers = {tuple(s["tiers"]): s for s in systems}
    rows = []
    for s in systems:
        if len(s["tiers"]) < 2 or s["match_llm"] is None:
            continue
        base = by_tiers.get(tuple(s["tiers"][1:]))
        if base is None or base["match_llm"] is None:
            continue
        pairs = zip(s["match_llm"]["folds"], base["match_llm"]["folds"], strict=True)
        d = np.array(
            [(a["usd"] - b["usd"], a["llm"] - b["llm"], a["ms"] - b["ms"]) for a, b in pairs]
        )
        rows.append(
            f"| {s['tiers'][0].replace('_teacher', '')} in front of {_name(base['tiers'])} | "
            f"{d[:, 0].mean():+.2f} | {int((d[:, 0] < 0).sum())} / {len(d)} | "
            f"{d[:, 1].mean():+.1%} | {d[:, 2].mean():+.2f} |"
        )
    if not rows:
        return []
    return [
        "",
        "### Does a cheaper first tier pay?",
        "",
        "Adding a first tier to a chain, compared on the same halves (matching the LLM, as "
        "above): change in $ per 1M, in how many halves it came out cheaper, change in the share "
        "sent to the LLM, change in mean student ms.",
        "",
        "| change | Δ $ per 1M | cheaper in | Δ sent to the LLM | Δ student ms |",
        "|---|---:|---:|---:|---:|",
        *rows,
    ]


def write_doc(out: dict) -> None:
    lines = [
        "# Multi-tier cascade",
        "",
        f"_Generated by `scripts/cascade.py`. {out['n_rows']:,} test queries with a teacher "
        f"label; the LLM alone scores **{out['teacher_acc']:.3f}** at "
        f"**${out['llm_usd_per_1m']:.2f}** per 1M requests (paid list price, one query per "
        "call). Students answer when their calibrated confidence clears their tier's threshold; "
        "the rest go down a tier and finally to the LLM. Every student is trained on teacher "
        "labels only. Thresholds are chosen on one half of the test split and scored on the "
        f"other, both ways round, over {out['seeds']} random halvings: every number is a mean "
        f"over {2 * out['seeds']} held-out halves, ± half the range._",
        "",
        "## Cheapest system that matches the LLM alone",
        "",
        "On the picking half: the cheapest thresholds whose accuracy reaches the LLM's own. "
        "Student ms is the mean CPU time a query spends in the student tiers; the LLM's own "
        "latency comes on top for the share sent to it.",
        "",
        "| system | held-out accuracy | sent to the LLM | answered by tier 1 | student ms "
        "| $ per 1M |",
        "|---|---:|---:|---:|---:|---:|",
        f"| LLM only | {out['teacher_acc']:.3f} | 100 % | — | — | {out['llm_usd_per_1m']:.2f} |",
    ]
    for s in out["systems"]:
        m = s["match_llm"]
        if m is None:
            continue
        lines.append(
            f"| {_name(s['tiers'])} | {m['acc']:.3f} ± {m['acc_pm']:.3f} | "
            f"{m['llm']:.1%} ± {m['llm_pm']:.1%} | {m['first']:.0%} | {m['ms']:.2f} | "
            f"**{m['usd']:.2f}** ± {m['usd_pm']:.2f} |"
        )
    lines += paired_lines(out["systems"])
    lines += [
        "",
        "## Best accuracy for a budget of LLM calls",
        "",
        "At most this share of queries may reach the LLM (on the picking half) → held-out "
        "accuracy and $ per 1M requests. ≤ 0 % is the student chain on its own.",
        "",
        "| system | " + " | ".join(f"≤ {b:.0%}" for b in BUDGETS) + " |",
        "|---|" + "---:|" * len(BUDGETS),
    ]
    for s in out["systems"]:
        if not s["tiers"]:
            continue
        cells = []
        for b in BUDGETS:
            r = s["budgets"].get(str(b))
            cells.append(f"{r['acc']:.3f} (${r['usd']:.0f})" if r else "—")
        lines.append(f"| {_name(s['tiers'])} | " + " | ".join(cells) + " |")
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "cascade.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
