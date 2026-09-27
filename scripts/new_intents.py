"""Customers start asking about something the router was never taught (roadmap 7.2)

    python scripts/new_intents.py               # 3 draws of 10 held-out intents
    python scripts/new_intents.py --draws 5 --hold 15

6.2 showed the students flag *far* out-of-scope traffic (cooking, sport) almost perfectly. The
harder, more realistic case is a **new banking intent**: a product launch, a new fee, a new
kind of fraud. Those messages look like banking, so the question is whether the router still
notices it has no queue for them. This simulates it on Banking77 itself:

1. **Hold out** `--hold` intents (a random draw; `--draws` draws for spread). Retrain the
   published student — frozen MiniLM + LR head on the 3,000 teacher labels — without every row
   the teacher filed under a held-out intent. Gold is not used to pick those rows: the student
   still never sees a human label here.
2. **Detect**: score the test queries of the held-out intents (the "new" traffic) against the
   test queries of the kept intents, by entropy (the 6.2 winner) and max probability. AUROC,
   and the share of new-intent traffic caught at a fixed in-scope escalation budget. The far-OOS
   numbers of the same student (results/oos.json) sit beside them for contrast.
3. **Add back** each held-out intent with k = 5 / 10 / 25 / 50 / 100 examples. These are gold
   train rows outside the pool, standing in for "someone labels k examples of the new intent" —
   which is how a new intent really arrives; the teacher never saw a list containing it. Report
   the new intents' test accuracy, the kept intents' accuracy (does adding a class hurt the
   others?) and overall accuracy, against the student trained on all 77 from the start.

Held-out intents are drawn from those with >= `--min-rows` gold train rows, so k = 100 is always
available. Uses the cached embeddings (models/cache/minilm_*.npy); no encoder run, a few
minutes on CPU. Writes results/new_intents.json and docs/new_intents.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distilroute.data import (  # noqa: E402
    DOCS,
    RAW,
    RESULTS,
    categories,
    load_split,
    teacher_train_labels,
)
from distilroute.monitor import Z_DEFAULT, shift_z_array  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_curve import minilm_embeddings  # noqa: E402
from oos import BUDGETS, catch_rate  # noqa: E402

KS = [5, 10, 25, 50, 100]
# How unfamiliar a query looks; higher = more likely an intent the router was not taught.
# The first two read the head's probabilities, the last two the embeddings alone.
SCORES = ["max_prob", "entropy", "mahalanobis", "rel_mahalanobis", "knn"]
SCORE_NAMES = {
    "max_prob": "max probability",
    "entropy": "entropy",
    "mahalanobis": "Mahalanobis distance",
    "rel_mahalanobis": "relative Mahalanobis",
    "knn": "k-NN distance (k = 10)",
}
KNN_K = 10
# The traffic monitor (distilroute/monitor.py), simulated: share of traffic the new intent takes,
# window sizes, streams per cell. The reference is 5,000 messages, the service's default.
MONITOR_SHARES = [0.02, 0.05, 0.10]
MONITOR_WINDOWS = [250, 500, 1000, 2000]
MONITOR_REFERENCE = 5000
MONITOR_SIMS = 500


def fit(x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """The published frozen-student head."""
    return LogisticRegression(C=10, max_iter=3000).fit(x, y)


def unit(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def unfamiliarity(
    head: LogisticRegression, x_fit: np.ndarray, y_fit: np.ndarray, x: np.ndarray
) -> dict[str, np.ndarray]:
    """Every score, for the queries `x`, given what the router was trained on.

    - max probability / entropy: the head's own uncertainty (6.2 used these).
    - Mahalanobis: distance to the nearest intent's centre under one shared covariance
      (Ledoit-Wolf shrinkage; 384 dimensions from ~2,600 rows), in the training embeddings.
    - k-NN: cosine distance to the k-th most similar training query.
    The distances look at *where* a query lies, not how the head splits its vote, so a query
    between two known intents and a query off to the side are told apart.
    """
    p = head.predict_proba(x)
    classes = np.unique(y_fit)
    pos = {c: i for i, c in enumerate(classes)}
    centres = np.stack([x_fit[y_fit == c].mean(0) for c in classes])
    precision = LedoitWolf().fit(x_fit - centres[[pos[c] for c in y_fit]]).precision_
    maha = np.min([np.einsum("ij,jk,ik->i", x - m, precision, x - m) for m in centres], axis=0)
    # Relative Mahalanobis (Ren et al., 2021): minus the distance to all training data at once,
    # so what counts is being far from every intent *compared with* being far from banking.
    bg = LedoitWolf().fit(x_fit)
    background = np.einsum("ij,jk,ik->i", x - bg.location_, bg.precision_, x - bg.location_)
    sims = unit(x) @ unit(x_fit).T
    kth = -np.partition(-sims, KNN_K - 1, axis=1)[:, KNN_K - 1]
    return {
        "max_prob": -p.max(1),
        "entropy": -(p * np.log(p + 1e-12)).sum(1),
        "mahalanobis": maha,
        "rel_mahalanobis": maha - background,
        "knn": -kth,
    }


def detect(u: dict[str, np.ndarray], unknown: np.ndarray) -> dict:
    """AUROC of each score, and the share of unknown queries caught at each in-scope budget."""
    out: dict = {}
    for kind in SCORES:
        v = u[kind]
        out[f"auroc_{kind}"] = float(roc_auc_score(unknown, v))
        for b in BUDGETS:
            out[f"caught_{kind}_at_{int(b * 100)}pct"] = catch_rate(v[~unknown], v[unknown], b)[1]
    return out


def landings(head: LogisticRegression, x: np.ndarray, gold: np.ndarray, held: list[str]) -> dict:
    """Per held-out intent: the kept intent absorbing most of its test queries, and its share."""
    pred = head.predict(x)
    out = {}
    for intent in held:
        top = pd.Series(pred[gold == intent]).value_counts(normalize=True)
        out[intent] = {"to": top.index[0], "share": float(top.iloc[0])}
    return out


def monitor_sim(
    head: LogisticRegression, x_test: np.ndarray, gold: np.ndarray, held: list[str], seed: int
) -> dict:
    """How soon does the traffic monitor notice one new intent, and does it name the right place?

    Normal traffic is the router's predictions on the kept intents' test queries; a reference
    of MONITOR_REFERENCE messages and every window are drawn from it, so apart from the new
    intent the traffic is stable. One held-out intent at a time takes `share` of the window.
    Detected = some intent's z passes Z_DEFAULT; named = the top-z intent is the one that
    absorbs most of the new intent's queries.
    """
    rng = np.random.default_rng(seed)
    pos = {c: i for i, c in enumerate(head.classes_)}

    def dist(pred: np.ndarray) -> np.ndarray:
        return np.bincount([pos[c] for c in pred], minlength=len(pos)) / len(pred)

    normal = dist(head.predict(x_test[~np.isin(gold, held)]))
    refs = rng.multinomial(MONITOR_REFERENCE, normal, size=MONITOR_SIMS)
    out = {"false_alarm": {}, "detected": {}, "named": {}}
    for n in MONITOR_WINDOWS:
        z = shift_z_array(refs, rng.multinomial(n, normal, size=MONITOR_SIMS))
        out["false_alarm"][n] = float((z.max(1) > Z_DEFAULT).mean())
    for share in MONITOR_SHARES:
        for n in MONITOR_WINDOWS:
            hits, named = [], []
            for intent in held:
                new = dist(head.predict(x_test[gold == intent]))
                z = shift_z_array(
                    refs, rng.multinomial(n, (1 - share) * normal + share * new, size=MONITOR_SIMS)
                )
                alarm = z.max(1) > Z_DEFAULT
                hits.append(alarm.mean())
                named.append((alarm & (z.argmax(1) == new.argmax())).mean())
            out["detected"][f"{share}_{n}"] = float(np.mean(hits))
            out["named"][f"{share}_{n}"] = float(np.mean(named))
    return out


def accuracies(pred: np.ndarray, gold: np.ndarray, is_new: np.ndarray) -> dict:
    return {
        "acc_new": float((pred[is_new] == gold[is_new]).mean()),
        "acc_kept": float((pred[~is_new] == gold[~is_new]).mean()),
        "acc_all": float((pred == gold).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=int, default=10, help="intents held out per draw")
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--min-rows", type=int, default=120, help="gold train rows an intent needs")
    args = ap.parse_args()

    train, test = load_split("train"), load_split("test")
    pool = teacher_train_labels()
    x_train = minilm_embeddings("train", train.text.tolist())
    x_test = minilm_embeddings("test", test.text.tolist())
    gold = test.category.values
    counts = train.category.value_counts()
    eligible = sorted(counts[counts >= args.min_rows].index)
    print(
        f"pool {len(pool):,} teacher labels; {len(eligible)} of {len(categories())} intents "
        f"have >= {args.min_rows} gold train rows; {args.draws} draws of {args.hold}"
    )

    # Reference: the published student, all 77 intents from the start.
    full = fit(x_train[pool.index], pool.y.values)
    full_pred = full.predict(x_test)
    # The same student against *far* out-of-scope traffic (CLINC150, as in 6.2), every score.
    oos = pd.read_csv(RAW / "clinc_oos.csv")
    x_oos = minilm_embeddings("clinc_oos", oos.text.tolist())
    u_in = unfamiliarity(full, x_train[pool.index], pool.y.values, x_test)
    u_out = unfamiliarity(full, x_train[pool.index], pool.y.values, x_oos)
    far = detect(
        {k: np.r_[u_in[k], u_out[k]] for k in SCORES},
        np.r_[np.zeros(len(x_test), bool), np.ones(len(x_oos), bool)],
    )

    draws = []
    for d in range(args.draws):
        rng = np.random.default_rng(d)
        held = sorted(rng.choice(eligible, size=args.hold, replace=False).tolist())
        is_new = np.isin(gold, held)
        kept = pool[~pool.y.isin(held)]
        head = fit(x_train[kept.index], kept.y.values)
        row = {
            "draw": d,
            "held_out": held,
            "pool_rows": int(len(kept)),
            "new_test_queries": int(is_new.sum()),
            "reference": accuracies(full_pred, gold, is_new),
            **detect(unfamiliarity(head, x_train[kept.index], kept.y.values, x_test), is_new),
            "landings": landings(head, x_test, gold, held),
            "monitor": monitor_sim(head, x_test, gold, held, seed=d),
            "add_back": [],
        }
        print(
            f"  draw {d}: held {held}\n"
            f"    {len(kept):,} rows kept; AUROC "
            + ", ".join(f"{k} {row[f'auroc_{k}']:.3f}" for k in SCORES)
            + "; caught at a 10 % budget: "
            + ", ".join(f"{k} {row[f'caught_{k}_at_10pct']:.0%}" for k in SCORES)
        )

        # Add each held-out intent back with k human-labelled examples (gold rows outside the pool).
        spare = train[~train.index.isin(kept.index) & train.category.isin(held)]
        order = spare.sample(frac=1, random_state=d)  # one fixed order, so k=50 contains k=25
        for k in [0, *KS]:
            add = order.groupby("category", sort=False).head(k)
            idx = np.r_[kept.index.values, add.index.values]
            y = np.r_[kept.y.values, add.category.values]
            pred = fit(x_train[idx], y).predict(x_test)
            step = {"k": k, "added": int(len(add)), **accuracies(pred, gold, is_new)}
            row["add_back"].append(step)
            print(
                f"    k={k:>3}  new {step['acc_new']:.3f}  kept {step['acc_kept']:.3f}  "
                f"all {step['acc_all']:.3f}   (all-77 student: new "
                f"{row['reference']['acc_new']:.3f}, all {row['reference']['acc_all']:.3f})",
                flush=True,
            )
        draws.append(row)

    print("  far out-of-scope: " + ", ".join(f"{k} {far[f'auroc_{k}']:.3f}" for k in SCORES))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "new_intents.json").write_text(
        json.dumps(
            {"hold": args.hold, "min_rows": args.min_rows, "far": far, "draws": draws}, indent=2
        )
    )
    write_doc(draws, args, far)
    print("-> docs/new_intents.md, results/new_intents.json")


def monitor_doc(draws: list[dict]) -> list[str]:
    def cell(key: str, part: str) -> float:
        return float(np.mean([d["monitor"][part][key] for d in draws]))

    lines = [
        "",
        "## The fix: watch the mix of intents, not each message",
        "",
        "`distilroute/monitor.py`, served at `GET /monitor`: the first "
        f"{MONITOR_REFERENCE:,} routed messages set each intent's expected share; after that "
        "the latest window is compared with it intent by intent, and an intent whose share "
        f"jumps (z > {Z_DEFAULT}, the one-sided Bonferroni threshold for 1 % over 77 intents; "
        "the measured false-alarm rate is the last row) is flagged by name. The service's "
        "default window is 2,000 messages. Simulated here with every held-out intent in turn "
        "taking a share of the "
        f"traffic ({MONITOR_SIMS} streams per cell, all draws): the share of streams flagged, "
        "and in brackets the share where the top flag is the intent the newcomer lands in.",
        "",
        "| new intent's share of traffic | "
        + " | ".join(f"window {n:,}" for n in MONITOR_WINDOWS)
        + " |",
        "|---|" + "---:|" * len(MONITOR_WINDOWS),
    ]
    for share in MONITOR_SHARES:
        lines.append(
            f"| {share:.0%} | "
            + " | ".join(
                f"**{cell(f'{share}_{n}', 'detected'):.0%}** ({cell(f'{share}_{n}', 'named'):.0%})"
                for n in MONITOR_WINDOWS
            )
            + " |"
        )
    lines.append(
        "| 0 % (false alarms) | "
        + " | ".join(f"{cell(n, 'false_alarm'):.1%}" for n in MONITOR_WINDOWS)
        + " |"
    )
    first = {
        share: next((n for n in MONITOR_WINDOWS if cell(f"{share}_{n}", "detected") >= 0.9), None)
        for share in MONITOR_SHARES
    }
    said = [
        f"at {share:.0%} of traffic within a {n:,}-message window"
        for share, n in first.items()
        if n
    ]
    lines += [
        "",
        "A new intent is flagged in 90 % of streams "
        + ("; ".join(said) if said else "in none of the windows tried")
        + ". Per message the router cannot tell (above); in aggregate it can, and it points at "
        "where the new queries are going. **Assumes** otherwise stable traffic: real traffic "
        "drifts by weekday and season, so in production the reference should be refreshed and "
        "the false-alarm rate checked on the service's own history.",
    ]
    return lines


def mean_sd(vals: list[float], pct: bool = False) -> str:
    m, s = float(np.mean(vals)), float(np.std(vals))
    return f"{m:.0%} ± {s:.0%}" if pct else f"{m:.3f} ± {s:.3f}"


def write_doc(draws: list[dict], args: argparse.Namespace, far: dict) -> None:
    n = len(draws)
    lines = [
        "# New intents: does the router notice, and how many labels does one need?",
        "",
        f"_Generated by `scripts/new_intents.py`. {n} random draws of {args.hold} Banking77 "
        "intents held out of the published student (frozen MiniLM + LR on 3,000 teacher labels); "
        "mean ± sd over draws._",
        "",
        "[Out-of-scope detection](oos.md) covered *far* out-of-scope traffic. Here the unknown "
        "messages are banking messages: the student is retrained without some intents, then "
        "shown their test queries, as if a new product had launched.",
        "",
        "## Does it notice?",
        "",
        "Four ways to score how unfamiliar a query is. The first two read the head's "
        "probabilities; the last two measure where the query's embedding lies relative to the "
        "training queries. Near = the held-out banking intents; far = CLINC150's 1,200 "
        "out-of-scope queries against the same student trained on all 77 intents.",
        "",
        "| score | near: AUROC | "
        + " | ".join(f"near: caught at {int(b * 100)} %" for b in BUDGETS)
        + " | far: AUROC | far: caught at 10 % |",
        "|---|---:|" + "---:|" * len(BUDGETS) + "---:|---:|",
    ]
    for kind in SCORES:
        lines.append(
            f"| {SCORE_NAMES[kind]} | {mean_sd([d[f'auroc_{kind}'] for d in draws])} | "
            + " | ".join(
                mean_sd([d[f"caught_{kind}_at_{int(b * 100)}pct"] for d in draws], pct=True)
                for b in BUDGETS
            )
            + f" | {far[f'auroc_{kind}']:.3f} | {far[f'caught_{kind}_at_10pct']:.0%} |"
        )
    lines += [
        "",
        "**Budget** = the share of in-scope (kept-intent) test queries escalated; the cell is the "
        "share of new-intent queries that threshold catches.",
        "",
        "Where the new intents' queries end up (draw 0) — the kept intent that absorbs most "
        "of each:",
        "",
        "| held-out intent | filed under | share |",
        "|---|---|---:|",
    ]
    for intent, land in draws[0]["landings"].items():
        lines.append(f"| `{intent}` | `{land['to']}` | {land['share']:.0%} |")

    ref_new = [d["reference"]["acc_new"] for d in draws]
    ref_all = [d["reference"]["acc_all"] for d in draws]
    near = {k: np.mean([d[f"auroc_{k}"] for d in draws]) for k in SCORES}
    near10 = {k: np.mean([d[f"caught_{k}_at_10pct"] for d in draws]) for k in SCORES}
    dist = ["mahalanobis", "knn"]
    rel = "rel_mahalanobis"
    absorbed = np.mean([v["share"] for d in draws for v in d["landings"].values()])
    new_by_k = {
        k: np.mean([d["add_back"][i]["acc_new"] for d in draws]) for i, k in enumerate([0, *KS])
    }
    parity = next((k for k, a in new_by_k.items() if a >= np.mean(ref_new)), f"> {KS[-1]}")
    kept_drop = np.mean(
        [d["add_back"][-1]["acc_kept"] - d["add_back"][0]["acc_kept"] for d in draws]
    )
    lines += [
        "",
        f"**Much harder than far out-of-scope.** With entropy, AUROC drops from "
        f"{far['auroc_entropy']:.3f} (far) to {near['entropy']:.3f} (near), and a 10 % budget "
        f"catches {near10['entropy']:.0%} of the new-intent traffic instead of "
        f"{far['caught_entropy_at_10pct']:.0%}. The distance scores, the usual remedy, split the "
        "two cases the other way: best on far traffic ("
        + ", ".join(f"{SCORE_NAMES[k]} {far[f'auroc_{k}']:.3f}" for k in dist)
        + ") and worst on near ("
        + ", ".join(f"{near[k]:.3f}" for k in dist)
        + "). A new banking intent lies among the known ones in a general-purpose embedding "
        "space, so being far from the training queries is exactly what it is not; and the head "
        "splits its vote on it the way it does on a hard in-scope query, so probabilities "
        "cannot tell either. The variant built for this case, relative Mahalanobis, moves near "
        f"AUROC by {(near[rel] - near['entropy']) * 100:+.1f} pt over entropy (within the spread "
        f"between draws) and far by {(far[f'auroc_{rel}'] - far['auroc_entropy']) * 100:+.1f} "
        "pt: no per-message score fixes this. What does give it away: the neighbours absorb "
        f"it — on average {absorbed:.0%} of a new intent's queries land in a single kept intent.",
    ]
    lines += monitor_doc(draws)
    lines += [
        "",
        "## How many labels does a new intent need?",
        "",
        "Each held-out intent added back with k examples (gold train rows, standing in for a "
        "person "
        "labelling k messages of the new intent); the rest of the pool is unchanged.",
        "",
        "| k per new intent | new intents' accuracy | kept intents' accuracy | overall accuracy |",
        "|---:|---:|---:|---:|",
    ]
    for i, k in enumerate([0, *KS]):
        steps = [d["add_back"][i] for d in draws]
        lines.append(
            f"| {k} | {mean_sd([s['acc_new'] for s in steps])} | "
            f"{mean_sd([s['acc_kept'] for s in steps])} | "
            f"{mean_sd([s['acc_all'] for s in steps])} |"
        )
    lines += [
        f"| all 77 from the start (~39 teacher labels each) | {mean_sd(ref_new)} | — | "
        f"{mean_sd(ref_all)} |",
        "",
        f"**The all-77 student's {np.mean(ref_new):.3f} on these intents is matched at k = "
        f"{parity} labelled examples.** Five already route "
        f"{np.mean([d['add_back'][1]['acc_new'] for d in draws]):.0%} of the new traffic "
        "correctly. The price is paid by the neighbours: kept-intent accuracy moves by "
        f"{kept_drop * 100:+.1f} pt from k = 0 to k = {KS[-1]}, where each new intent has more "
        "rows than a typical kept one.",
        "",
        "The all-77 row is the published student's configuration fitted on the whole pool (the "
        "published run holds 10 % back for calibration, hence 0.846 here vs 0.848 there); every "
        "row in this table is fitted the same way.",
        "",
        "**Caveats.** Rows are held out by the *teacher's* label, so the few held-out-intent "
        "messages the teacher had filed elsewhere stay in the pool (as they would: a teacher "
        "without the intent in its list files them somewhere). The added examples are human "
        "labels, cleaner than the teacher's ~85 %, which flatters k slightly against the "
        "all-77 row. Intents with fewer than "
        f"{args.min_rows} gold train rows are never held out.",
    ]
    DOCS.mkdir(exist_ok=True)
    (DOCS / "new_intents.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
