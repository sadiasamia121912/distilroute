"""Does the cascade teach the student for free? A router that learns from its escalations.

    python scripts/self_improving.py                  # Banking77, 3 seeds, ~10 min on a laptop
    DISTILROUTE_DATASET=clinc150 python scripts/self_improving.py

Every message the cascade escalates comes back with the LLM's answer. That call was paid to serve
the customer, so its answer is a training label at no extra cost. This replays the training split
as live traffic and asks whether retraining on those answers makes the student better and the LLM
bill smaller over time.

- **Start:** the frozen MiniLM student (the project's CPU student: cached encoder, logistic head)
  trained on the first `BOOTSTRAP` messages of a seeded shuffle, labelled by the LLM: a cold start
  that has to be paid for.
- **Weeks:** the next messages arrive `WEEK` at a time. The student in force routes each one and
  escalates it when its calibrated confidence is below `THRESHOLD` (0.8, the service's policy).
- **Arms**, each retrained at the end of every week:
  - *static* — never retrained;
  - *learn from escalations* — adds that week's escalated messages with the LLM's answer;
  - *random, same count* — adds as many messages as the arm above, picked at random from the
    traffic seen so far (a control: same labels count, not chosen by the cascade, and those
    would cost an extra LLM call each);
  - *label everything* — adds every message: the ceiling, at the price of the LLM alone.
- **Scored** on the untouched test split against human labels: the student alone, and the cascade
  at the same threshold (the teacher's test labels stand in for the LLM's answer, as in
  scripts/cascade.py). The escalation rate is measured on each week's live traffic.

Temperature is refitted at every retrain on a 10 % hold-out of that arm's labelled messages,
never on test. The LLM's labels are the committed teacher labels (data/labels/), so nothing is
called. Writes results/self_improving.json and docs/self_improving.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distilroute.calibration import apply_temperature, fit_temperature  # noqa: E402
from distilroute.data import (  # noqa: E402
    DATASET,
    DOCS,
    LABELS,
    RESULTS,
    categories,
    load_labels,
    load_split,
    rel,
    teacher_train_labels,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_curve import minilm_embeddings  # noqa: E402

BOOTSTRAP = 500
WEEK = 1000
THRESHOLD = 0.8
SEEDS = (0, 1, 2)
ARMS = ("static", "escalations", "random", "everything")
NAMES = {
    "static": "static (never retrained)",
    "escalations": "learn from escalations",
    "random": "random, same count",
    "everything": "label everything",
}


class Student:
    """Frozen-encoder head with its own temperature, fitted on a hold-out of its labels."""

    def __init__(self, x: np.ndarray, y: np.ndarray, n_classes: int, seed: int):
        cal = np.random.default_rng(seed).random(len(y)) < 0.1
        self.n = n_classes
        self.head = LogisticRegression(C=10, max_iter=3000).fit(x[~cal], y[~cal])
        self.t = fit_temperature(self._raw(x[cal]), y[cal])

    def _raw(self, x: np.ndarray) -> np.ndarray:
        p = np.full((len(x), self.n), 1e-9, dtype="float64")
        p[:, self.head.classes_] = self.head.predict_proba(x)
        return p / p.sum(1, keepdims=True)

    def proba(self, x: np.ndarray) -> np.ndarray:
        return apply_temperature(self._raw(x), self.t)


def score(student: Student, x_test, gold, teacher) -> dict:
    p = student.proba(x_test)
    pred, conf = p.argmax(1), p.max(1)
    esc = conf < THRESHOLD
    mixed = np.where(esc, teacher, pred)
    return {
        "student": float((pred == gold).mean()),
        "cascade": float((mixed == gold).mean()),
        "test_escalated": float(esc.mean()),
    }


def run(seed: int, x_all, pool, x_test, gold, teacher, n_classes) -> dict:
    order = np.random.default_rng(seed).permutation(len(pool))
    idx, y = pool.index.values[order], pool.y.values[order]
    weeks = (len(idx) - BOOTSTRAP) // WEEK
    boot = np.arange(BOOTSTRAP)
    labelled = {a: boot.copy() for a in ARMS}
    model = {a: Student(x_all[idx[boot]], y[boot], n_classes, seed) for a in ARMS}
    rng = np.random.default_rng(seed + 100)
    out = {a: [] for a in ARMS}
    llm_calls = {a: BOOTSTRAP for a in ARMS}
    for w in range(weeks + 1):
        for a in ARMS:  # the model in force during week w, scored before it sees that week
            row = {
                "week": w,
                "labels": int(len(labelled[a])),
                **score(model[a], x_test, gold, teacher),
            }
            out[a].append(row)
        if w == weeks:
            break
        week = np.arange(BOOTSTRAP + w * WEEK, BOOTSTRAP + (w + 1) * WEEK)
        seen = np.arange(BOOTSTRAP, BOOTSTRAP + (w + 1) * WEEK)
        for a in ARMS:
            conf = model[a].proba(x_all[idx[week]]).max(1)
            esc = week[conf < THRESHOLD]
            out[a][-1]["live_escalated"] = float(len(esc) / WEEK)
            # LLM calls this week: the escalations, plus any extra labels the arm buys
            add = {
                "static": np.array([], dtype=int),
                "escalations": esc,
                "everything": week,
            }.get(a)
            if a == "escalations":
                n_esc = len(esc)  # ARMS order: "random" below matches this count
            if a == "random":
                unused = np.setdiff1d(seen, labelled[a])
                add = rng.choice(unused, min(len(unused), n_esc), replace=False)
            extra = 0 if a in ("static", "escalations") else len(np.setdiff1d(add, esc))
            llm_calls[a] += len(esc) + extra
            out[a][-1]["llm_calls_total"] = int(llm_calls[a])
            if a != "static":
                labelled[a] = np.union1d(labelled[a], add)
                model[a] = Student(x_all[idx[labelled[a]]], y[labelled[a]], n_classes, seed + w + 1)
    return out


def mean_rows(runs: list[dict], arm: str) -> list[dict]:
    rows = []
    for w in range(len(runs[0][arm])):
        cells = [r[arm][w] for r in runs]
        keys = [k for k in cells[0] if k != "week" and all(k in c for c in cells)]
        rows.append({"week": w, **{k: float(np.mean([c[k] for c in cells])) for k in keys}})
        rows[-1]["student_sd"] = float(np.std([c["student"] for c in cells]))
    return rows


def write_doc(res: dict) -> None:
    arms, teacher_acc = res["arms"], res["teacher_test_acc"]
    last = {a: arms[a][-1] for a in ARMS}
    first_esc, last_esc = arms["escalations"][0], arms["escalations"][-2]
    weeks = len(arms["static"]) - 1
    lines = [
        "# A router that learns from its escalations",
        "",
        f"_Generated by `scripts/self_improving.py` ({res['dataset']}). The training split "
        "replayed as live traffic: a frozen-MiniLM student starts from "
        f"{BOOTSTRAP:,} LLM-labelled messages, then {weeks} weeks of {WEEK:,} messages arrive. "
        f"Below {THRESHOLD} calibrated confidence a message goes to the LLM, and its answer is "
        "kept. Every arm is retrained at the end of each week. Accuracy against human labels on "
        f"the {res['n_test']:,}-message test split; mean of {len(SEEDS)} seeds (different "
        f"traffic order and cold start). LLM teacher alone: {teacher_acc:.3f}._",
        "",
        f"## After {weeks} weeks",
        "",
        "| arm | labels | LLM calls in total | student alone | cascade | escalated, last week |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for a in ARMS:
        r, live = last[a], arms[a][-2].get("live_escalated")
        lines.append(
            f"| {NAMES[a]} | {r['labels']:,.0f} | {arms[a][-2]['llm_calls_total']:,.0f} | "
            f"{r['student']:.3f} ± {r['student_sd']:.3f} | {r['cascade']:.3f} | "
            f"{live:.0%} |"
        )
    lines += [
        "",
        "LLM calls count every message the LLM answered: the cold start, every escalation "
        "(each one serves a customer), and for *random* and *label everything* the extra calls "
        "made only to get a label. ± is the spread over seeds.",
        "",
        "## Week by week, learning from escalations",
        "",
        "| week | labels | student alone | cascade | escalated that week |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in arms["escalations"]:
        live = f"{r['live_escalated']:.0%}" if "live_escalated" in r else "—"
        lines.append(
            f"| {r['week']:.0f} | {r['labels']:,.0f} | {r['student']:.3f} | {r['cascade']:.3f} | "
            f"{live} |"
        )
    acc = {a: last[a]["student"] for a in ARMS}
    calls = {a: arms[a][-2]["llm_calls_total"] for a in ARMS}
    lines += [
        "",
        "## What it shows",
        "",
        f"Learning from its own escalations, the student goes from {first_esc['student']:.3f} to "
        f"**{acc['escalations']:.3f}**, and the share of live traffic it sends to the LLM falls "
        f"from {first_esc['live_escalated']:.0%} to {last_esc['live_escalated']:.0%}. Every "
        "label it learned from was an LLM call already made to answer a customer.",
        "",
        f"- **Against labelling everything** (the ceiling, {acc['everything']:.3f}): "
        f"{(acc['everything'] - acc['escalations']) * 100:.1f} pt short, with "
        f"**{calls['escalations'] / calls['everything']:.0%} of the LLM calls** "
        f"({calls['escalations']:,.0f} vs {calls['everything']:,.0f}); the cascades score "
        f"{last['escalations']['cascade']:.3f} and {last['everything']['cascade']:.3f}.",
        "- **Against never retraining:** "
        f"{(acc['escalations'] - acc['static']) * 100:+.1f} pt, and *fewer* LLM calls "
        f"({calls['escalations']:,.0f} vs {calls['static']:,.0f}), because a static student "
        "keeps escalating "
        f"{arms['static'][-2]['live_escalated']:.0%} of traffic. Retraining pays for itself.",
        f"- **Against the same number of random labels:** "
        f"{(acc['escalations'] - acc['random']) * 100:+.1f} pt, and random labels cost "
        f"{calls['random']:,.0f} calls, since each one is bought on top of the escalations.",
        "",
        "**Caveats.** Simulated traffic: the training split in random order, so no drift and no "
        "new intents. The frozen MiniLM stands in for the served fine-tuned model because it "
        "retrains in seconds on a CPU. The LLM's answers carry its own errors "
        f"({(1 - teacher_acc) * 100:.0f} % on the test split), and the escalated messages are "
        "where it errs most (finding 6), so the student learns some of those too.",
    ]
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "self_improving.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    classes = categories()
    pos = {c: i for i, c in enumerate(classes)}
    full, test = load_split("train"), load_split("test")
    x_all = minilm_embeddings("train", full.text.tolist())
    x_test = minilm_embeddings("test", test.text.tolist())
    with (LABELS / "train.jsonl").open(encoding="utf-8") as f:
        pool = teacher_train_labels(rows=sum(1 for line in f if line.strip()))  # every label
    pool = pool.assign(y=[pos[c] for c in pool.y])
    gold = np.array([pos[c] for c in test.category])
    t = load_labels("test").teacher.reindex(test.index)
    teacher = np.array([pos.get(c, -1) if isinstance(c, str) else -1 for c in t])
    runs = []
    for s in SEEDS:
        print(f"seed {s}: {len(pool):,} labelled messages, bootstrap {BOOTSTRAP}, weeks of {WEEK}")
        runs.append(run(s, x_all, pool, x_test, gold, teacher, len(classes)))
    res = {
        "dataset": DATASET,
        "bootstrap": BOOTSTRAP,
        "week": WEEK,
        "threshold": THRESHOLD,
        "seeds": list(SEEDS),
        "n_test": len(test),
        "teacher_test_acc": float((teacher == gold).mean()),
        "arms": {a: mean_rows(runs, a) for a in ARMS},
        "runs": runs,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "self_improving.json").write_text(json.dumps(res, indent=2) + "\n")
    write_doc(res)
    print(f"-> {rel(RESULTS)}/self_improving.json, {rel(DOCS)}/self_improving.md")


if __name__ == "__main__":
    main()
