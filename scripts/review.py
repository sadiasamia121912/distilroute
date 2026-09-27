"""A real human checks the teacher's labels (roadmap 7.1).

    python scripts/review.py              # review, in your own terminal; resumable, q to stop
    python scripts/review.py --report     # afterwards: score it (reads gold) and retrain

Finding 5 (6.9) says ~500 human checks, pointed at the rows whose teacher label the student
believes least, lift the student from 0.842 to 0.876 — but its "human" was the gold label.
This replaces the simulation with a person.

**Review mode** shows the rows in the exact order `correct.py`'s disagreement strategy checks
them (same pool, same calibration hold-out, same out-of-fold ranking), one at a time: the
message, the teacher's label, and a few suggestions (the teacher's own runners-up and the
student's top guesses). The reviewer accepts, picks a suggestion, or searches all 77 intents.
Gold is **never loaded** in this mode. Answers and seconds per row go to
data/labels/review.jsonl as they are given, so stopping and resuming loses nothing.

**Report mode** reads gold for the first time: the reviewer's agreement with gold vs the
teacher's on the same rows, what was fixed, missed and broken, the time cost, and the frozen
student retrained with the reviewer's answers vs with gold's answers on the same rows (the
simulated human) vs no checks. Writes results/human_review.json and docs/human_review.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distilroute.data import (
    DOCS,
    LABELS,
    RESULTS,
    categories,
    load_split,
    rel,
    teacher_train_labels,
)  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from correct import check_order  # noqa: E402
from data_curve import minilm_embeddings  # noqa: E402
from denoise import fit_head, proba  # noqa: E402

OUT = LABELS / "review.jsonl"
CAP_SECONDS = 180  # a row open longer than this is a break, not reviewing time
HELP = """  Enter        keep the teacher's label
  1-9          pick that suggestion
  <words>      search the 77 intents (e.g. "top up", "pin"), then pick a number
  ?            list all 77 intents
  s            skip (unsure; the teacher's label stays)
  q            stop; everything so far is saved"""


def queue() -> tuple[list[dict], list[str]]:
    """The rows to review, in checking order, with what the reviewer may see. No gold."""
    classes = categories()
    train = teacher_train_labels()
    x_all = minilm_embeddings("train", load_split("train").text.tolist())
    fit, ranking, oof = check_order(train, x_all, classes)
    rows = []
    for rank, p in enumerate(ranking["disagreement"]):
        r = train.iloc[fit[p]]
        student = [classes[i] for i in np.argsort(-oof[p])[:3]]
        suggest = [c for c in dict.fromkeys([*list(r.ranked)[:3], *student]) if c != r.y]
        rows.append(
            {
                "idx": int(train.index[fit[p]]),
                "rank": rank,
                "text": r.text,
                "teacher": r.y,
                "suggest": suggest[:5],
            }
        )
    return rows, classes


def done() -> dict[int, dict]:
    if not OUT.exists():
        return {}
    recs = [json.loads(line) for line in OUT.open(encoding="utf-8") if line.strip()]
    return {r["idx"]: r for r in recs}


def pretty(intent: str) -> str:
    return intent.replace("_", " ")


def ask(row: dict, classes: list[str], n: int, total: int) -> tuple[str | None, str] | None:
    """One row. Returns (answer, action), or None to quit."""
    print(f"\n{'─' * 72}\n[{n}/{total}]  {row['text']}\n")
    print(f"  teacher says:  {pretty(row['teacher'])}")
    options = row["suggest"]
    for i, c in enumerate(options, 1):
        print(f"  {i}. {pretty(c)}")
    while True:
        cmd = input("  > ").strip()
        if cmd == "":
            return row["teacher"], "accept"
        if cmd.lower() == "q":
            return None
        if cmd.lower() == "s":
            return None, "skip"
        if cmd == "?":
            options = classes
        elif not cmd.isdigit():
            words = cmd.lower().replace("_", " ").split()
            options = [c for c in classes if all(w in pretty(c).lower() for w in words)]
            if not options:
                print("  no intent matches; try fewer words, or ? for all")
                continue
        else:
            k = int(cmd)
            if 1 <= k <= len(options):
                pick = options[k - 1]
                return pick, "accept" if pick == row["teacher"] else "change"
            print(f"  pick 1-{len(options)}")
            continue
        cols = 3 if len(options) > 12 else 1
        cells = [f"{i:>2}. {pretty(c)[:34]:34}" for i, c in enumerate(options, 1)]
        for start in range(0, len(cells), cols):
            print("  " + " ".join(cells[start : start + cols]).rstrip())


def review(limit: int) -> None:
    print("Loading the review queue (same ranking as scripts/correct.py)...")
    rows, classes = queue()
    seen = done()
    todo = [r for r in rows[:limit] if r["idx"] not in seen]
    print(
        f"\n{len(seen)} of {limit} reviewed, {len(todo)} to go. Pick the intent the customer "
        f"means; the teacher may well be right.\n{HELP}"
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as f:
        for row in todo:
            t0 = time.monotonic()
            got = ask(row, classes, len(seen) + 1, limit)
            if got is None:
                break
            answer, action = got
            rec = {
                "idx": row["idx"],
                "rank": row["rank"],
                "teacher": row["teacher"],
                "answer": answer,
                "action": action,
                "seconds": round(time.monotonic() - t0, 1),
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            seen[row["idx"]] = rec
    print(f"\n{len(seen)} of {limit} reviewed -> {rel(OUT)}")


def report() -> None:
    recs = list(done().values())
    if not recs:
        sys.exit(f"nothing reviewed yet: run `python scripts/review.py` first ({rel(OUT)})")
    classes = categories()
    pos = {c: i for i, c in enumerate(classes)}
    full, test = load_split("train"), load_split("test")
    train = teacher_train_labels()
    x_all = minilm_embeddings("train", full.text.tolist())
    x_test = minilm_embeddings("test", test.text.tolist())
    fit, _, _ = check_order(train, x_all, classes)

    # Gold is read from here on, only to score.
    gold = full.category
    n = len(recs)
    answered = [r for r in recs if r["answer"] is not None]
    teacher_ok = np.array([r["teacher"] == gold[r["idx"]] for r in recs])
    user_ok = np.array([r["answer"] == gold[r["idx"]] for r in answered])
    wrong = [r for r in recs if r["teacher"] != gold[r["idx"]]]
    secs = np.minimum([r["seconds"] for r in recs], CAP_SECONDS)
    out = {
        "reviewed": n,
        "skipped": n - len(answered),
        "changed": sum(r["action"] == "change" for r in recs),
        "teacher_acc": float(teacher_ok.mean()),
        "reviewer_acc": float(user_ok.mean()) if len(answered) else None,
        "teacher_errors": len(wrong),
        "fixed": sum(r["answer"] == gold[r["idx"]] for r in wrong),
        "missed": sum(r["action"] == "accept" for r in wrong),
        "changed_to_other_wrong": sum(
            r["action"] == "change" and r["answer"] != gold[r["idx"]] for r in wrong
        ),
        "broken": sum(r["action"] == "change" and r["teacher"] == gold[r["idx"]] for r in recs),
        "median_seconds": float(np.median(secs)),
        "total_minutes": float(secs.sum() / 60),
    }

    # Retrain: the pool with the reviewed rows relabelled by the reviewer, by gold, or not at all.
    idx = train.index.values[fit]
    base_y = np.array([pos[y] for y in train.y.iloc[fit]])
    where = {i: k for k, i in enumerate(idx)}

    def acc(fixes: dict[int, str]) -> float:
        y = base_y.copy()
        for i, c in fixes.items():
            y[where[i]] = pos[c]
        head = fit_head(x_all[idx], np.eye(len(classes), dtype="float32")[y])
        return float(
            (np.array(classes)[proba(head, x_test).argmax(1)] == test.category.values).mean()
        )

    out["acc_no_checks"] = acc({})
    out["acc_reviewer"] = acc({r["idx"]: r["answer"] for r in answered})
    out["acc_simulated"] = acc({r["idx"]: gold[r["idx"]] for r in recs})
    sim = json.loads((RESULTS / "correction.json").read_text())
    out["simulated_curve"] = {
        r["budget"]: r["acc"] for r in sim["runs"] if r["strategy"] == "disagreement"
    }
    out["teacher_test_acc"] = sim["teacher"]

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "human_review.json").write_text(json.dumps(out, indent=2) + "\n")
    write_doc(out)
    for k, v in out.items():
        print(f"  {k:24} {v}")
    print(f"-> {rel(RESULTS)}/human_review.json, {rel(DOCS)}/human_review.md")


def write_doc(o: dict) -> None:
    n = o["reviewed"]
    per_1k = o["median_seconds"] * 1000 / 3600
    curve = " · ".join(f"{b:,} → {a:.3f}" for b, a in o["simulated_curve"].items())
    lines = [
        "# A real human checks the teacher's labels",
        "",
        "_Generated by `scripts/review.py --report`. One reviewer, shown the rows the student "
        "believes least (the disagreement ranking of [human corrections](correction.md)), the "
        "teacher's label and a few suggestions — never the gold label. Gold was read only "
        "afterwards, to score._",
        "",
        "## The reviewer vs gold",
        "",
        "| | on the reviewed rows |",
        "|---|---:|",
        f"| rows reviewed | {n:,} ({o['skipped']} skipped, {o['changed']} changed) |",
        f"| teacher agrees with gold | {o['teacher_acc']:.1%} |",
        f"| reviewer agrees with gold | {o['reviewer_acc']:.1%} |"
        if o["reviewer_acc"] is not None
        else "| reviewer agrees with gold | — |",
        f"| teacher errors among them | {o['teacher_errors']} |",
        f"| … fixed | {o['fixed']} |",
        f"| … missed (teacher's label kept) | {o['missed']} |",
        f"| … changed, but to another wrong intent | {o['changed_to_other_wrong']} |",
        f"| correct teacher labels the reviewer broke | {o['broken']} |",
        f"| time per row (median) | {o['median_seconds']:.0f} s |",
        f"| total reviewing time | {o['total_minutes']:.0f} min (rows capped at {CAP_SECONDS} s) |",
        "",
        f"At the median pace, 1,000 checks take **{per_1k:.1f} hours** of one person's time.",
        "",
        "## Does the student gain what the simulation promised?",
        "",
        f"Frozen MiniLM on the {n:,}-row-corrected pool, accuracy vs gold on the full test split "
        f"(teacher: {o['teacher_test_acc']:.3f}):",
        "",
        "| labels on the reviewed rows | test accuracy |",
        "|---|---:|",
        f"| teacher's (no checks) | {o['acc_no_checks']:.3f} |",
        f"| **the reviewer's** | **{o['acc_reviewer']:.3f}** |",
        f"| gold (the simulated human of finding 5) | {o['acc_simulated']:.3f} |",
        "",
        f"The simulated curve for comparison (checks → accuracy): {curve}. The gap between the "
        "last two rows is the price of a real, fallible human; the reviewer's row is the number "
        "finding 5 can now quote.",
        "",
        "**Caveats.** One reviewer, not a trained annotator, and Banking77's own gold labels "
        'are not perfect — some "disagreements with gold" are the dataset\'s errors, not the '
        "reviewer's. The suggestions come from the teacher and the student, which can anchor "
        "the reviewer on their mistakes; that is also how a real review tool would work.",
    ]
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "human_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report", action="store_true", help="score the review (reads gold)")
    ap.add_argument("--limit", type=int, default=200, help="rows to review, in ranking order")
    args = ap.parse_args()
    report() if args.report else review(args.limit)


if __name__ == "__main__":
    main()
