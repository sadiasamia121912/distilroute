"""Label a Banking77 split with the teacher LLM, resumably, within free-tier limits.

    python scripts/label.py --split test                 # 3,080 queries, ~155 calls
    python scripts/label.py --split train --limit 2000   # first 2,000 of train
    python scripts/label.py --split test --run self_agreement --limit 300 --seed 1

Every labelled query is appended to data/labels/<split>[.<run>].jsonl as soon as its batch
returns, so a rate-limit stop, a crash, or Ctrl-C loses at most one batch; re-running the
same command skips what is already done. Gold labels ride along in each record purely so the
evaluation scripts never have to re-join on text.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distilroute.teacher import RateLimited, Teacher, TeacherError  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
LABELS = ROOT / "data" / "labels"


def load_done(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {json.loads(line)["idx"] for line in f if line.strip()}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add = ap.add_argument
    add("--split", choices=["train", "test"], required=True)
    add("--provider", default="groq", choices=["groq", "gemini"])
    add("--model", default=None, help="override the provider's default model")
    add("--batch-size", type=int, default=20)
    add("--limit", type=int, default=None, help="label only the first N (after --seed shuffle)")
    add("--seed", type=int, default=None, help="shuffle order with this seed (for samples)")
    add("--run", default=None, help="tag for a separate output file, e.g. self_agreement")
    add("--max-calls", type=int, default=None, help="stop after this many API calls")
    add(
        "--descriptions",
        action="store_true",
        help="add data/intent_descriptions.json to the prompt",
    )
    add("--reasoning", default="low", choices=["low", "medium", "high"], help="gpt-oss only")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    df = pd.read_csv(RAW / f"{args.split}.csv")
    labels = json.loads((RAW / "categories.json").read_text())

    order = list(range(len(df)))
    if args.seed is not None:
        random.Random(args.seed).shuffle(order)
    if args.limit:
        order = order[: args.limit]

    name = f"{args.split}{'.' + args.run if args.run else ''}.jsonl"
    out = LABELS / name
    LABELS.mkdir(parents=True, exist_ok=True)
    done = load_done(out)
    todo = [i for i in order if i not in done]
    print(
        f"{args.split}: {len(order):,} planned, {len(done):,} already done, "
        f"{len(todo):,} to go -> {out.name}"
    )
    if not todo:
        return

    desc = None
    if args.descriptions:
        desc = json.loads((ROOT / "data" / "intent_descriptions.json").read_text(encoding="utf-8"))
        missing = [n for n in labels if n not in desc]
        assert not missing, f"no description for {missing}"
    teacher = Teacher(
        labels=labels,
        provider=args.provider,
        model=args.model,
        descriptions=desc,
        reasoning_effort=args.reasoning,
    )
    print(
        f"teacher: {args.provider} / {teacher.model}, batch {args.batch_size}, "
        f"reasoning {args.reasoning}, descriptions {'on' if desc else 'off'}"
    )
    t0 = time.time()
    failed = 0
    backoff = 5.0
    with out.open("a", encoding="utf-8") as f:
        for start in range(0, len(todo), args.batch_size):
            if args.max_calls and teacher.calls >= args.max_calls:
                print(f"stopping at --max-calls {args.max_calls}")
                break
            idxs = todo[start : start + args.batch_size]
            queries = df.text.iloc[idxs].tolist()
            while True:
                try:
                    results = teacher.label_batch(queries)
                    backoff = 5.0
                    break
                except TeacherError as e:
                    sys.exit(f"\n{e}\nNot retrying — fix the model/key and re-run to resume.")
                except RateLimited as e:
                    wait = e.retry_after or backoff
                    backoff = min(backoff * 2, 300)
                    print(f"  429 — sleeping {wait:.0f}s", flush=True)
                    time.sleep(wait)
                except requests.RequestException as e:
                    print(f"  {type(e).__name__}: {e} — sleeping {backoff:.0f}s", flush=True)
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 300)
            ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            for idx, r in zip(idxs, results, strict=True):
                failed += r.label is None
                rec = {
                    "idx": int(idx),
                    "split": args.split,
                    "text": r.text,
                    "gold": df.category.iloc[idx],
                    "teacher": r.label,
                    "raw": r.raw,
                    "provider": args.provider,
                    "model": teacher.model,
                    "reasoning": args.reasoning,
                    "descriptions": bool(desc),
                    "ts": ts,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_done = start + len(idxs)
            el = time.time() - t0
            print(
                f"  {n_done:>6,}/{len(todo):,}  calls {teacher.calls:>4}  "
                f"tokens {teacher.tokens_in + teacher.tokens_out:>8,}  unparsed {failed}  "
                f"{el:6.0f}s",
                flush=True,
            )

    print(
        f"done: calls {teacher.calls}, tokens in/out {teacher.tokens_in:,}/{teacher.tokens_out:,}, "
        f"unparsed {failed}, {time.time() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
