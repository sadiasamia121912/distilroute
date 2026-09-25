"""Every pending LLM job, in order, resumable: stop it any time and run it again later.

    python scripts/jobs.py            # run what is left (both lanes at once)
    python scripts/jobs.py --status   # what is done, what is left, how far along

The free tiers cap tokens *per day*, so the remaining labelling needs about a week of calendar
time but only ~an hour of running per day: start this, let it use the day's allowance, stop it
(Ctrl-C, or just shut the laptop), run it again tomorrow. When a lane's allowance runs out,
`label.py` exits with DAILY_LIMIT and the lane pauses itself, so an unattended run ends on its
own. Every step checks whether its output already exists and skips itself; `label.py` resumes
from its checkpoint file and repairs a line cut off mid-write. A lock file stops two copies from
running at once and labelling the same rows twice.

Two lanes run side by side because their limits are separate: `groq` (the teacher,
gpt-oss-120b) and `openrouter` (candidate second teachers, roadmap 1b.6). Within a lane steps
run in order, and a failed step stops its lane: its output is not there, so the steps after it
would be built on nothing. Results are only written, never committed: review, then commit.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
LABELS = ROOT / "data" / "labels"
LOGS = ROOT / "logs"
LOCK = LOGS / "queue.lock"
DAILY_LIMIT = 75  # label.py exit code: the free tier's allowance for today is used up
NOISE = ["typo1", "typo3", "chat", "slang", "wrap"]
GATES = ["google/gemma-4-31b-it:free", "qwen/qwen3.8-27b:free", "z-ai/glm-5.2:free"]


def rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def label_step(name, lane, file, target, args, env=None, stop_lane=True):
    return Step(
        name,
        lane,
        [[PY, "-u", "scripts/label.py", *args]],
        done=lambda: rows(file) >= target,
        progress=lambda: f"{rows(file):,} / {target:,}",
        env=env or {},
        stop_lane=stop_lane,
    )


@dataclass
class Step:
    name: str
    lane: str
    cmds: list[list[str]]
    done: callable
    progress: callable = lambda: ""
    env: dict = field(default_factory=dict)
    stop_lane: bool = True  # a failure stops the lane (False: move on, e.g. a model that refuses)
    needs: Path | None = None  # skip (and move on) where this is missing, e.g. models/ in the cloud


def gate_tag(model: str) -> str:
    return model.split("/")[1].split(":")[0].replace(".", "").replace("-", "")


def json_has(path: Path, check) -> bool:
    try:
        return bool(check(json.loads(path.read_text(encoding="utf-8"))))
    except (OSError, ValueError, KeyError, TypeError):
        return False


CLINC = {"DISTILROUTE_DATASET": "clinc150"}
STEPS = [
    # --- groq: the teacher ------------------------------------------------------------------
    label_step(
        "clinc150 train labels (6.4)",
        "groq",
        LABELS / "clinc150" / "train.jsonl",
        3000,
        ["--split", "train", "--limit", "3000", "--top-k", "3"],
        env=CLINC,
    ),
    Step(
        "clinc150 students on teacher labels (6.4)",
        "groq",
        [
            [PY, "scripts/baseline.py", "--labels", "teacher"],
            [PY, "scripts/setfit_student.py", "--mode", "frozen", "--labels", "teacher"],
            [PY, "scripts/evaluate.py"],
        ],
        done=lambda: (ROOT / "results" / "clinc150" / "minilm_frozen_teacher.json").exists(),
        env=CLINC,
    ),
    Step(
        "teacher latency (3.2)",
        "groq",
        [[PY, "-u", "scripts/bench_latency.py", "--models", "", "--teacher", "30"]],
        done=lambda: json_has(ROOT / "results" / "latency.json", lambda d: "teacher" in d),
    ),
    label_step(
        "self-agreement labels (1.7)",
        "groq",
        LABELS / "test.self_agreement.jsonl",
        300,
        [
            "--split",
            "test",
            "--run",
            "self_agreement",
            "--limit",
            "300",
            "--seed",
            "1",
            "--descriptions",
            "--top-k",
            "3",
        ],
    ),
    *[
        label_step(
            f"teacher on noise: {k} (6.6)",
            "groq",
            LABELS / f"test.perturb_{k}.jsonl",
            300,
            [
                "--split",
                "test",
                "--perturb",
                k,
                "--limit",
                "300",
                "--seed",
                "0",
                "--descriptions",
                "--top-k",
                "3",
            ],
        )
        for k in NOISE
    ],
    Step(
        "robustness with the teacher (6.6)",
        "groq",
        [[PY, "-u", "scripts/robustness.py"]],
        done=lambda: json_has(
            ROOT / "results" / "robustness.json", lambda d: len(d["teacher"]) == len(NOISE)
        ),
        needs=ROOT / "models",  # the trained students live only on the laptop
    ),
    label_step(
        "remaining banking77 train labels (1.6)",
        "groq",
        LABELS / "train.jsonl",
        10003,
        ["--split", "train", "--seed", "0", "--descriptions", "--top-k", "3"],
    ),
    # --- openrouter: second-teacher candidates (1b.6) ---------------------------------------
    *[
        label_step(
            f"second teacher gate: {m}",
            "openrouter",
            LABELS / f"test.gate2_{gate_tag(m)}.jsonl",
            200,
            [
                "--split",
                "test",
                "--provider",
                "openrouter",
                "--model",
                m,
                "--run",
                f"gate2_{gate_tag(m)}",
                "--limit",
                "200",
                "--seed",
                "1",
                "--descriptions",
                "--top-k",
                "3",
            ],
            stop_lane=False,
        )
        for m in GATES
    ],
]


def status() -> None:
    for s in STEPS:
        mark = "done   " if s.done() else "pending"
        print(f"  [{mark}] {s.lane:10} {s.name:48} {s.progress()}")


def run_lane(lane: str, log) -> None:
    for s in [s for s in STEPS if s.lane == lane]:
        if s.done():
            continue
        if s.needs and not s.needs.exists():
            log(f"{lane}: skip   {s.name} (needs {s.needs.name}/, run it on the laptop)")
            continue
        log(f"{lane}: start  {s.name}")
        slug = "".join(c if c.isalnum() else "_" for c in s.name).strip("_")[:60]
        with (LOGS / f"queue_{slug}.log").open("a", encoding="utf-8") as out:
            ok = True
            for cmd in s.cmds:
                r = subprocess.run(
                    cmd, cwd=ROOT, env={**os.environ, **s.env}, stdout=out, stderr=subprocess.STDOUT
                )
                if r.returncode != 0:
                    ok = False
                    break
        if ok and s.done():
            log(f"{lane}: done   {s.name}")
        elif r.returncode == DAILY_LIMIT:
            log(f"{lane}: daily limit reached in {s.name}; lane paused, run again tomorrow")
            return
        else:
            log(f"{lane}: FAILED {s.name} (exit {r.returncode}, see logs/queue_{slug}.log)")
            if s.stop_lane:
                log(f"{lane}: lane stopped; fix the cause and run the queue again")
                return
    log(f"{lane}: nothing left")


def take_lock():
    """An OS-level lock on a file: released by the OS when the process dies, however it dies."""
    LOGS.mkdir(exist_ok=True)
    f = LOCK.open("a+")
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("another jobs.py is already running (logs/queue.lock); not starting a second")
    return f


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--status", action="store_true", help="show progress and exit")
    args = ap.parse_args()
    if args.status:
        status()
        return
    lock = take_lock()  # noqa: F841 — held for the life of the process
    lines = threading.Lock()

    def log(msg: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M")
        with lines, (LOGS / "queue.log").open("a", encoding="utf-8") as f:
            f.write(f"{stamp}  {msg}\n")
        print(f"{stamp}  {msg}", flush=True)

    status()
    lanes = [threading.Thread(target=run_lane, args=(ln, log)) for ln in ("groq", "openrouter")]
    for t in lanes:
        t.start()
    for t in lanes:
        t.join()
    status()


if __name__ == "__main__":
    main()
