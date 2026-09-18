"""Per-query latency of every loadable model on this machine's CPU (roadmap 3.2).

    python scripts/bench_latency.py                  # every model under models/, 500 queries
    python scripts/bench_latency.py --teacher 30     # also the LLM, end to end, 30 queries
    python scripts/bench_latency.py --http 127.0.0.1:8000 --models tfidf_lr_gold

In-process by default: one query per call through `distilroute.students`, which is exactly
what the service does per request, minus HTTP. `--http` measures through a running
`distilroute.serve` instead, so the framework overhead is visible (use 127.0.0.1, not
localhost: on Windows the latter tries ::1 first and adds ~2 s per request). The teacher
goes through the free-tier API, so it is rate-limited and measured on far fewer queries.

Writes results/latency.json (name -> p50/p95/n/host; HTTP runs under `<name>@http`);
`scripts/evaluate.py` takes the in-process numbers over the ones the training scripts
recorded, which may have come from Colab. On Windows loopback the HTTP numbers are dominated
by TCP delayed-ACK (~30 ms bimodal), so they say little about the service itself.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distilroute import students  # noqa: E402
from distilroute.data import RESULTS, load_split  # noqa: E402

OUT = RESULTS / "latency.json"


def bench(call, texts: list[str], warmup: int = 20) -> dict:
    for t in texts[:warmup]:
        call(t)
    times = []
    for t in texts:
        t0 = time.perf_counter()
        call(t)
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "mean_ms": float(np.mean(times)),
        "n": len(times),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--models", default=None, help="comma-separated subset of models/")
    ap.add_argument("--teacher", type=int, default=0, help="also time the LLM on N queries")
    ap.add_argument("--http", default=None, help="host:port of a running distilroute.serve")
    args = ap.parse_args()

    texts = load_split("test").text.sample(frac=1, random_state=0).tolist()
    names = [n for n in students.available() if n != students.TEACHER]
    if args.models:
        names = args.models.split(",")
    if args.teacher:
        names.append(students.TEACHER)

    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    host = platform.node()
    for name in names:
        n = args.teacher if name == students.TEACHER else args.n
        if args.http:
            import requests

            def call(t, url=f"http://{args.http}/route", name=name):
                requests.post(url, json={"text": t, "model": name}, timeout=120).raise_for_status()

            mode = "http"
        else:
            router = students.load(name)

            def call(t, router=router):
                router.route(t)

            mode = "in-process"
        r = bench(call, texts[:n], warmup=0 if name == students.TEACHER else 20)
        r.update(host=host, mode=mode, kind=students.available()[name]["kind"])
        results[name if mode == "in-process" else f"{name}@http"] = r
        print(f"{name:32} {mode:11} p50 {r['p50_ms']:8.1f} ms  p95 {r['p95_ms']:8.1f} ms  (n={n})")
        OUT.write_text(json.dumps(results, indent=2))  # after each model: the teacher can stall
    print(f"-> {OUT.relative_to(OUT.parents[1])}")


if __name__ == "__main__":
    main()
