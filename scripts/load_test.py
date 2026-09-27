"""How many requests per second does the served router take, and what does that cost? (roadmap 7.4)

    python scripts/load_test.py --url http://127.0.0.1:8000 --cpus 2   # against a running server
    python scripts/load_test.py --report                               # results -> docs

`scripts/cost.py` prices a student as CPU-seconds per query on one core, one request at a time:
an estimate. This measures it. Real Banking77 test queries are sent to `POST /route` by 1, 2, 4,
... `--max-concurrency` clients at once, each for `--seconds`, and every response is timed end to
end (network, JSON, tokenizer, model): throughput, p50 / p95 / p99 latency, errors. The server is
the Docker image exactly as shipped, pinned to `--cpus` CPUs; `.github/workflows/load-test.yml`
runs it on a Linux GitHub runner (Windows loopback would distort it) and uploads the JSON.

Cost per 1M requests is then the VM's hourly price over the throughput it sustains: the best
throughput whose p99 stays under `--slo-ms`. Standard library only in load mode, so it runs on
a bare runner; `--report` merges the downloaded runs into results/load_test.json and writes
docs/load_test.md.
"""

from __future__ import annotations

import argparse
import csv
import http.client
import io
import itertools
import json
import os
import platform
import sys
import threading
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
TEST_CSV = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"
SLO_MS = 500


def queries() -> list[str]:
    local = ROOT / "data" / "raw" / "test.csv"
    text = (
        local.read_text(encoding="utf-8")
        if local.exists()
        else urllib.request.urlopen(TEST_CSV, timeout=60).read().decode("utf-8")
    )
    return [row["text"] for row in csv.DictReader(io.StringIO(text))]


def pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def level(url: str, texts: list[str], clients: int, seconds: float) -> dict:
    """`clients` threads, each with its own keep-alive connection, sending as fast as answered."""
    u = urlparse(url)
    order = itertools.cycle(texts)
    lock = threading.Lock()
    lat, server_ms, errors = [], [], []
    deadline = time.perf_counter() + seconds

    def client() -> None:
        conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=30)
        while time.perf_counter() < deadline:
            with lock:
                text = next(order)
            body = json.dumps({"text": text})
            t0 = time.perf_counter()
            try:
                conn.request("POST", "/route", body, {"content-type": "application/json"})
                resp = conn.getresponse()
                data = resp.read()
                ms = (time.perf_counter() - t0) * 1000
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}")
                with lock:
                    lat.append(ms)
                    server_ms.append(json.loads(data)["latency_ms"])
            except Exception as e:  # noqa: BLE001 - every failure is a data point here
                with lock:
                    errors.append(type(e).__name__ + ": " + str(e)[:80])
                conn.close()
                conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=30)
        conn.close()

    t0 = time.perf_counter()
    threads = [threading.Thread(target=client) for _ in range(clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0
    return {
        "clients": clients,
        "seconds": round(elapsed, 2),
        "ok": len(lat),
        "errors": len(errors),
        "error_examples": sorted(set(errors))[:3],
        "rps": round(len(lat) / elapsed, 1),
        "p50_ms": pct(lat, 0.50),
        "p95_ms": pct(lat, 0.95),
        "p99_ms": pct(lat, 0.99),
        "server_mean_ms": round(sum(server_ms) / len(server_ms), 2) if server_ms else None,
    }


def run(args: argparse.Namespace) -> None:
    texts = queries()
    base = args.url.rstrip("/")
    health = json.loads(urllib.request.urlopen(base + "/health", timeout=10).read())
    model = json.loads(urllib.request.urlopen(base + "/models", timeout=10).read())["default"]
    print(f"{len(texts):,} queries; server ok={health['ok']}, model {model}, {args.cpus:g} CPUs")
    level(args.url, texts[:50], 1, 3)  # warm-up: model load, first-call allocations

    levels, clients = [], 1
    while clients <= args.max_concurrency:
        r = level(args.url, texts, clients, args.seconds)
        levels.append(r)
        print(
            f"  {clients:>3} clients  {r['rps']:>7.1f} req/s  p50 {r['p50_ms']:.1f}  "
            f"p95 {r['p95_ms']:.1f}  p99 {r['p99_ms']:.1f} ms  (server {r['server_mean_ms']} ms)"
            f"  errors {r['errors']}",
            flush=True,
        )
        clients *= 2
    out = {
        "model": model,
        "cpus": args.cpus,
        "host": {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "processor": platform.processor() or None,
            "runner": os.environ.get("RUNNER_NAME"),
        },
        "seconds_per_level": args.seconds,
        "levels": levels,
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"-> {args.out}")


def sustained(run: dict, slo_ms: float) -> dict | None:
    """The highest-throughput level whose p99 is within the SLO and that had no errors."""
    ok = [lv for lv in run["levels"] if lv["errors"] == 0 and lv["p99_ms"] <= slo_ms]
    return max(ok, key=lambda lv: lv["rps"]) if ok else None


BASELINE = "load_test_unpinned.json"  # the first run, before threads followed the CPU quota


def before_after(runs: list[dict], slo_ms: float) -> list[str]:
    """Doc lines comparing these runs with the unpinned first run, if it is kept."""
    sys.path.insert(0, str(ROOT))
    from distilroute.data import RESULTS

    path = RESULTS / BASELINE
    if not path.exists():
        return []
    old = {r["cpus"]: r for r in json.loads(path.read_text())["runs"]}
    lines = [
        "## Before and after pinning onnxruntime's threads",
        "",
        f"The first run (`results/{BASELINE}`) let onnxruntime size its thread pool to the "
        "runner's cores; since then `students.cpu_limit` sizes it to the container's CPU quota.",
        "",
        "| CPUs | | 1 client p50 / p95 / p99 ms | sustained req/s | usd per 1M |",
        "|---:|---|---:|---:|---:|",
    ]
    for r in runs:
        o = old.get(r["cpus"])
        for label, x in [("before", o), ("after", r)]:
            if not x:
                continue
            one, best = x["levels"][0], sustained(x, slo_ms)
            lines.append(
                f"| {r['cpus']:g} | {label} | {one['p50_ms']:.1f} / {one['p95_ms']:.1f} / "
                f"{one['p99_ms']:.1f} | {best['rps']:.0f} | ${x['usd_per_1m']:.3f} |"
                if best
                else f"| {r['cpus']:g} | {label} | — | — | — |"
            )
    return [*lines, ""]


def report(paths: list[str], slo_ms: float) -> None:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    from cost import PRICES

    from distilroute.data import DOCS, RESULTS, rel

    results = RESULTS / "load_test.json"
    runs = [json.loads(Path(p).read_text()) for p in paths] if paths else None
    if runs is None:
        runs = json.loads(results.read_text())["runs"]
    runs.sort(key=lambda r: r["cpus"])
    vm = PRICES["vm"]
    for r in runs:
        best = sustained(r, slo_ms)
        r["sustained"] = best
        # The VM's price scales with its vCPUs; a t3.small is 2, so price a run by its share.
        r["usd_per_1m"] = (
            round(vm["usd_per_hour"] * r["cpus"] / 2 / (best["rps"] * 3600) * 1e6, 3)
            if best
            else None
        )
    estimate = json.loads((RESULTS / "cost.json").read_text())["rows"]
    est = estimate.get(runs[0]["model"])
    teacher = estimate["teacher_single"]["usd_per_1m"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    results.write_text(json.dumps({"slo_ms": slo_ms, "vm": vm, "runs": runs}, indent=2) + "\n")

    host = runs[0]["host"]
    lines = [
        "# Load test: measured throughput, latency and cost",
        "",
        f"_Generated by `scripts/load_test.py --report` from runs of "
        "`.github/workflows/load-test.yml`: the Docker image as shipped "
        f"(`{runs[0]['model']}`, int8 ONNX, one uvicorn process) on a GitHub-hosted Linux runner "
        f"({host['cpu_count']} vCPUs, {host['platform']}), pinned with `docker run --cpus`. Real "
        f"Banking77 test queries, {runs[0]['seconds_per_level']:.0f} s per concurrency level, "
        "timed end to end by clients on the same runner._",
        "",
    ]
    for r in runs:
        lines += [
            f"## {r['cpus']:g} CPU{'s' if r['cpus'] != 1 else ''}",
            "",
            "| clients | req/s | p50 ms | p95 ms | p99 ms | model ms (server) | errors |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for lv in r["levels"]:
            mark = "**" if lv is r["sustained"] else ""
            lines.append(
                f"| {lv['clients']} | {mark}{lv['rps']:.0f}{mark} | {lv['p50_ms']:.1f} | "
                f"{lv['p95_ms']:.1f} | {lv['p99_ms']:.1f} | {lv['server_mean_ms']} | "
                f"{lv['errors']} |"
            )
        lines.append("")
    lines += [
        f"**Bold** = the sustained rate: the highest throughput with p99 under {slo_ms:.0f} ms "
        "and no errors.",
        "",
        "## Cost per 1M requests",
        "",
        f"The VM price of `scripts/cost.py` ({vm['name']}, ${vm['usd_per_hour']}/h, scaled by "
        "the CPUs used) over the sustained rate:",
        "",
        "| | usd per 1M requests |",
        "|---|---:|",
    ]
    for r in runs:
        if r["usd_per_1m"] is not None:
            lines.append(
                f"| measured, {r['cpus']:g} CPU{'s' if r['cpus'] != 1 else ''} at "
                f"{r['sustained']['rps']:.0f} req/s | **${r['usd_per_1m']:.3f}** |"
            )
    if est:
        lines.append(
            f"| estimated before (CPU-seconds, one at a time) | ${est['usd_per_1m']:.3f} |"
        )
    lines += [
        f"| the teacher LLM, one query per call (paid list price) | ${teacher:,.2f} |",
        "",
        *before_after(runs, slo_ms),
        "**Tail latency.** "
        + " ".join(
            f"At {r['cpus']:g} CPU{'s' if r['cpus'] != 1 else ''} and one client, the median "
            f"request takes {r['levels'][0]['p50_ms']:.1f} ms but p95 is "
            f"{r['levels'][0]['p95_ms']:.0f} ms."
            for r in runs
        )
        + (
            " The table above shows what pinning the threads to the CPU quota changed; any tail "
            "left is the runner's (shared host, the load generator on the same machine)."
            if (RESULTS / BASELINE).exists()
            else " A tail that jumps like that with no queueing is typical of CPU-quota "
            "throttling: onnxruntime sizes its thread pool to the host's cores, not to the "
            "`--cpus` limit, spends the quota early in each scheduling period and then waits."
        ),
        "",
        "**Caveats.** A GitHub runner is not a t3.small: different CPU, and a t3 is burstable — "
        'run flat out it needs "unlimited" CPU credits, which cost extra. The load generator '
        "shares the runner with the server (outside its CPU limit, but on the same host). "
        "One run per configuration; runner speed varies from day to day.",
    ]
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "load_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {rel(results)}, {rel(DOCS)}/load_test.md")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--cpus", type=float, default=2, help="the server's CPU limit, recorded")
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--max-concurrency", type=int, default=32)
    ap.add_argument("--out", default="load_test.json")
    ap.add_argument("--report", nargs="*", metavar="RUN_JSON", help="merge runs, write docs")
    ap.add_argument("--slo-ms", type=float, default=SLO_MS)
    args = ap.parse_args()
    if args.report is not None:
        report(args.report, args.slo_ms)
    else:
        run(args)


if __name__ == "__main__":
    main()
