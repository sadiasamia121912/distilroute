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


def cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def run(args: argparse.Namespace) -> None:
    texts = queries()
    base = args.url.rstrip("/")
    model = json.loads(urllib.request.urlopen(base + "/models", timeout=10).read())["default"]
    level(args.url, texts[:50], 1, 3)  # warm-up: model load, first-call allocations
    health = json.loads(urllib.request.urlopen(base + "/health", timeout=10).read())
    print(
        f"{len(texts):,} queries; model {model}, {args.cpus:g} CPUs, mode {args.mode}; "
        f"server sees {health.get('cpus')} CPUs, runs {health.get('concurrency')} inferences "
        f"at once on {health.get('threads_per_inference')} thread(s) each"
    )

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
        "mode": args.mode,
        "health": health,
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "host": {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "cpu_model": cpu_model(),
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


# Earlier runs, one service change each, kept for the record: file, what the service did then.
# Each landed on a different runner CPU, so they are not a fair comparison of the changes.
STAGES = [
    ("load_test_unpinned.json", "threads = host cores"),
    ("load_test_pinned.json", "threads = CPU quota"),
    ("load_test_serial.json", "+ one inference at a time"),
    ("load_test_parallel.json", "one single-thread inference per CPU"),
]
MODES = {  # the setups the workflow runs side by side on one runner
    "default": "one single-thread inference per CPU (the service default)",
    "serial": "one inference at a time, threads = CPUs",
}
SUMMARY_HEAD = [
    "| CPUs | service | model ms, 1 client | 1 client p50 / p95 ms | 8 clients p50 / p95 ms "
    "| sustained req/s | usd per 1M |",
    "|---:|---|---:|---:|---:|---:|---:|",
]


def cpus_label(c: float) -> str:
    return f"{c:g} CPU{'s' if c != 1 else ''}"


def price(r: dict, vm: dict, slo_ms: float) -> None:
    """Adds `sustained` and `usd_per_1m`: the VM's price, scaled by the CPUs used (a t3.small
    has 2), over the sustained rate."""
    best = r["sustained"] = sustained(r, slo_ms)
    r["usd_per_1m"] = (
        round(vm["usd_per_hour"] * r["cpus"] / 2 / (best["rps"] * 3600) * 1e6, 3) if best else None
    )


def summary_row(label: str, x: dict) -> str:
    by = {lv["clients"]: lv for lv in x["levels"]}
    one, eight, best, usd = by[1], by.get(8), x.get("sustained"), x.get("usd_per_1m")
    return (
        f"| {cpus_label(x['cpus'])} | {label} | {one['server_mean_ms']:.2f} | "
        f"{one['p50_ms']:.1f} / {one['p95_ms']:.1f} | "
        + (f"{eight['p50_ms']:.1f} / {eight['p95_ms']:.1f}" if eight else "—")
        + (f" | {best['rps']:.0f} | ${usd:.3f} |" if best and usd else " | — | — |")
    )


def same_runner(by_mode: dict[str, dict[float, dict]], cpus: list[float]) -> list[str]:
    """The fair comparison: every setup of one workflow run, side by side."""
    lines = [
        "## Two setups, same runner",
        "",
        "Both setups back to back in one job, on one machine — the only fair comparison.",
        "",
        *SUMMARY_HEAD,
    ]
    for c in cpus:
        for mode, rs in by_mode.items():
            if c in rs:
                lines.append(summary_row(MODES.get(mode, mode), rs[c]))
    for c in cpus:
        d, s = by_mode.get("default", {}).get(c), by_mode.get("serial", {}).get(c)
        if not (d and s and d["sustained"] and s["sustained"]) or c == 1:
            continue
        ed = {lv["clients"]: lv for lv in d["levels"]}.get(8)
        es = {lv["clients"]: lv for lv in s["levels"]}.get(8)
        lines += [
            "",
            f"At {cpus_label(c)} the default sustains "
            f"{d['sustained']['rps'] / s['sustained']['rps']:.1f}× the throughput of one "
            "inference at a time"
            + (
                f", with p95 at 8 clients {ed['p95_ms']:.0f} ms against {es['p95_ms']:.0f} ms"
                if ed and es
                else ""
            )
            + ".",
        ]
    d1, s1 = by_mode.get("default", {}).get(1), by_mode.get("serial", {}).get(1)
    if d1 and s1:
        setup = [
            (
                x.get("health", {}).get("concurrency"),
                x.get("health", {}).get("threads_per_inference"),
            )
            for x in (d1, s1)
        ]
        gap = abs(d1["levels"][0]["rps"] / s1["levels"][0]["rps"] - 1)
        lines += [
            "",
            f"On 1 CPU, `/health` reported {setup[0][0]} inference(s) at once on {setup[0][1]} "
            f"thread(s) for the default and {setup[1][0]} on {setup[1][1]} for the serial setup"
            + (
                f": the same configuration, so their {gap:.0%} gap at one client is run-to-run "
                "noise within one runner."
                if setup[0] == setup[1]
                else f": not the same configuration, so their {gap:.0%} gap at one client is "
                "not just noise."
            ),
        ]
    return [*lines, ""]


def history(results: Path, cpus: list[float]) -> list[str]:
    """The earlier runs, one per service change, with the CPU each landed on."""
    kept = [
        (json.loads((results / f).read_text())["runs"], label, f)
        for f, label in STAGES
        if (results / f).exists()
    ]
    if not kept:
        return []
    lines = [
        "## How the service got here (earlier runs, different runners)",
        "",
        "One run after each change: onnxruntime's pool sized to the host's cores, then to the "
        "container's quota, then one inference at a time, then one single-thread inference per "
        "CPU. **Each run landed on a different runner CPU**, and the int8 model's own time "
        "(third column) moves with the hardware, so these rows mix the service change with the "
        "machine; only the same-runner table above compares setups fairly. Files: "
        + ", ".join(f"`results/{f}`" for _, _, f in kept)
        + ".",
        "",
        *SUMMARY_HEAD,
    ]
    for c in cpus:
        for rs, label, _ in kept:
            x = next((r for r in rs if r["cpus"] == c), None)
            if x:
                cpu = (x["host"].get("cpu_model") or "?").replace(" Processor", "")
                lines.append(summary_row(f"{label} — {cpu}", x))
    if len(kept) > 1:
        first = {r["cpus"]: r for r in kept[0][0]}
        second = {r["cpus"]: r for r in kept[1][0]}
        tails = [
            f"{first[c]['levels'][0]['p95_ms']:.0f} → {second[c]['levels'][0]['p95_ms']:.0f} ms "
            f"at {cpus_label(c)}"
            for c in sorted(first.keys() & second.keys())
        ]
        lines += [
            "",
            "One change survives the hardware caveat: sizing the pool to the quota removed a "
            "tail at one client (p95 " + ", ".join(tails) + "). With one client nothing "
            "queues, so a tail of tens of ms on a model that takes a few is CPU-quota "
            "throttling, not a slower machine.",
        ]
    return [*lines, ""]


def report(paths: list[str], slo_ms: float) -> None:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    from cost import PRICES

    from distilroute.data import DOCS, RESULTS, rel

    results = RESULTS / "load_test.json"
    runs = (
        [json.loads(Path(p).read_text()) for p in paths]
        if paths
        else json.loads(results.read_text())["runs"]
    )
    runs.sort(key=lambda r: (r.get("mode", "default") != "default", r["cpus"]))
    vm = PRICES["vm"]
    by_mode: dict[str, dict[float, dict]] = {}
    for r in runs:
        price(r, vm, slo_ms)
        by_mode.setdefault(r.get("mode", "default"), {})[r["cpus"]] = r
    main = by_mode.get("default") or next(iter(by_mode.values()))
    cpus = sorted(main)
    estimate = json.loads((RESULTS / "cost.json").read_text())["rows"]
    est = estimate.get(runs[0]["model"])
    teacher = estimate["teacher_single"]["usd_per_1m"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    results.write_text(json.dumps({"slo_ms": slo_ms, "vm": vm, "runs": runs}, indent=2) + "\n")

    first = main[cpus[0]]
    host, h = first["host"], first.get("health", {})
    setup = ", ".join(
        f"{cpus_label(c)}: {main[c].get('health', {}).get('concurrency', '?')} at once"
        for c in cpus
    )
    lines = [
        "# Load test: measured throughput, latency and cost",
        "",
        "_Generated by `scripts/load_test.py --report` from GitHub Actions run "
        f"{first.get('run_id') or '?'} of `.github/workflows/load-test.yml`: the Docker image "
        f"as shipped (`{first['model']}`, int8 ONNX, one uvicorn process) on a GitHub-hosted "
        f"Linux runner ({host.get('cpu_model')}, {host['cpu_count']} vCPUs), pinned with "
        "`docker run --cpus`. Real Banking77 test queries, "
        f"{first['seconds_per_level']:.0f} s per concurrency level, timed end to end by clients "
        "on the same runner._",
        "",
        "The service runs one single-thread inference per CPU it may use and queues the rest; "
        f"`/health` in the container reported {h.get('threads_per_inference')} thread per "
        f"inference ({setup}).",
        "",
    ]
    for c in cpus:
        r = main[c]
        lines += [
            f"## {cpus_label(c)}",
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
        "and no errors. Model ms is the inference alone, as the server times it; the rest of "
        "the latency is HTTP, JSON and waiting for a free inference slot.",
        "",
        "## Cost per 1M requests",
        "",
        f"The VM price of `scripts/cost.py` ({vm['name']}, ${vm['usd_per_hour']}/h, scaled by "
        "the CPUs used) over the sustained rate:",
        "",
        "| | usd per 1M requests |",
        "|---|---:|",
    ]
    for c in cpus:
        if main[c]["usd_per_1m"] is not None:
            lines.append(
                f"| measured, {cpus_label(c)} at {main[c]['sustained']['rps']:.0f} req/s | "
                f"**${main[c]['usd_per_1m']:.3f}** |"
            )
    if est:
        lines.append(
            f"| estimated before (CPU-seconds, one at a time) | ${est['usd_per_1m']:.3f} |"
        )
    lines += [f"| the teacher LLM, one query per call (paid list price) | ${teacher:,.2f} |", ""]
    if len(by_mode) > 1:
        lines += same_runner(by_mode, cpus)
    lines += history(RESULTS, cpus)
    lines += [
        "**Caveats.** A GitHub runner is not a t3.small: different CPU, and a t3 is burstable — "
        'run flat out it needs "unlimited" CPU credits, which cost extra. The load generator '
        "shares the runner with the server (outside its CPU limit, but on the same host). "
        "Runners differ in CPU generation from run to run, so absolute numbers move between "
        "runs; compare setups only within one run.",
    ]
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "load_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {rel(results)}, {rel(DOCS)}/load_test.md")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--cpus", type=float, default=2, help="the server's CPU limit, recorded")
    ap.add_argument("--mode", default="default", help="the service setup, recorded")
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
