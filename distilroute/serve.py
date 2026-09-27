"""FastAPI routing service: one endpoint, a `model=` switch, the same schema for every model.

    uvicorn distilroute.serve:app --port 8000
    curl -X POST localhost:8000/route -H 'content-type: application/json' \\
         -d '{"text": "my card still has not arrived", "model": "minilm_frozen_gold"}'

`GET /models` lists what is loadable (every models/<name>/ from the student scripts, plus
`teacher` when a key is configured). Models load lazily on first use and stay resident. The
default model is `DISTILROUTE_MODEL` or the first student found, so the Docker image (student
only, no torch) works with no flags.

Each model runs at most `DISTILROUTE_CONCURRENCY` inferences at once (default: the CPUs the
container may use, `students.cpu_limit`), each on `students.inference_threads` threads (default
1); further requests queue. One single-thread inference per CPU beat one multi-thread inference
at a time in the load test (docs/load_test.md). `GET /health` reports all three numbers.
"""

from __future__ import annotations

import os
import threading
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from distilroute import students
from distilroute.monitor import IntentMonitor

app = FastAPI(title="distilroute", version="0.0.1")
_loaded: dict[str, students.Router] = {}
_slots: dict[str, threading.BoundedSemaphore] = {}
_monitors: dict[str, IntentMonitor] = {}
_load_lock = threading.Lock()


def concurrency() -> int:
    return max(1, int(os.environ.get("DISTILROUTE_CONCURRENCY") or students.cpu_limit()))


class RouteRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    model: str | None = None


class RouteResponse(BaseModel):
    intent: str | None
    confidence: float | None
    ranked: list[str]
    model: str
    calibrated: bool  # confidence went through the model's fitted temperature
    latency_ms: float


def default_model() -> str:
    if os.environ.get("DISTILROUTE_MODEL"):
        return os.environ["DISTILROUTE_MODEL"]
    names = [n for n in students.available() if n != students.TEACHER]
    if not names:
        raise HTTPException(503, "no trained model under models/; run a student script first")
    return names[0]


def get_router(name: str) -> students.Router:
    if name not in _loaded:
        if name not in students.available():
            raise HTTPException(404, f"unknown model {name!r}; see GET /models")
        with _load_lock:  # concurrent first requests load the model once
            if name not in _loaded:
                _slots[name] = threading.BoundedSemaphore(concurrency())
                router = students.load(name)
                if getattr(router, "classes", None):  # the teacher has no fixed class list
                    _monitors[name] = IntentMonitor(
                        router.classes,
                        reference=int(os.environ.get("DISTILROUTE_MONITOR_REFERENCE") or 5000),
                        window=int(os.environ.get("DISTILROUTE_MONITOR_WINDOW") or 2000),
                    )
                _loaded[name] = router
    return _loaded[name]


@app.get("/models")
def models() -> dict:
    return {"default": default_model(), "available": list(students.available())}


@app.post("/route", response_model=RouteResponse)
def route(req: RouteRequest) -> RouteResponse:
    name = req.model or default_model()
    router = get_router(name)
    with _slots[name]:  # at most one inference per CPU; the rest queue here
        t0 = time.perf_counter()
        r = router.route(req.text)
        ms = (time.perf_counter() - t0) * 1000
    if name in _monitors:
        _monitors[name].add(r.intent)
    return RouteResponse(
        intent=r.intent,
        confidence=r.confidence,
        ranked=r.ranked,
        model=name,
        calibrated=router.temperature is not None,
        latency_ms=round(ms, 2),
    )


@app.get("/monitor")
def monitor(model: str | None = None) -> dict:
    """Has the mix of routed intents shifted? A new intent shows up as one neighbour's share
    jumping (distilroute/monitor.py, docs/new_intents.md)."""
    name = model or default_model()
    if name not in _monitors:
        raise HTTPException(404, f"no traffic monitored for {name!r} yet; route something first")
    return {"model": name, **_monitors[name].report()}


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "loaded": list(_loaded),
        "cpus": students.cpu_limit(),
        "concurrency": concurrency(),
        "threads_per_inference": students.inference_threads(),
    }
