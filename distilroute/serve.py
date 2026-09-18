"""FastAPI routing service: one endpoint, a `model=` switch, the same schema for every model.

    uvicorn distilroute.serve:app --port 8000
    curl -X POST localhost:8000/route -H 'content-type: application/json' \\
         -d '{"text": "my card still has not arrived", "model": "minilm_frozen_gold"}'

`GET /models` lists what is loadable (every models/<name>/ from the student scripts, plus
`teacher` when a key is configured). Models load lazily on first use and stay resident. The
default model is `DISTILROUTE_MODEL` or the first student found, so the Docker image (student
only, no torch) works with no flags.
"""

from __future__ import annotations

import os
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from distilroute import students

app = FastAPI(title="distilroute", version="0.0.1")
_loaded: dict[str, students.Router] = {}


class RouteRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    model: str | None = None


class RouteResponse(BaseModel):
    intent: str | None
    confidence: float | None
    ranked: list[str]
    model: str
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
        _loaded[name] = students.load(name)
    return _loaded[name]


@app.get("/models")
def models() -> dict:
    return {"default": default_model(), "available": list(students.available())}


@app.post("/route", response_model=RouteResponse)
def route(req: RouteRequest) -> RouteResponse:
    name = req.model or default_model()
    router = get_router(name)
    t0 = time.perf_counter()
    r = router.route(req.text)
    return RouteResponse(
        intent=r.intent,
        confidence=r.confidence,
        ranked=r.ranked,
        model=name,
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


@app.get("/health")
def health() -> dict:
    return {"ok": True, "loaded": list(_loaded)}
