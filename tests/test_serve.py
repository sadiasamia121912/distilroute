"""Service tests with a fake router injected — no trained model, no key, no network."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from distilroute import serve, students


class FakeRouter(students.Router):
    name, kind = "fake_gold", "tfidf"
    classes = ["card_arrival", "exchange_rate", "top_up_failed"]

    def proba(self, text: str):
        p = np.array([0.7, 0.2, 0.1]) if "card" in text else np.array([0.1, 0.1, 0.8])
        return self.classes, p


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(students, "available", lambda: {"fake_gold": {"kind": "tfidf"}})
    monkeypatch.setattr(students, "load", lambda name: FakeRouter())
    monkeypatch.delenv("DISTILROUTE_MODEL", raising=False)
    serve._loaded.clear()
    serve._slots.clear()
    return TestClient(serve.app)


def test_route_returns_ranked_intents_and_latency(client):
    r = client.post("/route", json={"text": "my card never came"})
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "card_arrival"
    assert body["ranked"] == ["card_arrival", "exchange_rate", "top_up_failed"]
    assert body["confidence"] == pytest.approx(0.7)
    assert body["model"] == "fake_gold"
    assert body["calibrated"] is False
    assert body["latency_ms"] >= 0


def test_model_switch_and_listing(client):
    assert client.get("/models").json() == {"default": "fake_gold", "available": ["fake_gold"]}
    assert client.post("/route", json={"text": "top up", "model": "fake_gold"}).status_code == 200
    r = client.post("/route", json={"text": "top up", "model": "nope"})
    assert r.status_code == 404
    assert "nope" in r.json()["detail"]


def test_models_load_once_and_stay_resident(client):
    client.post("/route", json={"text": "a"})
    client.post("/route", json={"text": "b"})
    assert client.get("/health").json() == {"ok": True, "loaded": ["fake_gold"]}


def test_confidence_is_calibrated_when_the_model_has_a_temperature(client, monkeypatch):
    class Calibrated(FakeRouter):
        temperature = 0.5  # sharpen: 0.7 -> 0.7^2 / (0.7^2 + 0.2^2 + 0.1^2)

    monkeypatch.setattr(students, "load", lambda name: Calibrated())
    body = client.post("/route", json={"text": "card"}).json()
    assert body["calibrated"] is True
    assert body["confidence"] == pytest.approx(0.49 / 0.54, rel=1e-3)
    assert body["ranked"][0] == "card_arrival"


def test_empty_text_rejected(client):
    assert client.post("/route", json={"text": ""}).status_code == 422


def test_no_models_is_a_clear_503(client, monkeypatch):
    monkeypatch.setattr(students, "available", lambda: {})
    r = client.post("/route", json={"text": "hello"})
    assert r.status_code == 503


def test_cpu_limit_follows_the_container_quota(monkeypatch):
    monkeypatch.delenv("DISTILROUTE_THREADS", raising=False)
    host = students.cpu_limit("max 100000")  # no quota: every CPU the process may run on
    assert host >= 1
    assert students.cpu_limit("100000 100000") == 1  # docker run --cpus=1
    assert students.cpu_limit("150000 100000") == min(host, 2)  # --cpus=1.5 rounds up
    assert students.cpu_limit("") == host  # no cgroup file (Windows, macOS)
    monkeypatch.setenv("DISTILROUTE_THREADS", "3")
    assert students.cpu_limit("100000 100000") == 3


def test_inference_is_serialised_per_model(client, monkeypatch):
    import threading
    import time

    running, peak = [0], [0]
    lock = threading.Lock()

    class Slow(FakeRouter):
        def proba(self, text):
            with lock:
                running[0] += 1
                peak[0] = max(peak[0], running[0])
            time.sleep(0.02)
            with lock:
                running[0] -= 1
            return super().proba(text)

    monkeypatch.setattr(students, "load", lambda name: Slow())
    threads = [
        threading.Thread(target=client.post, args=("/route",), kwargs={"json": {"text": "card"}})
        for _ in range(6)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == 1
