"""Smoke tests for the FastAPI app factory."""

from __future__ import annotations

from fastapi.testclient import TestClient

from docsqa.api.app import create_app


def test_health_ok() -> None:
    client = TestClient(create_app())
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"


def test_metrics_exposed() -> None:
    client = TestClient(create_app())
    resp = client.get("/metrics")
    assert resp.status_code == 200
    # A gauge registers immediately at import time.
    assert "docsqa_index_chunks" in resp.text
