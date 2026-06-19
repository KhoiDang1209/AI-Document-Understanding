"""Tests for the health endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from docintel import __version__


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "DocIntel"
    assert body["version"] == __version__
    assert "environment" in body


def test_openapi_served(client: TestClient) -> None:
    assert client.get("/openapi.json").status_code == 200
