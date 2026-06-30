"""Unit tests for the HTTP demo script (mocked transport; no live server)."""

from __future__ import annotations

import json

import httpx
import pytest

from docintel.scripts.demo_pipeline import build_sample_pdf, run_demo


def _route(req: httpx.Request) -> httpx.Response:
    path = req.url.path
    if path == "/contracts/extract":
        return httpx.Response(
            200, json={"id": "c1", "source": "digital", "clauses": [], "derived": {}}
        )
    if path == "/ask":
        return httpx.Response(
            200, json={"answer": "New York", "generation_skipped": False, "citations": []}
        )
    if path == "/agent":
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "steps": ["route:vector"],
                "answer": "ok",
                "citations": [],
                "trace_id": None,
            },
        )
    return httpx.Response(404)


def test_run_demo_threads_contract_id(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return _route(req)

    transport = httpx.MockTransport(handler)

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        with httpx.Client(transport=transport) as client:
            return client.post(url, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx, "post", fake_post)
    rc = run_demo("http://api:8000", 5.0, b"%PDF-1.4")
    assert rc == 0
    assert [r.url.path for r in calls] == ["/contracts/extract", "/ask", "/agent"]
    assert json.loads(calls[1].content)["contract_id"] == "c1"
    assert json.loads(calls[2].content)["contract_id"] == "c1"


def test_run_demo_reports_connection_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", fake_post)
    rc = run_demo("http://api:8000", 5.0, b"%PDF")
    assert rc == 1
    assert "Is the API running?" in capsys.readouterr().err


def test_build_sample_pdf_returns_pdf_bytes() -> None:
    pytest.importorskip("fitz")
    data = build_sample_pdf()
    assert data.startswith(b"%PDF")
