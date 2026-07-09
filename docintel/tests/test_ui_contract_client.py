"""Unit tests for the contract UI client helpers."""

from __future__ import annotations

import httpx
import pytest

from docintel.ui.contract_client import (
    ContractApiError,
    ask_question,
    citation_rows,
    clause_rows,
    extract_contract,
    fetch_health,
    graph_dot,
    run_agent,
)

_BASE_URL = "http://api:8000"


def _with_transport(transport: httpx.MockTransport, call):
    """Run `call` with httpx.post routed through a mock transport."""
    real_post = httpx.post

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        with httpx.Client(transport=transport) as client:
            return client.post(url, **kwargs)  # type: ignore[arg-type]

    httpx.post = fake_post  # type: ignore[assignment]
    try:
        return call()
    finally:
        httpx.post = real_post  # type: ignore[assignment]


def _with_get_transport(transport: httpx.MockTransport, call):
    """Run `call` with httpx.get routed through a mock transport."""
    real_get = httpx.get

    def fake_get(url: str, **kwargs: object) -> httpx.Response:
        with httpx.Client(transport=transport) as client:
            return client.get(url, **kwargs)  # type: ignore[arg-type]

    httpx.get = fake_get  # type: ignore[assignment]
    try:
        return call()
    finally:
        httpx.get = real_get  # type: ignore[assignment]


def test_extract_contract_success() -> None:
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={"id": "c1"}))
    result = _with_transport(transport, lambda: extract_contract(_BASE_URL, 5.0, "c.pdf", b"%PDF"))
    assert result == {"id": "c1"}


def test_ask_question_sends_contract_id() -> None:
    seen: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(req.content))
        return httpx.Response(200, json={"answer": "NY"})

    transport = httpx.MockTransport(handler)
    result = _with_transport(transport, lambda: ask_question(_BASE_URL, 5.0, "law?", "c1"))
    assert result == {"answer": "NY"}
    assert seen == {"question": "law?", "contract_id": "c1"}


def test_ask_question_omits_empty_contract_id() -> None:
    seen: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(req.content))
        return httpx.Response(200, json={"answer": "NY"})

    _with_transport(
        httpx.MockTransport(handler), lambda: ask_question(_BASE_URL, 5.0, "law?", None)
    )
    assert "contract_id" not in seen


def test_run_agent_success() -> None:
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={"status": "ok"}))
    result = _with_transport(transport, lambda: run_agent(_BASE_URL, 5.0, "summarize", "c1"))
    assert result == {"status": "ok"}


def test_error_detail_and_status() -> None:
    transport = httpx.MockTransport(
        lambda req: httpx.Response(503, json={"detail": "Vector store unavailable."})
    )
    with pytest.raises(ContractApiError, match=r"503.*Vector store unavailable"):
        _with_transport(transport, lambda: ask_question(_BASE_URL, 5.0, "q", None))


def test_timeout_and_connection_errors() -> None:
    def raise_timeout(req: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("slow", request=req)

    with pytest.raises(ContractApiError, match="timed out"):
        _with_transport(
            httpx.MockTransport(raise_timeout),
            lambda: extract_contract(_BASE_URL, 5.0, "c.pdf", b"%PDF"),
        )

    def raise_connect(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=req)

    with pytest.raises(ContractApiError, match="Could not reach"):
        _with_transport(
            httpx.MockTransport(raise_connect),
            lambda: run_agent(_BASE_URL, 5.0, "t", None),
        )


def test_clause_rows_maps_fields_and_missing() -> None:
    document = {
        "clauses": [
            {"clause_type": "Parties", "answer_text": "Acme", "confidence": 0.9},
            {"clause_type": None, "answer_text": None, "confidence": None},
        ]
    }
    rows = clause_rows(document)
    assert rows[0] == {"Type": "Parties", "Text": "Acme", "Confidence": 0.9}
    assert rows[1] == {"Type": "—", "Text": "—", "Confidence": None}


def test_citation_rows_maps_fields() -> None:
    response = {
        "citations": [
            {"contract_id": "c1", "clause_type": "Governing Law", "score": 0.8, "text": "NY"}
        ]
    }
    rows = citation_rows(response)
    assert rows[0] == {"Contract": "c1", "Clause": "Governing Law", "Score": 0.8, "Text": "NY"}


def test_fetch_health_success() -> None:
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, json={"status": "ok", "version": "1.2.3"})
    )
    result = _with_get_transport(transport, lambda: fetch_health(_BASE_URL, 5.0))
    assert result["status"] == "ok"
    assert result["version"] == "1.2.3"


def test_fetch_health_maps_connection_error() -> None:
    def raise_connect(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=req)

    with pytest.raises(ContractApiError, match="Could not reach"):
        _with_get_transport(
            httpx.MockTransport(raise_connect), lambda: fetch_health(_BASE_URL, 5.0)
        )


def test_graph_dot_renders_contract_and_fact_nodes() -> None:
    response = {
        "citations": [
            {"contract_id": "c1", "clause_type": "Expiration Date", "text": "expires 2026-01-01"},
            {"contract_id": "c1", "clause_type": "Renewal Term", "text": "auto-renews yearly"},
        ]
    }
    dot = graph_dot(response)
    assert dot.startswith("digraph G {")
    # One contract node (deduped) linked to two fact nodes.
    assert dot.count("shape=box") == 1
    assert dot.count("shape=ellipse") == 2
    assert '"contract:c1" -> "fact:0";' in dot
    assert "Expiration Date" in dot


def test_graph_dot_escapes_quotes() -> None:
    response = {"citations": [{"contract_id": 'a"b', "clause_type": "X", "text": ""}]}
    dot = graph_dot(response)
    assert '\\"' in dot  # the embedded quote is escaped, not left raw
