from __future__ import annotations

import pytest
from pydantic import ValidationError

from docintel.rag.schema import AskRequest, AskResponse, RetrievedChunk


def test_ask_request_defaults() -> None:
    req = AskRequest(question="What is the governing law?")
    assert req.contract_id is None
    assert req.top_k is None


def test_ask_request_requires_question() -> None:
    with pytest.raises(ValidationError):
        AskRequest()  # type: ignore[call-arg]


def test_ask_request_rejects_empty_question() -> None:
    with pytest.raises(ValidationError):
        AskRequest(question="")


def test_ask_request_rejects_nonpositive_top_k() -> None:
    with pytest.raises(ValidationError):
        AskRequest(question="q", top_k=0)


def test_ask_response_roundtrip() -> None:
    chunk = RetrievedChunk(
        contract_id="c1",
        chunk_index=0,
        chunk_kind="clause",
        clause_type="Governing Law",
        text="New York",
        score=0.42,
        char_start=0,
        char_end=8,
    )
    resp = AskResponse(
        question="q", answer=None, generation_skipped=True, contract_id=None, citations=[chunk]
    )
    assert resp.model_dump()["citations"][0]["clause_type"] == "Governing Law"
    assert resp.answer is None
