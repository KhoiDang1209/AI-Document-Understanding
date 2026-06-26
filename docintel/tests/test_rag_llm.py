from __future__ import annotations

from docintel.config import Settings
from docintel.rag.llm import build_llm, format_context
from docintel.rag.schema import RetrievedChunk


def test_build_llm_is_none_when_unconfigured() -> None:
    assert build_llm(Settings(llm_base_url=None)) is None


def test_build_llm_when_configured() -> None:
    llm = build_llm(Settings(llm_base_url="http://ngrok/v1", llm_api_key="k", llm_model="m"))
    assert llm is not None
    assert llm.model_name == "m"


def test_format_context_numbers_and_labels_chunks() -> None:
    chunks = [
        RetrievedChunk(
            contract_id="c1",
            chunk_index=0,
            chunk_kind="clause",
            clause_type="Governing Law",
            text="New York",
            score=0.9,
            char_start=0,
            char_end=8,
        ),
        RetrievedChunk(
            contract_id="c1",
            chunk_index=1,
            chunk_kind="paragraph",
            clause_type=None,
            text="misc text",
            score=0.5,
            char_start=10,
            char_end=19,
        ),
    ]
    out = format_context(chunks)
    assert "[1] (Governing Law, contract c1): New York" in out
    assert "[2] (Excerpt, contract c1): misc text" in out
