# tests/test_rag_answer.py
from __future__ import annotations

from typing import Any

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda
from qdrant_client import QdrantClient

from docintel.config import Settings
from docintel.contracts.schema import ExtractedClause
from docintel.rag.answer import answer_question
from docintel.rag.index import index_contract
from docintel.rag.store import build_vector_store

_DIM = 8


def _seeded_store(settings: Settings) -> Any:
    store = build_vector_store(
        settings, DeterministicFakeEmbedding(size=_DIM), client=QdrantClient(location=":memory:")
    )
    clauses = [
        ExtractedClause(
            clause_type="Governing Law",
            answer_text="New York",
            char_start=0,
            char_end=8,
            confidence=0.9,
        )
    ]
    index_contract("c1", "New York law applies here.", clauses, store, settings)
    return store


def _settings() -> Settings:
    return Settings(
        rag_embedding_dim=_DIM,
        qdrant_collection="t",
        rag_chunk_size=8,
        rag_chunk_overlap=2,
        rag_top_k=5,
    )


def test_generate_path_returns_answer_and_citations() -> None:
    settings = _settings()
    store = _seeded_store(settings)
    llm = FakeListChatModel(responses=["Governed by New York law."])
    resp = answer_question("governing law?", store, llm, settings, contract_id="c1")
    assert resp.answer == "Governed by New York law."
    assert resp.generation_skipped is False
    assert len(resp.citations) >= 1


def test_degrade_when_no_llm() -> None:
    settings = _settings()
    store = _seeded_store(settings)
    resp = answer_question("governing law?", store, None, settings, contract_id="c1")
    assert resp.answer is None
    assert resp.generation_skipped is True
    assert len(resp.citations) >= 1


def test_degrade_when_llm_errors() -> None:
    settings = _settings()
    store = _seeded_store(settings)

    def _boom(_: Any) -> Any:
        raise RuntimeError("llm down")

    resp = answer_question(
        "governing law?", store, RunnableLambda(_boom), settings, contract_id="c1"
    )
    assert resp.answer is None
    assert resp.generation_skipped is True
    assert len(resp.citations) >= 1
