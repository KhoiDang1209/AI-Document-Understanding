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


def test_retrieval_query_overrides_search_text_but_not_question() -> None:
    settings = Settings(
        rag_embedding_dim=_DIM,
        qdrant_collection="t",
        rag_chunk_size=25,
        rag_chunk_overlap=0,
        rag_top_k=5,
    )
    store = build_vector_store(
        settings, DeterministicFakeEmbedding(size=_DIM), client=QdrantClient(location=":memory:")
    )
    text = "escrow deposit duty. insurance coverage. termination notice."
    index_contract("c1", text, [], store, settings)
    resp = answer_question(
        "highlight the relevant parts",
        store,
        None,
        settings,
        contract_id="c1",
        retrieval_query="escrow",
    )
    assert resp.question == "highlight the relevant parts"
    assert "escrow" in resp.citations[0].text


def test_reranker_reorders_and_truncates_citations() -> None:
    from docintel.rag.rerank import ChunkReranker

    class _KeywordEncoder:
        def rerank(self, query: str, texts: list[str]) -> list[float]:
            return [1.0 if query in text else 0.0 for text in texts]

    settings = _settings()
    store = _seeded_store(settings)
    resp = answer_question(
        "applies",
        store,
        None,
        settings,
        contract_id="c1",
        top_k=1,
        reranker=ChunkReranker(_KeywordEncoder()),
    )
    assert len(resp.citations) == 1
    assert "applies" in resp.citations[0].text


def test_generate_or_degrade_is_shared() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from docintel.rag.answer import generate_or_degrade
    from docintel.rag.schema import RetrievedChunk

    cite = RetrievedChunk(
        contract_id="c1",
        chunk_index=0,
        chunk_kind="graph",
        clause_type="Expiration Date",
        text="expires 2026-01-01",
        score=1.0,
        char_start=0,
        char_end=18,
    )
    ok = generate_or_degrade("q", [cite], FakeListChatModel(responses=["A."]), "c1")
    assert ok.answer == "A." and ok.generation_skipped is False and ok.citations == [cite]
    degraded = generate_or_degrade("q", [cite], None, "c1")
    assert degraded.answer is None and degraded.generation_skipped is True
