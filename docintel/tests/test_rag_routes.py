# tests/test_rag_routes.py
from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from qdrant_client import QdrantClient

from docintel.api.main import create_app
from docintel.api.routes.ask import get_rag_llm, get_rag_reranker, get_rag_store
from docintel.config import Settings, get_settings
from docintel.contracts.schema import ExtractedClause
from docintel.rag.index import index_contract
from docintel.rag.store import build_vector_store

_DIM = 8


def _settings() -> Settings:
    return Settings(
        rag_embedding_dim=_DIM, qdrant_collection="t", rag_chunk_size=8, rag_chunk_overlap=2
    )


def _seeded_store(settings: Settings) -> Any:
    store = build_vector_store(
        settings, DeterministicFakeEmbedding(size=_DIM), client=QdrantClient(location=":memory:")
    )
    index_contract(
        "c1",
        "New York law applies here.",
        [
            ExtractedClause(
                clause_type="Governing Law",
                answer_text="New York",
                char_start=0,
                char_end=8,
                confidence=0.9,
            )
        ],
        store,
        settings,
    )
    return store


def test_ask_degrades_without_llm() -> None:
    settings = _settings()
    store = _seeded_store(settings)
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_rag_store] = lambda: store
    app.dependency_overrides[get_rag_llm] = lambda: None
    app.dependency_overrides[get_rag_reranker] = lambda: None
    with TestClient(app) as client:
        resp = client.post("/ask", json={"question": "governing law?", "contract_id": "c1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] is None
    assert body["generation_skipped"] is True
    assert len(body["citations"]) >= 1


def test_ask_generates_with_llm() -> None:
    settings = _settings()
    store = _seeded_store(settings)
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_rag_store] = lambda: store
    app.dependency_overrides[get_rag_llm] = lambda: FakeListChatModel(responses=["NY law."])
    app.dependency_overrides[get_rag_reranker] = lambda: None
    with TestClient(app) as client:
        resp = client.post("/ask", json={"question": "governing law?"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "NY law."


def test_get_rag_reranker_is_none_when_disabled_and_cached() -> None:
    from types import SimpleNamespace

    from docintel.api.routes.ask import get_rag_reranker

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    assert get_rag_reranker(request, Settings(rag_rerank_model="")) is None
    assert request.app.state.rag_reranker_loaded is True


def test_get_rag_reranker_degrades_to_none_when_load_fails() -> None:
    from types import SimpleNamespace

    from docintel.api.routes.ask import get_rag_reranker

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    assert get_rag_reranker(request, Settings(rag_rerank_model="not/a-model")) is None


def test_ask_returns_503_when_store_unavailable() -> None:
    class _BoomStore:
        def similarity_search_with_score(self, *a: Any, **k: Any) -> Any:
            raise ConnectionError("qdrant down")

    settings = _settings()
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_rag_store] = lambda: _BoomStore()
    app.dependency_overrides[get_rag_llm] = lambda: None
    app.dependency_overrides[get_rag_reranker] = lambda: None
    with TestClient(app) as client:
        resp = client.post("/ask", json={"question": "x"})
    assert resp.status_code == 503
