from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from qdrant_client import QdrantClient

from docintel.api.main import create_app
from docintel.api.routes.ask import get_graph_store, get_rag_llm, get_rag_store
from docintel.config import Settings, get_settings
from docintel.graph.schema import ExpirationFact, GraphContract
from docintel.graph.store import InMemoryGraphStore
from docintel.rag.chunk import build_chunks
from docintel.rag.embed import build_embedder
from docintel.rag.store import build_vector_store, ensure_collection, upsert_chunks


def _vector_store(settings: Settings) -> Any:
    client = QdrantClient(":memory:")
    ensure_collection(client, settings.qdrant_collection, settings.rag_embedding_dim)
    store = build_vector_store(settings, build_embedder(settings), client=client)
    upsert_chunks(store, "a", build_chunks("Governing law is New York.", [], 1200, 200))
    return store


def _graph_store() -> InMemoryGraphStore:
    store = InMemoryGraphStore()
    store.upsert_contract(
        GraphContract(
            contract_id="a",
            expiration=ExpirationFact(
                iso_date="2999-01-01", answer_text="expires 2999-01-01", char_start=0, char_end=18
            ),
        )
    )
    return store


def _client(llm: Any) -> TestClient:
    app = create_app()
    settings = Settings()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_rag_store] = lambda: _vector_store(settings)
    app.dependency_overrides[get_graph_store] = lambda: _graph_store()
    app.dependency_overrides[get_rag_llm] = lambda: llm
    return TestClient(app)


def test_agent_route_generates_answer() -> None:
    with _client(FakeListChatModel(responses=["New York."])) as client:
        resp = client.post("/agent", json={"task": "What is the governing law?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok" and body["answer"] == "New York."
    assert body["steps"] and body["trace_id"] is None


def test_agent_route_degrades_without_llm() -> None:
    with _client(None) as client:
        resp = client.post("/agent", json={"task": "which contracts expire within 400000 days?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded" and body["answer"] is None
    assert [c["contract_id"] for c in body["citations"]] == ["a"]


def test_agent_route_validates_empty_task() -> None:
    with _client(None) as client:
        resp = client.post("/agent", json={"task": ""})
    assert resp.status_code == 422
