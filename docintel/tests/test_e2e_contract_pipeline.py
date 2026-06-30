"""End-to-end wiring test: extract -> (index + graph) -> ask -> agent, in-process.

Heavy deps are faked (stub extractor, in-memory Qdrant + graph, fake LLM). The
RAG and graph stores are SINGLE shared instances bound to both the optional
(extract-time) and non-optional (ask/agent) getters, so data written during
extract is visible to ask/agent - that shared visibility is what proves the wiring.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from qdrant_client import QdrantClient

from docintel.api.main import create_app
from docintel.api.routes.ask import (
    get_graph_store,
    get_graph_store_optional,
    get_rag_llm,
    get_rag_store,
    get_rag_store_optional,
)
from docintel.api.routes.contracts import get_contract_extractor
from docintel.api.routes.extract import get_ocr_engine, get_s3_client
from docintel.config import Settings, get_settings
from docintel.contracts.schema import ExtractedClause
from docintel.graph.store import InMemoryGraphStore
from docintel.rag.embed import build_embedder
from docintel.rag.store import build_vector_store, ensure_collection, search
from tests.test_documents import _FakeS3

_DOC_TEXT = (
    "This Master Services Agreement is entered into by and between Acme Corporation "
    "and Globex Inc. Governing Law: this Agreement is governed by the laws of the "
    "State of New York. The Agreement expires on 2030-12-31 unless terminated earlier."
)


class _StubExtractor:
    def extract(self, text: str) -> list[ExtractedClause]:
        return [
            ExtractedClause(
                clause_type="Parties",
                answer_text="Acme Corporation and Globex Inc.",
                char_start=0,
                char_end=32,
                confidence=0.95,
            ),
            ExtractedClause(
                clause_type="Governing Law",
                answer_text="State of New York",
                char_start=33,
                char_end=50,
                confidence=0.9,
            ),
        ]


@pytest.fixture
def pipeline(tmp_path: Any, monkeypatch: Any) -> Iterator[tuple[TestClient, Any, Any]]:
    from docintel.contracts.ingest import IngestedDoc

    monkeypatch.setattr(
        "docintel.api.routes.contracts.ingest_pdf",
        lambda data, ocr_engine, settings: IngestedDoc(
            text=_DOC_TEXT, page_count=1, source="digital"
        ),
    )
    settings = Settings(sqlite_path=str(tmp_path / "c.db"))
    qdrant = QdrantClient(":memory:")
    ensure_collection(qdrant, settings.qdrant_collection, settings.rag_embedding_dim)
    rag_store = build_vector_store(settings, build_embedder(settings), client=qdrant)
    graph_store = InMemoryGraphStore()

    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_ocr_engine] = lambda: (lambda image: None)
    app.dependency_overrides[get_s3_client] = lambda: _FakeS3()
    app.dependency_overrides[get_contract_extractor] = lambda: _StubExtractor()
    app.dependency_overrides[get_rag_store_optional] = lambda: rag_store
    app.dependency_overrides[get_rag_store] = lambda: rag_store
    app.dependency_overrides[get_graph_store_optional] = lambda: graph_store
    app.dependency_overrides[get_graph_store] = lambda: graph_store
    app.dependency_overrides[get_rag_llm] = lambda: FakeListChatModel(
        responses=["The governing law is the State of New York."]
    )
    with TestClient(app) as client:
        yield client, rag_store, graph_store
    app.dependency_overrides.clear()


def test_extract_feeds_index_and_graph_then_ask_and_agent(
    pipeline: tuple[TestClient, Any, Any],
) -> None:
    client, rag_store, graph_store = pipeline

    # C1: extract
    resp = client.post(
        "/contracts/extract", files={"file": ("c.pdf", b"%PDF-1.4", "application/pdf")}
    )
    assert resp.status_code == 200
    doc = resp.json()
    contract_id = doc["id"]
    assert doc["derived"]["Parties"] == ["Acme Corporation and Globex Inc."]

    # C2: extract fed the vector store
    hits = search(rag_store, "governing law", 5, contract_id)
    assert hits and all(h.contract_id == contract_id for h in hits)

    # C3: extract fed the graph store
    assert contract_id in graph_store._data  # the fake stores contracts in a dict by id

    # C2/C3 via /ask, scoped to the contract
    ask = client.post(
        "/ask", json={"question": "What is the governing law?", "contract_id": contract_id}
    )
    assert ask.status_code == 200
    ask_body = ask.json()
    assert ask_body["answer"] == "The governing law is the State of New York."
    assert ask_body["citations"]
    assert {c["contract_id"] for c in ask_body["citations"]} == {contract_id}

    # C4 via /agent
    agent = client.post(
        "/agent", json={"task": "Summarize the governing law.", "contract_id": contract_id}
    )
    assert agent.status_code == 200
    agent_body = agent.json()
    assert agent_body["status"] == "ok"
    assert agent_body["steps"]
    assert agent_body["citations"]
    assert {c["contract_id"] for c in agent_body["citations"]} == {contract_id}
