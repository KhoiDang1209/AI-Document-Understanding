from __future__ import annotations

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from qdrant_client import QdrantClient

from docintel.config import Settings
from docintel.rag.chunk import TextChunk
from docintel.rag.store import (
    build_vector_store,
    chunk_point_id,
    ensure_collection,
    search,
    upsert_chunks,
)

_DIM = 8


@pytest.fixture
def store() -> object:
    settings = Settings(rag_embedding_dim=_DIM, qdrant_collection="contracts_test")
    client = QdrantClient(location=":memory:")
    vector_store = build_vector_store(
        settings, DeterministicFakeEmbedding(size=_DIM), client=client
    )
    ensure_collection(client, settings.qdrant_collection, _DIM)
    return vector_store


def _chunks() -> list[TextChunk]:
    return [
        TextChunk("New York law", 0, 12, 0, "clause", "Governing Law"),
        TextChunk("some paragraph body", 0, 19, 1, "paragraph", None),
    ]


def test_point_id_is_deterministic() -> None:
    assert chunk_point_id("c1", 0) == chunk_point_id("c1", 0)
    assert chunk_point_id("c1", 0) != chunk_point_id("c1", 1)


def test_upsert_then_search_filters_by_contract(store: object) -> None:
    assert upsert_chunks(store, "c1", _chunks()) == 2
    hits = search(store, "law", top_k=5, contract_id="c1")
    assert {h.contract_id for h in hits} == {"c1"}
    assert {h.chunk_kind for h in hits} == {"clause", "paragraph"}
    assert search(store, "law", top_k=5, contract_id="c2") == []


def test_reupsert_is_idempotent(store: object) -> None:
    upsert_chunks(store, "c1", _chunks())
    upsert_chunks(store, "c1", _chunks())
    assert store.client.count("contracts_test").count == 2
