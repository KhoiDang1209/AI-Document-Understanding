from __future__ import annotations

from collections import Counter

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_qdrant.sparse_embeddings import SparseEmbeddings, SparseVector
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


class _FakeSparse(SparseEmbeddings):
    """Deterministic token-count sparse vectors (no model download)."""

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> SparseVector:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> SparseVector:
        counts = Counter(abs(hash(token)) % 100_000 for token in text.lower().split())
        return SparseVector(indices=list(counts.keys()), values=[float(v) for v in counts.values()])


@pytest.fixture
def store() -> object:
    settings = Settings(rag_embedding_dim=_DIM, qdrant_collection="contracts_test")
    client = QdrantClient(location=":memory:")
    vector_store = build_vector_store(
        settings,
        DeterministicFakeEmbedding(size=_DIM),
        client=client,
        sparse_embedder=_FakeSparse(),
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


def test_hybrid_search_surfaces_exact_keyword_match_first(store: object) -> None:
    words = ["escrow", "insurance", "termination", "renewal", "audit", "warranty"]
    chunks = [
        TextChunk(f"clause about {word} obligations", 0, 10, i, "paragraph", None)
        for i, word in enumerate(words)
    ]
    upsert_chunks(store, "c1", chunks)
    hits = search(store, "escrow", top_k=3, contract_id="c1")
    assert hits and "escrow" in hits[0].text
