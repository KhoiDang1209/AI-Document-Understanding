from __future__ import annotations

from langchain_core.embeddings import DeterministicFakeEmbedding
from qdrant_client import QdrantClient

from docintel.config import Settings
from docintel.contracts.schema import ExtractedClause
from docintel.rag.index import index_contract
from docintel.rag.store import build_vector_store, search

_DIM = 8


def test_index_contract_indexes_clause_and_paragraph_chunks() -> None:
    settings = Settings(
        rag_embedding_dim=_DIM, qdrant_collection="t", rag_chunk_size=8, rag_chunk_overlap=2
    )
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
    count = index_contract("c1", "New York law applies here.", clauses, store, settings)
    assert count >= 2  # one clause chunk + paragraph chunks
    hits = search(store, "law", top_k=10, contract_id="c1")
    assert any(h.chunk_kind == "clause" for h in hits)
