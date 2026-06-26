"""Index one contract's text + clauses into Qdrant (called at extract time)."""

from __future__ import annotations

from typing import Any

from docintel.config import Settings
from docintel.contracts.schema import ExtractedClause
from docintel.rag.chunk import build_chunks
from docintel.rag.store import ensure_collection, upsert_chunks


def index_contract(
    contract_id: str,
    text: str,
    clauses: list[ExtractedClause],
    store: Any,
    settings: Settings,
) -> int:
    """Chunk, embed, and upsert one contract; returns the number of chunks indexed."""
    ensure_collection(store.client, settings.qdrant_collection, settings.rag_embedding_dim)
    chunks = build_chunks(text, clauses, settings.rag_chunk_size, settings.rag_chunk_overlap)
    return upsert_chunks(store, contract_id, chunks)
