"""Qdrant vector store wiring via langchain-qdrant.

Deterministic point ids make re-indexing a contract idempotent. ``build_vector_store``
does no network I/O (so /ask can construct it without connecting); ``ensure_collection``
and ``search``/``upsert_chunks`` are the network calls. Heavy imports live in functions.
"""

from __future__ import annotations

import uuid
from typing import Any

from langchain_core.embeddings import Embeddings

from docintel.config import Settings
from docintel.rag.chunk import TextChunk
from docintel.rag.schema import RetrievedChunk

# Fixed namespace for deterministic chunk point ids (reference data, not a tunable knob).
_POINT_NAMESPACE = uuid.UUID("6f1a2b3c-0000-4000-8000-000000000000")


def chunk_point_id(contract_id: str, chunk_index: int) -> str:
    """Stable UUID for a (contract, chunk) so re-indexing overwrites instead of duplicating."""
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{contract_id}:{chunk_index}"))


def ensure_collection(client: Any, collection: str, dim: int) -> None:
    """Create the cosine collection if it does not yet exist."""
    from qdrant_client.http.models import Distance, VectorParams

    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


def build_vector_store(settings: Settings, embedder: Embeddings, client: Any | None = None) -> Any:
    """Construct a QdrantVectorStore (no network). Pass ``client`` for tests (``:memory:``)."""
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient

    if client is None:
        client = QdrantClient(url=settings.qdrant_url)
    return QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=embedder,
        validate_collection_config=False,
    )


def upsert_chunks(store: Any, contract_id: str, chunks: list[TextChunk]) -> int:
    """Embed and upsert chunks with deterministic ids; returns the number upserted."""
    if not chunks:
        return 0
    texts = [c.text for c in chunks]
    metadatas = [
        {
            "contract_id": contract_id,
            "chunk_index": c.chunk_index,
            "chunk_kind": c.chunk_kind,
            "clause_type": c.clause_type,
            "char_start": c.char_start,
            "char_end": c.char_end,
        }
        for c in chunks
    ]
    ids = [chunk_point_id(contract_id, c.chunk_index) for c in chunks]
    store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    return len(chunks)


def search(
    store: Any, query: str, top_k: int, contract_id: str | None = None
) -> list[RetrievedChunk]:
    """Top-k similarity search, optionally filtered to one contract."""
    from qdrant_client.http.models import FieldCondition, Filter, MatchValue

    query_filter = None
    if contract_id is not None:
        query_filter = Filter(
            must=[FieldCondition(key="metadata.contract_id", match=MatchValue(value=contract_id))]
        )
    results = store.similarity_search_with_score(query, k=top_k, filter=query_filter)
    chunks: list[RetrievedChunk] = []
    for document, score in results:
        meta = document.metadata
        chunks.append(
            RetrievedChunk(
                contract_id=meta["contract_id"],
                chunk_index=meta["chunk_index"],
                chunk_kind=meta["chunk_kind"],
                clause_type=meta.get("clause_type"),
                text=document.page_content,
                score=float(score),
                char_start=meta["char_start"],
                char_end=meta["char_end"],
            )
        )
    return chunks
