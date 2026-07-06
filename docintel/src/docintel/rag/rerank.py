"""Cross-encoder reranking of retrieved chunks (fastembed ONNX, CPU, no torch).

Retrieval fetches a wider candidate pool; the cross-encoder rescores (query, text)
pairs jointly and keeps the top-k. ``build_reranker`` returns ``None`` when no model
is configured, which drives the rerank-disabled path. Heavy imports live in functions.
"""

from __future__ import annotations

from typing import Any

from docintel.config import Settings
from docintel.rag.schema import RetrievedChunk


class ChunkReranker:
    """Adapt a fastembed ``TextCrossEncoder`` to score (query, texts) pairs."""

    def __init__(self, encoder: Any) -> None:
        self._encoder = encoder

    def scores(self, query: str, texts: list[str]) -> list[float]:
        return [float(score) for score in self._encoder.rerank(query, texts)]


def build_reranker(settings: Settings) -> ChunkReranker | None:
    """Load the configured cross-encoder, or None when reranking is disabled."""
    if not settings.rag_rerank_model:
        return None
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return ChunkReranker(TextCrossEncoder(model_name=settings.rag_rerank_model))


def rerank_chunks(
    reranker: ChunkReranker, query: str, chunks: list[RetrievedChunk], top_k: int
) -> list[RetrievedChunk]:
    """Rescore chunks with the cross-encoder and return the top-k by that score."""
    if not chunks:
        return []
    scores = reranker.scores(query, [chunk.text for chunk in chunks])
    rescored = [
        chunk.model_copy(update={"score": score})
        for chunk, score in zip(chunks, scores, strict=True)
    ]
    rescored.sort(key=lambda chunk: chunk.score, reverse=True)
    return rescored[:top_k]
