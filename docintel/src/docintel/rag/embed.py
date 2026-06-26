"""bge-small-en-v1.5 embeddings via fastembed (ONNX, CPU, no torch).

Wrapped behind LangChain's ``Embeddings`` interface so the Qdrant store can use it.
The fastembed model is imported lazily so this module stays cheap to import.
"""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings

from docintel.config import Settings


class FastEmbedEmbeddings(Embeddings):
    """Adapt a fastembed text-embedding model to the LangChain Embeddings API."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        vector = next(iter(self._model.query_embed([text])))
        return list(vector.tolist())


def build_embedder(settings: Settings) -> FastEmbedEmbeddings:
    """Load the configured fastembed model and wrap it."""
    from fastembed import TextEmbedding

    return FastEmbedEmbeddings(TextEmbedding(model_name=settings.rag_embedding_model))
