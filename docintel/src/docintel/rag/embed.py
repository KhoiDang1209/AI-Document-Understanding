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


_CUSTOM_MODEL_NAME = "docintel/bge-small-cuad"
_registered_custom_models: set[str] = set()


def _register_custom_model(name: str, dim: int) -> None:
    """Idempotently register the local fine-tuned bundle layout with fastembed."""
    if name in _registered_custom_models:
        return
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType

    TextEmbedding.add_custom_model(
        model=name,
        pooling=PoolingType.CLS,
        normalization=True,
        sources=ModelSource(hf=name),  # never downloaded: a local path is always passed
        dim=dim,
        model_file="model.onnx",
    )
    _registered_custom_models.add(name)


def build_embedder(settings: Settings) -> FastEmbedEmbeddings:
    """Load the configured fastembed model (local fine-tuned bundle if set) and wrap it."""
    from fastembed import TextEmbedding

    if settings.rag_embedding_local_path:
        _register_custom_model(_CUSTOM_MODEL_NAME, settings.rag_embedding_dim)
        return FastEmbedEmbeddings(
            TextEmbedding(
                model_name=_CUSTOM_MODEL_NAME,
                specific_model_path=settings.rag_embedding_local_path,
            )
        )
    return FastEmbedEmbeddings(TextEmbedding(model_name=settings.rag_embedding_model))
