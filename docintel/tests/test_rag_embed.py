from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from docintel.rag.embed import FastEmbedEmbeddings


class _FakeModel:
    def embed(self, texts: list[str]) -> list[Any]:
        return [np.array([float(len(t))] * 3) for t in texts]

    def query_embed(self, texts: list[str]) -> list[Any]:
        return [np.array([1.0, 2.0, 3.0]) for _ in texts]


def test_embed_documents_returns_lists() -> None:
    embedder = FastEmbedEmbeddings(_FakeModel())
    assert embedder.embed_documents(["ab", "cde"]) == [[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]


def test_embed_query_returns_single_vector() -> None:
    embedder = FastEmbedEmbeddings(_FakeModel())
    assert embedder.embed_query("x") == [1.0, 2.0, 3.0]


@pytest.mark.slow
def test_build_embedder_real_model_has_384_dims() -> None:
    from docintel.config import Settings
    from docintel.rag.embed import build_embedder

    embedder = build_embedder(Settings())
    vector = embedder.embed_query("termination for convenience")
    assert len(vector) == 384
