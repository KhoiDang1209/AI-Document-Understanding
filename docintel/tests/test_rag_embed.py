"""Unit tests for build_embedder's stock vs local-bundle branches (no model downloads)."""

from __future__ import annotations

from typing import Any

import pytest

import docintel.rag.embed as embed_module
from docintel.config import Settings
from docintel.rag.embed import build_embedder


class _FakeTextEmbedding:
    """Records constructor and add_custom_model calls; embeds nothing."""

    events: list[tuple[str, Any]]

    @classmethod
    def add_custom_model(cls, **kwargs: Any) -> None:
        cls.events.append(("register", kwargs))

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        type(self).events.append(("init", (model_name, kwargs)))


@pytest.fixture()
def fake_fastembed(monkeypatch: pytest.MonkeyPatch) -> type[_FakeTextEmbedding]:
    _FakeTextEmbedding.events = []
    monkeypatch.setattr("fastembed.TextEmbedding", _FakeTextEmbedding)
    monkeypatch.setattr(embed_module, "_registered_custom_models", set())
    return _FakeTextEmbedding


def test_build_embedder_stock_path_unchanged(fake_fastembed: type[_FakeTextEmbedding]) -> None:
    build_embedder(Settings())
    assert fake_fastembed.events == [
        ("init", ("BAAI/bge-small-en-v1.5", {}))
    ]  # no registration, no extra kwargs


def test_build_embedder_local_bundle(fake_fastembed: type[_FakeTextEmbedding]) -> None:
    settings = Settings(rag_embedding_local_path="models/rag-embed-cuad")
    build_embedder(settings)
    registers = [e for e in fake_fastembed.events if e[0] == "register"]
    inits = [e for e in fake_fastembed.events if e[0] == "init"]
    assert len(registers) == 1
    assert registers[0][1]["model_file"] == "model.onnx"
    assert registers[0][1]["dim"] == settings.rag_embedding_dim
    assert inits[0][1][0] == embed_module._CUSTOM_MODEL_NAME
    assert inits[0][1][1]["specific_model_path"] == "models/rag-embed-cuad"


def test_build_embedder_registers_once(fake_fastembed: type[_FakeTextEmbedding]) -> None:
    settings = Settings(rag_embedding_local_path="models/rag-embed-cuad")
    build_embedder(settings)
    build_embedder(settings)
    registers = [e for e in fake_fastembed.events if e[0] == "register"]
    assert len(registers) == 1  # idempotent registration
