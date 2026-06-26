"""Tests for configuration loading."""

from __future__ import annotations

import pytest

from docintel.config import Settings


def test_defaults() -> None:
    settings = Settings()
    assert settings.app_name == "DocIntel"
    assert settings.environment == "local"
    assert settings.port == 8000


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCINTEL_PORT", "9001")
    monkeypatch.setenv("DOCINTEL_ENVIRONMENT", "ci")
    settings = Settings()
    assert settings.port == 9001
    assert settings.environment == "ci"


def test_ocr_settings_defaults() -> None:
    settings = Settings()
    assert settings.ocr_engine == "doctr"
    assert settings.preprocess_enabled is False
    assert settings.preprocess_max_dim == 2000
    assert settings.max_upload_mb == 10.0


def test_contract_settings_defaults() -> None:
    from docintel.config import Settings

    s = Settings()
    assert s.contract_model_name == "microsoft/deberta-v3-base"
    assert s.contract_onnx_registered_model_name == "cuad-extractor-onnx-int8"
    assert s.contract_onnx_local_path is None
    assert s.contract_max_seq_length == 512
    assert s.contract_doc_stride == 128
    assert s.contract_n_best == 5
    assert s.contract_max_answer_length == 256
    assert s.contract_max_upload_mb == 25.0


def test_contract_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from docintel.config import Settings

    monkeypatch.setenv("DOCINTEL_CONTRACT_ONNX_LOCAL_PATH", "/models/cuad")
    s = Settings()
    assert s.contract_onnx_local_path == "/models/cuad"


def test_rag_settings_defaults() -> None:
    s = Settings()
    assert s.rag_embedding_model == "BAAI/bge-small-en-v1.5"
    assert s.rag_embedding_dim == 384
    assert s.rag_chunk_size == 1200
    assert s.rag_chunk_overlap == 200
    assert s.qdrant_url == "http://qdrant:6333"
    assert s.qdrant_collection == "contract_chunks"
    assert s.rag_top_k == 5
    assert s.llm_base_url is None
    assert s.llm_api_key is None
    assert s.llm_model == "Qwen/Qwen2.5-7B-Instruct"
    assert s.llm_timeout_s == 60.0


def test_rag_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCINTEL_LLM_BASE_URL", "http://ngrok/v1")
    monkeypatch.setenv("DOCINTEL_QDRANT_URL", "http://localhost:6333")
    s = Settings()
    assert s.llm_base_url == "http://ngrok/v1"
    assert s.qdrant_url == "http://localhost:6333"
