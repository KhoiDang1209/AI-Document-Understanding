"""Application configuration.

All settings are read from the environment (prefixed ``DOCINTEL_``) or a local
``.env`` file. No configuration value is hardcoded elsewhere in the codebase.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "ci", "production"]


class Settings(BaseSettings):
    """Runtime configuration for the DocIntel service."""

    model_config = SettingsConfigDict(
        env_prefix="DOCINTEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Service identity
    app_name: str = "DocIntel"
    environment: Environment = "local"
    log_level: str = "INFO"

    # HTTP server
    host: str = "0.0.0.0"
    port: int = 8000

    # Backing services (used from later phases; defaults target docker-compose).
    mlflow_tracking_uri: str = "http://mlflow:5000"
    minio_endpoint: str = "minio:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"

    # Filesystem
    data_dir: str = Field(default="data", description="Root directory for datasets.")

    # OCR pipeline (Phase 1)
    ocr_engine: Literal["doctr"] = "doctr"
    preprocess_enabled: bool = False
    preprocess_max_dim: int = 2000
    max_upload_mb: float = 10.0

    # KIE (Phase 2)
    kie_model_name: str = "microsoft/layoutlmv3-base"
    kie_registered_model_name: str = "cord-layoutlmv3"
    kie_onnx_registered_model_name: str = "cord-layoutlmv3-onnx-int8"

    # Serving + persistence (Phase 4)
    kie_onnx_model_version: str = "1"
    kie_onnx_local_path: str | None = Field(
        default=None,
        description="Local ONNX bundle dir; if set, load it instead of the MLflow registry.",
    )
    sqlite_path: str = "data/docintel.db"
    minio_bucket: str = "documents"
    minio_secure: bool = False
    validation_tolerance: float = 1.0
    confidence_threshold: float = 0.5
    default_currency: str = "IDR"

    # Streamlit UI (Phase 6)
    ui_api_base_url: str = "http://localhost:8000"
    ui_request_timeout_s: float = 120.0

    # Contract Intelligence (C1)
    contract_model_name: str = "microsoft/deberta-v3-base"
    contract_registered_model_name: str = "cuad-extractor"
    contract_onnx_registered_model_name: str = "cuad-extractor-onnx-int8"
    contract_onnx_model_version: str = "1"
    contract_onnx_local_path: str | None = Field(
        default=None,
        description="Local contract ONNX bundle dir; if set, load it instead of the registry.",
    )
    contract_max_seq_length: int = 512
    contract_doc_stride: int = 128
    contract_n_best: int = 5
    contract_max_answer_length: int = 256
    contract_no_answer_threshold: float = 0.0
    contract_max_upload_mb: float = 25.0

    # RAG / Vector retrieval (C2)
    rag_embedding_model: str = "BAAI/bge-small-en-v1.5"
    rag_embedding_dim: int = 384
    rag_chunk_size: int = 1200
    rag_chunk_overlap: int = 200
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "contract_chunks"
    rag_top_k: int = 5
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    llm_timeout_s: float = 60.0


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    return Settings()
