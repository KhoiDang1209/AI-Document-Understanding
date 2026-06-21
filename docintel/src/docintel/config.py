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


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    return Settings()
