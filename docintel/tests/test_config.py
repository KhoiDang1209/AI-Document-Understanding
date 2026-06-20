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
