"""Tests for KIE settings and the TrainingConfig dataclass."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from docintel.config import Settings
from docintel.kie.config import TrainingConfig


def test_settings_have_kie_defaults() -> None:
    settings = Settings()
    assert settings.kie_model_name == "microsoft/layoutlmv3-base"
    assert settings.kie_registered_model_name == "cord-layoutlmv3"


def test_training_config_is_frozen_dataclass() -> None:
    config = TrainingConfig(model_name="microsoft/layoutlmv3-base")
    assert dataclasses.is_dataclass(config)
    assert config.seed == 42
    assert config.num_train_epochs == 4.0


def test_training_config_from_settings_uses_model_name() -> None:
    settings = Settings(kie_model_name="microsoft/layoutlmv3-base")
    config = TrainingConfig.from_settings(settings)
    assert config.model_name == settings.kie_model_name
    assert config.learning_rate == 1e-5


def test_env_example_lists_every_kie_setting() -> None:
    # Drift guard: every DOCINTEL_KIE_* setting must be documented in .env.example.
    env_example = Path(__file__).resolve().parents[1] / ".env.example"
    text = env_example.read_text(encoding="utf-8")
    assert "DOCINTEL_KIE_MODEL_NAME=" in text
    assert "DOCINTEL_KIE_REGISTERED_MODEL_NAME=" in text
