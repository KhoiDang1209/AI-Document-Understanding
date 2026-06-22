"""Tests for the BenchmarkConfig data container."""

from __future__ import annotations

from docintel.config import Settings
from docintel.optimize.config import BenchmarkConfig


def test_defaults_are_named_not_literal() -> None:
    config = BenchmarkConfig(source_model_name="cord-layoutlmv3")
    assert config.source_model_version == "1"
    assert config.sample_size == 50
    assert config.warmup_runs == 3
    assert config.repeats == 5
    assert config.quant_type == "dynamic-int8"


def test_from_settings_pulls_model_names() -> None:
    settings = Settings(
        kie_registered_model_name="cord-layoutlmv3",
        kie_onnx_registered_model_name="cord-layoutlmv3-onnx-int8",
    )
    config = BenchmarkConfig.from_settings(settings)
    assert config.source_model_name == "cord-layoutlmv3"
    assert config.onnx_registered_model_name == "cord-layoutlmv3-onnx-int8"


def test_with_overrides_replaces_fields() -> None:
    config = BenchmarkConfig(source_model_name="m").with_overrides(sample_size=8, repeats=2)
    assert config.sample_size == 8
    assert config.repeats == 2
    assert config.source_model_name == "m"
