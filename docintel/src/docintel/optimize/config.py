"""Benchmark configuration for Phase 3 optimization.

A plain data container holding the knobs for one export/quantize/benchmark
run. Defaults are named here rather than scattered as literals.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from docintel.config import Settings


@dataclass(frozen=True)
class BenchmarkConfig:
    """Knobs for one ONNX export + INT8 quantize + benchmark run."""

    source_model_name: str
    source_model_version: str = "1"
    onnx_registered_model_name: str = "cord-layoutlmv3-onnx-int8"
    sample_size: int = 50
    warmup_runs: int = 3
    repeats: int = 5
    num_threads: int = 4
    quant_type: str = "dynamic-int8"
    seed: int = 42

    @classmethod
    def from_settings(cls, settings: Settings) -> BenchmarkConfig:
        """Build a config whose model names come from service settings."""
        return cls(
            source_model_name=settings.kie_registered_model_name,
            onnx_registered_model_name=settings.kie_onnx_registered_model_name,
        )

    def with_overrides(self, **changes: Any) -> BenchmarkConfig:
        """Return a copy with the given fields replaced."""
        return replace(self, **changes)
