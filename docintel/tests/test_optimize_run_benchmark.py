"""Tests for the pure seams in optimize.run_benchmark."""

from __future__ import annotations

from docintel.optimize.benchmark import LatencyStats
from docintel.optimize.report import ConfigResult
from docintel.optimize.run_benchmark import flatten_results_for_mlflow, slugify


def test_slugify_replaces_dashes_and_spaces() -> None:
    assert slugify("onnx-int8") == "onnx_int8"
    assert slugify("torch fp32") == "torch_fp32"


def test_flatten_results_for_mlflow_prefixes_per_config() -> None:
    result = ConfigResult(
        name="onnx-int8",
        f1=0.872,
        precision=0.9,
        recall=0.85,
        accuracy=0.93,
        latency=LatencyStats(p50_ms=10.0, p95_ms=20.0, mean_ms=12.0, throughput=80.0),
        size_mb=130.0,
    )
    flat = flatten_results_for_mlflow([result])
    assert flat["f1_onnx_int8"] == 0.872
    assert flat["p95_ms_onnx_int8"] == 20.0
    assert flat["throughput_onnx_int8"] == 80.0
    assert flat["size_mb_onnx_int8"] == 130.0
