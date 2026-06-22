"""Tests for benchmark report rendering."""

from __future__ import annotations

from pathlib import Path

from docintel.optimize.benchmark import LatencyStats
from docintel.optimize.report import (
    ConfigResult,
    build_report_markdown,
    render_markdown_report,
    write_report,
)


def _result(name: str, f1: float) -> ConfigResult:
    return ConfigResult(
        name=name,
        f1=f1,
        precision=0.9,
        recall=0.8,
        accuracy=0.95,
        latency=LatencyStats(p50_ms=12.3, p95_ms=45.6, mean_ms=20.0, throughput=50.0),
        size_mb=123.4,
    )


def test_render_markdown_report_has_a_row_per_config() -> None:
    table = render_markdown_report([_result("torch-fp32", 0.881), _result("onnx-int8", 0.872)])
    assert "torch-fp32" in table
    assert "onnx-int8" in table
    assert "0.8810" in table
    assert "0.8720" in table
    # header columns present
    assert "F1" in table and "p95 (ms)" in table and "Size (MB)" in table


def test_build_report_markdown_links_plots() -> None:
    md = build_report_markdown([_result("onnx-int8", 0.872)], [Path("latency.png")])
    assert "# " in md  # has a title heading
    assert "latency.png" in md


def test_write_report_writes_file(tmp_path: Path) -> None:
    out = write_report("# Benchmark\n", tmp_path / "benchmark.md")
    assert out.read_text(encoding="utf-8") == "# Benchmark\n"
