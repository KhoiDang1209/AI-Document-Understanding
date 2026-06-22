"""Tests for the pure timing harness."""

from __future__ import annotations

from pathlib import Path

import pytest

from docintel.optimize.benchmark import (
    LatencyStats,
    dir_size_mb,
    latency_stats,
    measure_latency,
)


def test_latency_stats_nearest_rank_percentiles() -> None:
    stats = latency_stats([10.0, 20.0, 30.0, 40.0])
    assert stats.p50_ms == 20.0
    assert stats.p95_ms == 40.0
    assert stats.mean_ms == 25.0
    # total time 0.1s over 4 items -> 40 items/sec
    assert stats.throughput == pytest.approx(40.0)


def test_latency_stats_rejects_empty() -> None:
    with pytest.raises(ValueError):
        latency_stats([])


def test_measure_latency_calls_run_one_expected_times() -> None:
    calls: list[object] = []

    def run_one(sample: object) -> None:
        calls.append(sample)

    stats = measure_latency(run_one, samples=["a", "b"], warmup=3, repeats=2)
    # 3 warmup + 2 repeats * 2 samples = 7
    assert len(calls) == 7
    assert isinstance(stats, LatencyStats)


def test_measure_latency_rejects_empty_samples() -> None:
    with pytest.raises(ValueError):
        measure_latency(lambda s: None, samples=[], warmup=1, repeats=1)


def test_measure_latency_rejects_non_positive_repeats() -> None:
    with pytest.raises(ValueError):
        measure_latency(lambda s: None, samples=["a"], warmup=1, repeats=0)


def test_dir_size_mb_sums_files(tmp_path: Path) -> None:
    (tmp_path / "a.bin").write_bytes(b"x" * (1024 * 1024))
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"y" * (1024 * 1024))
    assert dir_size_mb(tmp_path) == pytest.approx(2.0, abs=0.01)
