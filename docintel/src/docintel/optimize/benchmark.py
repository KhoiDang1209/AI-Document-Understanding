"""Pure CPU timing harness for the optimization benchmark.

No model-specific code: callers inject a ``run_one`` callable, so the same
harness times PyTorch and ONNX Runtime configs identically.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LatencyStats:
    """Per-document latency summary plus throughput."""

    p50_ms: float
    p95_ms: float
    mean_ms: float
    throughput: float


def latency_stats(latencies_ms: Sequence[float]) -> LatencyStats:
    """Summarize per-call latencies with nearest-rank percentiles."""
    if not latencies_ms:
        raise ValueError("latencies_ms must be non-empty")
    ordered = sorted(latencies_ms)
    n = len(ordered)

    def _percentile(p: float) -> float:
        rank = max(1, math.ceil(p / 100.0 * n))
        return ordered[rank - 1]

    mean_ms = sum(ordered) / n
    total_seconds = sum(ordered) / 1000.0
    throughput = n / total_seconds if total_seconds > 0 else 0.0
    return LatencyStats(
        p50_ms=_percentile(50.0),
        p95_ms=_percentile(95.0),
        mean_ms=mean_ms,
        throughput=throughput,
    )


def measure_latency(
    run_one: Callable[[Any], Any],
    samples: Sequence[Any],
    warmup: int,
    repeats: int,
) -> LatencyStats:
    """Time ``run_one`` over ``samples``: discard warmup, average ``repeats`` passes."""
    if not samples:
        raise ValueError("samples must be non-empty")
    for i in range(warmup):
        run_one(samples[i % len(samples)])
    latencies_ms: list[float] = []
    for _ in range(repeats):
        for sample in samples:
            start = time.perf_counter()
            run_one(sample)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
    return latency_stats(latencies_ms)


def dir_size_mb(path: Path) -> float:
    """Total size of all files under ``path`` in megabytes."""
    total_bytes = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return total_bytes / (1024.0 * 1024.0)
