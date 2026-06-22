# Phase 3 — Optimization (ONNX + INT8 + Benchmark) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export the registered `cord-layoutlmv3` model to ONNX, dynamic-INT8 quantize it, and benchmark PyTorch fp32 vs ONNX fp32 vs ONNX INT8 (F1 / latency / throughput / size) on the laptop CPU, producing a reproducible `docs/benchmark.md`, an MLflow run, and a registered `cord-layoutlmv3-onnx-int8` for Phase 4.

**Architecture:** A new `src/docintel/optimize/` package with one responsibility per file: `config` (BenchmarkConfig), `benchmark` (pure timing harness), `report` (markdown + plots), `evaluate` (F1 reusing `kie/metrics`), `export`/`quantize` (Optimum + ONNX Runtime), and `run_benchmark` (CLI orchestrator). Heavy libraries are imported **inside functions** so modules load cheaply and pure helpers stay unit-testable, mirroring `kie/train.py` from Phase 2.

**Tech Stack:** Python 3.12, Hugging Face Optimum + ONNX Runtime, transformers, datasets, seqeval, matplotlib, MLflow, pytest, ruff, mypy.

## Global Constraints

- Python `>=3.12`; full type hints; mypy `strict` must pass over `src`.
- Prefer functional components over classes; frozen dataclasses (`BenchmarkConfig`, `LatencyStats`, `ConfigResult`) are allowed as plain data containers.
- No hardcoded constants — knobs live in `BenchmarkConfig` / `Settings`.
- Source model name from `Settings.kie_registered_model_name` (`cord-layoutlmv3`); served INT8 model registered as `cord-layoutlmv3-onnx-int8` via `Settings.kie_onnx_registered_model_name`.
- Export/quantize via **Optimum + ONNX Runtime**; quantization is **dynamic INT8** (no calibration).
- Accuracy metric is **F1** (reuse `kie/metrics.py` seqeval); CER/WER are excluded (OCR-only).
- Evaluation uses the CORD **`test`** split, identical across all three configs.
- Compare three configs: **PyTorch fp32**, **ONNX fp32**, **ONNX INT8**.
- Minimal changes: touch only what the task needs; reuse `kie/dataset.py`, `kie/labels.py`, `kie/metrics.py` — do not duplicate parsing/metric logic.
- Heavy libs (optimum, onnxruntime, transformers, datasets, matplotlib, mlflow) imported **inside functions**.
- Every commit message ends with: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Environment: run `uv sync --extra dev --extra train --extra optimize` (the `dev` extra keeps pytest/ruff/mypy inside `.venv` rather than falling through to the Anaconda base — a known repo gotcha). Verify with `uv run python -c "import shutil; print(shutil.which('pytest'))"` pointing inside `.venv`.

## File Structure

| File | Responsibility |
|---|---|
| `src/docintel/optimize/__init__.py` | Package marker (empty). |
| `src/docintel/optimize/config.py` | `BenchmarkConfig` frozen dataclass + `from_settings` + `with_overrides`. |
| `src/docintel/optimize/benchmark.py` | `LatencyStats`, `latency_stats`, `measure_latency`, `dir_size_mb`. |
| `src/docintel/optimize/report.py` | `ConfigResult`, `render_markdown_report`, `build_report_markdown`, `render_plots`, `write_report`. |
| `src/docintel/optimize/evaluate.py` | `collect_predictions`, `evaluate_model` (reuse `kie/metrics`). |
| `src/docintel/optimize/export.py` | `resolve_model_uri`, `download_registered_model`, `export_to_onnx`. |
| `src/docintel/optimize/quantize.py` | `quantize_dynamic_int8`. |
| `src/docintel/optimize/run_benchmark.py` | `slugify`, `flatten_results_for_mlflow`, `main` (CLI `docintel-benchmark-kie`). |
| `src/docintel/config.py` | (modify) add `kie_onnx_registered_model_name`. |
| `pyproject.toml` | (modify) add `optimize` extra, mypy overrides, script entry. |

## Controller Pre-Flight (before Task 1 — de-risk the toolchain) — ✅ DONE

**Result (2026-06-22):** The risk was real. Installing Optimum 2.1.0 forces transformers down
to 4.57.6 (`optimum` requires `transformers<5`). With `transformers>=4.40,<5` pinned in the
`optimize` extra, both the export and the dynamic-INT8 quantize succeed:
- `ORTModelForTokenClassification.from_pretrained('microsoft/layoutlmv3-base', export=True)` → `model.onnx`.
- `ORTQuantizer` + `AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)` → `model_quantized.onnx`.
Resolved versions: optimum 2.1.0, onnxruntime 1.27.0, onnx 1.22.0, transformers 4.57.6.
Task 1's extra includes the pin; no fallback to Approach B is needed.

---

The spec's key risk is **Optimum × transformers 5.x × LayoutLMv3 export**. Before dispatching Task 1, the controller runs a real smoke-export to confirm the stack works:

```bash
uv sync --extra dev --extra train --extra optimize
uv run python -c "
from optimum.onnxruntime import ORTModelForTokenClassification
m = ORTModelForTokenClassification.from_pretrained('microsoft/layoutlmv3-base', export=True)
print('ONNX export OK:', type(m).__name__)
"
```
Expected: prints `ONNX export OK: ORTModelForTokenClassification` (downloads base weights once). **If it fails** (e.g. Optimum rejects transformers 5.x or lacks a LayoutLMv3 ONNX config): pin transformers for the optimize path or fall back to raw `torch.onnx.export` for `export_to_onnx` only — record the decision in the SDD ledger before proceeding. The pure-helper tasks (1–4, 6 seams) are unaffected either way.

---

## Task 1: Package scaffold — deps, settings, BenchmarkConfig

**Files:**
- Create: `src/docintel/optimize/__init__.py`
- Create: `src/docintel/optimize/config.py`
- Create: `tests/test_optimize_config.py`
- Modify: `src/docintel/config.py` (add one setting)
- Modify: `pyproject.toml` (add `optimize` extra + mypy overrides)

**Interfaces:**
- Consumes: `docintel.config.Settings`.
- Produces:
  - `Settings.kie_onnx_registered_model_name: str = "cord-layoutlmv3-onnx-int8"`
  - `BenchmarkConfig(source_model_name: str, source_model_version: str = "1", onnx_registered_model_name: str = "cord-layoutlmv3-onnx-int8", sample_size: int = 50, warmup_runs: int = 3, repeats: int = 5, num_threads: int = 4, quant_type: str = "dynamic-int8", seed: int = 42)`
  - `BenchmarkConfig.from_settings(settings: Settings) -> BenchmarkConfig`
  - `BenchmarkConfig.with_overrides(**changes: Any) -> BenchmarkConfig`

- [ ] **Step 1: Add the `optimize` extra and mypy overrides to `pyproject.toml`**

In `[project.optional-dependencies]`, after the `kie` extra, add:
```toml
optimize = [
    "optimum[onnxruntime]>=1.20",
    "onnx>=1.16",
    "matplotlib>=3.8",
    "transformers>=4.40,<5",
]
```
The `transformers>=4.40,<5` pin is **required**: Optimum (2.1.0) is incompatible with
transformers 5.x and will not export otherwise — confirmed by the controller pre-flight.
Training runs on Colab (transformers 5.x); optimization runs on the laptop (4.x); they are
separate environments, so this pin does not affect Phase 2.
Extend the existing mypy overrides `module` list (currently ends `"seqeval.*", "mlflow.*"`) to also include `"optimum.*", "onnx.*", "onnxruntime.*", "matplotlib.*"`:
```toml
module = ["datasets.*", "huggingface_hub.*", "cv2.*", "doctr.*", "PIL.*", "seqeval.*", "mlflow.*", "optimum.*", "onnx.*", "onnxruntime.*", "matplotlib.*"]
```

- [ ] **Step 2: Add the setting**

In `src/docintel/config.py`, under the `# KIE (Phase 2)` block, add a line after `kie_registered_model_name`:
```python
    kie_registered_model_name: str = "cord-layoutlmv3"
    kie_onnx_registered_model_name: str = "cord-layoutlmv3-onnx-int8"
```

- [ ] **Step 3: Write the failing test**

Create `tests/test_optimize_config.py`:
```python
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
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `uv run pytest tests/test_optimize_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'docintel.optimize'`.

- [ ] **Step 5: Create the package and config**

Create empty `src/docintel/optimize/__init__.py`.

Create `src/docintel/optimize/config.py`:
```python
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
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `uv run pytest tests/test_optimize_config.py -v`
Expected: 3 passed.

- [ ] **Step 7: Lint + type-check**

Run: `uv run ruff check src/docintel/optimize/ tests/test_optimize_config.py && uv run mypy src/docintel/optimize/config.py src/docintel/config.py`
Expected: all checks pass; no mypy issues.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/docintel/config.py src/docintel/optimize/__init__.py src/docintel/optimize/config.py tests/test_optimize_config.py
git commit -m "feat(optimize): scaffold optimize package, deps, and BenchmarkConfig

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Pure timing harness (`benchmark.py`)

**Files:**
- Create: `src/docintel/optimize/benchmark.py`
- Create: `tests/test_optimize_benchmark.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) LatencyStats(p50_ms: float, p95_ms: float, mean_ms: float, throughput: float)`
  - `latency_stats(latencies_ms: Sequence[float]) -> LatencyStats` (nearest-rank percentile; raises `ValueError` on empty)
  - `measure_latency(run_one: Callable[[Any], Any], samples: Sequence[Any], warmup: int, repeats: int) -> LatencyStats` (calls `run_one` `warmup + repeats*len(samples)` times; raises `ValueError` on empty samples)
  - `dir_size_mb(path: Path) -> float`

- [ ] **Step 1: Write the failing test**

Create `tests/test_optimize_benchmark.py`:
```python
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


def test_dir_size_mb_sums_files(tmp_path: Path) -> None:
    (tmp_path / "a.bin").write_bytes(b"x" * (1024 * 1024))
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"y" * (1024 * 1024))
    assert dir_size_mb(tmp_path) == pytest.approx(2.0, abs=0.01)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_optimize_benchmark.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'docintel.optimize.benchmark'`.

- [ ] **Step 3: Implement `benchmark.py`**

Create `src/docintel/optimize/benchmark.py`:
```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_optimize_benchmark.py -v`
Expected: 5 passed.

- [ ] **Step 5: Lint + type-check**

Run: `uv run ruff check src/docintel/optimize/benchmark.py tests/test_optimize_benchmark.py && uv run mypy src/docintel/optimize/benchmark.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/docintel/optimize/benchmark.py tests/test_optimize_benchmark.py
git commit -m "feat(optimize): add pure latency/throughput timing harness

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Report rendering (`report.py`)

**Files:**
- Create: `src/docintel/optimize/report.py`
- Create: `tests/test_optimize_report.py`

**Interfaces:**
- Consumes: `LatencyStats` from `docintel.optimize.benchmark`.
- Produces:
  - `@dataclass(frozen=True) ConfigResult(name: str, f1: float, precision: float, recall: float, accuracy: float, latency: LatencyStats, size_mb: float)`
  - `render_markdown_report(results: Sequence[ConfigResult]) -> str` (the comparison table)
  - `build_report_markdown(results: Sequence[ConfigResult], plot_paths: Sequence[Path]) -> str` (title + table + plot image links)
  - `render_plots(results: Sequence[ConfigResult], out_dir: Path) -> list[Path]` (heavy: matplotlib inside)
  - `write_report(markdown: str, out_path: Path) -> Path`

- [ ] **Step 1: Write the failing test**

Create `tests/test_optimize_report.py`:
```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_optimize_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'docintel.optimize.report'`.

- [ ] **Step 3: Implement `report.py`**

Create `src/docintel/optimize/report.py`:
```python
"""Render the benchmark comparison into markdown tables and plots."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from docintel.optimize.benchmark import LatencyStats

_COLUMNS = [
    "Config",
    "F1",
    "Precision",
    "Recall",
    "Accuracy",
    "p50 (ms)",
    "p95 (ms)",
    "Throughput (doc/s)",
    "Size (MB)",
]


@dataclass(frozen=True)
class ConfigResult:
    """One benchmarked configuration: accuracy + latency + artifact size."""

    name: str
    f1: float
    precision: float
    recall: float
    accuracy: float
    latency: LatencyStats
    size_mb: float


def render_markdown_report(results: Sequence[ConfigResult]) -> str:
    """Render the comparison table as a GitHub-flavored markdown table."""
    header = "| " + " | ".join(_COLUMNS) + " |"
    separator = "|" + "---|" * len(_COLUMNS)
    rows = [header, separator]
    for r in results:
        rows.append(
            f"| {r.name} | {r.f1:.4f} | {r.precision:.4f} | {r.recall:.4f} | "
            f"{r.accuracy:.4f} | {r.latency.p50_ms:.1f} | {r.latency.p95_ms:.1f} | "
            f"{r.latency.throughput:.2f} | {r.size_mb:.1f} |"
        )
    return "\n".join(rows)


def build_report_markdown(
    results: Sequence[ConfigResult],
    plot_paths: Sequence[Path],
) -> str:
    """Assemble the full benchmark document: title, table, and plot links."""
    parts = [
        "# DocIntel KIE Benchmark — LayoutLMv3 (fp32 vs ONNX vs INT8)",
        "",
        render_markdown_report(results),
        "",
    ]
    for plot in plot_paths:
        parts.append(f"![{plot.stem}]({plot.name})")
    return "\n".join(parts) + "\n"


def render_plots(results: Sequence[ConfigResult], out_dir: Path) -> list[Path]:
    """Write bar charts (latency p95, F1, size) as PNGs; return their paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    names = [r.name for r in results]
    specs = {
        "latency_p95": [r.latency.p95_ms for r in results],
        "f1": [r.f1 for r in results],
        "size_mb": [r.size_mb for r in results],
    }
    paths: list[Path] = []
    for key, values in specs.items():
        fig, ax = plt.subplots()
        ax.bar(names, values)
        ax.set_title(key)
        fig.tight_layout()
        path = out_dir / f"benchmark_{key}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def write_report(markdown: str, out_path: Path) -> Path:
    """Write the markdown report to disk and return the path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    return out_path
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_optimize_report.py -v`
Expected: 3 passed.

- [ ] **Step 5: Lint + type-check**

Run: `uv run ruff check src/docintel/optimize/report.py tests/test_optimize_report.py && uv run mypy src/docintel/optimize/report.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/docintel/optimize/report.py tests/test_optimize_report.py
git commit -m "feat(optimize): add benchmark report tables and plots

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Model-agnostic evaluation (`evaluate.py`)

**Files:**
- Create: `src/docintel/optimize/evaluate.py`
- Create: `tests/test_optimize_evaluate.py`

**Interfaces:**
- Consumes: `docintel.kie.metrics.compute_seqeval_metrics(predictions, label_ids, id2label) -> dict[str, float]`.
- Produces:
  - `collect_predictions(run_logits: Callable[[Mapping[str, Any]], Any], encoded: Sequence[Mapping[str, Any]]) -> tuple[Any, Any]` — stacks per-example logits `(n, seq, num_labels)` and label ids `(n, seq)`; each encoded example must carry a `"labels"` key.
  - `evaluate_model(run_logits, encoded, id2label: Mapping[int, str]) -> dict[str, float]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_optimize_evaluate.py`:
```python
"""Tests for model-agnostic F1 evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from docintel.optimize.evaluate import evaluate_model


def test_evaluate_model_perfect_predictions_scores_f1_one() -> None:
    # Two examples; labels use -100 to mark ignored subword positions.
    encoded = [
        {"labels": [0, 1, 1]},
        {"labels": [0, 1, -100]},
    ]
    id2label = {0: "O", 1: "B-menu.nm"}

    def run_logits(sample: Mapping[str, Any]) -> list[list[float]]:
        # Emit logits whose argmax equals the (clamped) label at each position.
        logits: list[list[float]] = []
        for label in sample["labels"]:
            target = 0 if label in (0, -100) else 1
            row = [0.0, 0.0]
            row[target] = 9.0
            logits.append(row)
        return logits

    metrics = evaluate_model(run_logits, encoded, id2label)
    assert metrics["f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_optimize_evaluate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'docintel.optimize.evaluate'`.

- [ ] **Step 3: Implement `evaluate.py`**

Create `src/docintel/optimize/evaluate.py`:
```python
"""Evaluate any model (PyTorch or ONNX) over encoded examples and score F1.

The model is injected as ``run_logits`` — a callable mapping one encoded
example to its per-token logits — so the same evaluation drives every config.
Metric computation is delegated to ``kie.metrics`` (no duplication).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from docintel.kie.metrics import compute_seqeval_metrics


def collect_predictions(
    run_logits: Callable[[Mapping[str, Any]], Any],
    encoded: Sequence[Mapping[str, Any]],
) -> tuple[Any, Any]:
    """Run ``run_logits`` over each example; stack logits and label ids."""
    predictions: list[Any] = []
    label_ids: list[Any] = []
    for sample in encoded:
        logits = np.asarray(run_logits(sample), dtype=np.float32)
        predictions.append(logits)
        label_ids.append(np.asarray(sample["labels"]))
    return np.stack(predictions), np.stack(label_ids)


def evaluate_model(
    run_logits: Callable[[Mapping[str, Any]], Any],
    encoded: Sequence[Mapping[str, Any]],
    id2label: Mapping[int, str],
) -> dict[str, float]:
    """Compute seqeval F1 (+ per-field) for one model over encoded examples."""
    predictions, label_ids = collect_predictions(run_logits, encoded)
    return compute_seqeval_metrics(predictions, label_ids, id2label)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_optimize_evaluate.py -v`
Expected: 1 passed.

- [ ] **Step 5: Lint + type-check**

Run: `uv run ruff check src/docintel/optimize/evaluate.py tests/test_optimize_evaluate.py && uv run mypy src/docintel/optimize/evaluate.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/docintel/optimize/evaluate.py tests/test_optimize_evaluate.py
git commit -m "feat(optimize): add model-agnostic F1 evaluation reusing kie.metrics

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Export + quantize (`export.py`, `quantize.py`)

These functions are heavy (Optimum, ONNX Runtime, MLflow) and are imported inside their function bodies. Only the pure seam (`resolve_model_uri`) is unit-tested in CI; the full export/quantize path is exercised by the controller's final integration run (see end of plan), mirroring how Phase 2's `run_training` was integration-verified rather than unit-tested.

**Files:**
- Create: `src/docintel/optimize/export.py`
- Create: `src/docintel/optimize/quantize.py`
- Create: `tests/test_optimize_export.py`

**Interfaces:**
- Produces:
  - `resolve_model_uri(name: str, version: str) -> str` → `f"models:/{name}/{version}"`
  - `download_registered_model(name: str, version: str, dest: Path, tracking_uri: str | None = None) -> Path` → returns the local **bundle root** (containing `model/`, `processor/`, `label_map.json`).
  - `export_to_onnx(model_dir: Path, out_dir: Path) -> Path` → returns `out_dir` (contains the ONNX fp32 model).
  - `quantize_dynamic_int8(onnx_dir: Path, out_dir: Path) -> Path` → returns `out_dir` (contains the INT8 model).

- [ ] **Step 1: Write the failing test (pure seam only)**

Create `tests/test_optimize_export.py`:
```python
"""Tests for the pure seam in optimize.export."""

from __future__ import annotations

from docintel.optimize.export import resolve_model_uri


def test_resolve_model_uri_builds_registry_uri() -> None:
    assert resolve_model_uri("cord-layoutlmv3", "1") == "models:/cord-layoutlmv3/1"
    assert resolve_model_uri("m", "3") == "models:/m/3"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_optimize_export.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'docintel.optimize.export'`.

- [ ] **Step 3: Implement `export.py`**

Create `src/docintel/optimize/export.py`:
```python
"""Pull the registered model from MLflow and export it to ONNX (fp32).

Heavy libraries (mlflow, optimum) are imported inside the functions so the
module loads cheaply and the pure ``resolve_model_uri`` seam stays testable.
"""

from __future__ import annotations

from pathlib import Path


def resolve_model_uri(name: str, version: str) -> str:
    """MLflow registry URI for a registered model version."""
    return f"models:/{name}/{version}"


def download_registered_model(
    name: str,
    version: str,
    dest: Path,
    tracking_uri: str | None = None,
) -> Path:
    """Download a registered model's bundle artifacts; return the bundle root."""
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    dest.mkdir(parents=True, exist_ok=True)
    local = mlflow.artifacts.download_artifacts(
        artifact_uri=resolve_model_uri(name, version),
        dst_path=str(dest),
    )
    return Path(local)


def export_to_onnx(model_dir: Path, out_dir: Path) -> Path:
    """Export a LayoutLMv3 token-classification model to ONNX (fp32) via Optimum."""
    from optimum.onnxruntime import ORTModelForTokenClassification

    model = ORTModelForTokenClassification.from_pretrained(str(model_dir), export=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    return out_dir
```

- [ ] **Step 4: Implement `quantize.py`**

Create `src/docintel/optimize/quantize.py`:
```python
"""Dynamic-INT8 quantization of an exported ONNX model via Optimum."""

from __future__ import annotations

from pathlib import Path


def quantize_dynamic_int8(onnx_dir: Path, out_dir: Path) -> Path:
    """Dynamic-INT8 quantize the ONNX model in ``onnx_dir`` into ``out_dir``."""
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    out_dir.mkdir(parents=True, exist_ok=True)
    quantizer = ORTQuantizer.from_pretrained(str(onnx_dir))
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer.quantize(quantization_config=qconfig, save_dir=str(out_dir))
    return out_dir
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_optimize_export.py -v`
Expected: 1 passed.

- [ ] **Step 6: Lint + type-check**

Run: `uv run ruff check src/docintel/optimize/export.py src/docintel/optimize/quantize.py tests/test_optimize_export.py && uv run mypy src/docintel/optimize/export.py src/docintel/optimize/quantize.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/docintel/optimize/export.py src/docintel/optimize/quantize.py tests/test_optimize_export.py
git commit -m "feat(optimize): add ONNX export and dynamic-INT8 quantization

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: CLI orchestrator (`run_benchmark.py`)

The `main` orchestrator is heavy (it downloads the model, loads CORD via `datasets`, runs three configs through ONNX Runtime / PyTorch, and logs to MLflow). Its pure seams (`slugify`, `flatten_results_for_mlflow`) are unit-tested in CI; `main` is integration-verified by the controller's final run.

**Files:**
- Create: `src/docintel/optimize/run_benchmark.py`
- Create: `tests/test_optimize_run_benchmark.py`
- Modify: `pyproject.toml` (`[project.scripts]` — add `docintel-benchmark-kie`)

**Interfaces:**
- Consumes: `BenchmarkConfig`, `export`/`quantize`/`evaluate`/`benchmark`/`report` modules, `docintel.kie.dataset.parse_cord_example`/`encode_example`, `docintel.kie.labels.build_label_maps`, `docintel.config.get_settings`.
- Produces:
  - `slugify(name: str) -> str` (`"onnx-int8"` → `"onnx_int8"`)
  - `flatten_results_for_mlflow(results: Sequence[ConfigResult]) -> dict[str, float]`
  - `main() -> None` (CLI entry `docintel-benchmark-kie`)

- [ ] **Step 1: Add the script entry to `pyproject.toml`**

Under `[project.scripts]`, after `docintel-import-kie`, add:
```toml
docintel-benchmark-kie = "docintel.optimize.run_benchmark:main"
```

- [ ] **Step 2: Write the failing test (pure seams)**

Create `tests/test_optimize_run_benchmark.py`:
```python
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
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_optimize_run_benchmark.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'docintel.optimize.run_benchmark'`.

- [ ] **Step 4: Implement `run_benchmark.py`**

Create `src/docintel/optimize/run_benchmark.py`. The pure seams are at module top; `main` imports heavy libs inside.
```python
"""CLI: export -> quantize -> evaluate -> benchmark -> report -> MLflow.

Runs entirely on the laptop CPU. Pure seams (slugify, flatten) are testable;
``main`` wires the heavy steps together and is run by hand on the laptop.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from docintel.optimize.report import ConfigResult

logger = logging.getLogger("docintel.optimize.benchmark")


def slugify(name: str) -> str:
    """Make a config name safe for an MLflow metric key."""
    return name.replace("-", "_").replace(" ", "_")


def flatten_results_for_mlflow(results: Sequence[ConfigResult]) -> dict[str, float]:
    """Flatten per-config metrics into a single ``{metric_config: value}`` map."""
    flat: dict[str, float] = {}
    for r in results:
        suffix = slugify(r.name)
        flat[f"f1_{suffix}"] = r.f1
        flat[f"precision_{suffix}"] = r.precision
        flat[f"recall_{suffix}"] = r.recall
        flat[f"accuracy_{suffix}"] = r.accuracy
        flat[f"p50_ms_{suffix}"] = r.latency.p50_ms
        flat[f"p95_ms_{suffix}"] = r.latency.p95_ms
        flat[f"mean_ms_{suffix}"] = r.latency.mean_ms
        flat[f"throughput_{suffix}"] = r.latency.throughput
        flat[f"size_mb_{suffix}"] = r.size_mb
    return flat


def _build_encoded_samples(
    bundle_root: Path,
    sample_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[int, str], dict[str, int]]:
    """Load the CORD test split and encode ``sample_size`` examples."""
    import numpy as np
    from datasets import load_dataset
    from transformers import LayoutLMv3Processor

    from docintel.kie.dataset import encode_example, parse_cord_example
    from docintel.kie.labels import build_label_maps

    label_list_map: dict[str, str] = json.loads(
        (bundle_root / "label_map.json").read_text(encoding="utf-8")
    )
    id2label = {int(k): v for k, v in label_list_map.items()}
    label_list = [id2label[i] for i in range(len(id2label))]
    _id2label, label2id = build_label_maps(label_list)

    processor = LayoutLMv3Processor.from_pretrained(str(bundle_root / "processor"), apply_ocr=False)
    dataset = load_dataset("naver-clova-ix/cord-v2", split="test")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset))[:sample_size]

    encoded: list[dict[str, Any]] = []
    for idx in indices:
        example = dataset[int(idx)]
        words, boxes, bio = parse_cord_example(json.loads(example["ground_truth"]))
        if not words:
            continue
        enc = encode_example(example["image"], words, boxes, bio, processor, label2id)
        enc["pixel_values"] = enc["pixel_values"][0]
        encoded.append({k: enc[k] for k in ("input_ids", "attention_mask", "bbox", "pixel_values", "labels")})
    return encoded, id2label, label2id


def main() -> None:
    """CLI entry point for ``docintel-benchmark-kie``."""
    import io
    import sys

    # MLflow prints run URLs with an emoji; force UTF-8 on non-UTF-8 consoles.
    if isinstance(sys.stdout, io.TextIOWrapper) and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    import mlflow
    import torch
    from optimum.onnxruntime import ORTModelForTokenClassification
    from transformers import AutoModelForTokenClassification

    from docintel.config import get_settings
    from docintel.logging_config import configure_logging
    from docintel.optimize.benchmark import dir_size_mb, measure_latency
    from docintel.optimize.config import BenchmarkConfig
    from docintel.optimize.evaluate import evaluate_model
    from docintel.optimize.export import download_registered_model, export_to_onnx
    from docintel.optimize.quantize import quantize_dynamic_int8
    from docintel.optimize.report import (
        ConfigResult,
        build_report_markdown,
        render_plots,
        write_report,
    )

    settings = get_settings()
    configure_logging(settings.log_level)

    parser = argparse.ArgumentParser(description="Export, quantize, and benchmark the KIE model.")
    parser.add_argument("--work-dir", type=Path, default=Path("artifacts/phase3"))
    parser.add_argument("--report-path", type=Path, default=Path("docs/benchmark.md"))
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--tracking-uri", type=str, default=None)
    args = parser.parse_args()

    config = BenchmarkConfig.from_settings(settings)
    if args.sample_size is not None:
        config = config.with_overrides(sample_size=args.sample_size)
    torch.set_num_threads(config.num_threads)
    tracking_uri = args.tracking_uri or settings.mlflow_tracking_uri

    work_dir: Path = args.work_dir
    bundle_root = download_registered_model(
        config.source_model_name, config.source_model_version, work_dir / "bundle", tracking_uri
    )
    onnx_fp32 = export_to_onnx(bundle_root / "model", work_dir / "onnx-fp32")
    onnx_int8 = quantize_dynamic_int8(onnx_fp32, work_dir / "onnx-int8")

    encoded, id2label, _ = _build_encoded_samples(bundle_root, config.sample_size, config.seed)

    torch_model = AutoModelForTokenClassification.from_pretrained(str(bundle_root / "model"))
    torch_model.eval()
    ort_fp32 = ORTModelForTokenClassification.from_pretrained(str(onnx_fp32))
    ort_int8 = ORTModelForTokenClassification.from_pretrained(str(onnx_int8), file_name="model_quantized.onnx")

    def _torch_logits(sample: dict[str, Any]) -> Any:
        inputs = {k: torch.tensor([sample[k]]) for k in ("input_ids", "attention_mask", "bbox")}
        inputs["pixel_values"] = torch.tensor([sample["pixel_values"]])
        with torch.no_grad():
            return torch_model(**inputs).logits[0].numpy()

    def _ort_logits_factory(ort_model: Any) -> Any:
        def _run(sample: dict[str, Any]) -> Any:
            inputs = {k: torch.tensor([sample[k]]) for k in ("input_ids", "attention_mask", "bbox")}
            inputs["pixel_values"] = torch.tensor([sample["pixel_values"]])
            return ort_model(**inputs).logits[0].numpy()

        return _run

    configs = [
        ("torch-fp32", _torch_logits, bundle_root / "model"),
        ("onnx-fp32", _ort_logits_factory(ort_fp32), onnx_fp32),
        ("onnx-int8", _ort_logits_factory(ort_int8), onnx_int8),
    ]

    results: list[ConfigResult] = []
    for name, run_logits, artifact_dir in configs:
        metrics = evaluate_model(run_logits, encoded, id2label)
        latency = measure_latency(run_logits, encoded, config.warmup_runs, config.repeats)
        results.append(
            ConfigResult(
                name=name,
                f1=metrics["f1"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                accuracy=metrics["accuracy"],
                latency=latency,
                size_mb=dir_size_mb(artifact_dir),
            )
        )
        logger.info("optimize.config.done", extra={"config": name, "f1": metrics["f1"]})

    plot_paths = render_plots(results, args.report_path.parent)
    write_report(build_report_markdown(results, plot_paths), args.report_path)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("cord-kie-benchmark")
    with mlflow.start_run() as run:
        mlflow.log_param("source_model", f"{config.source_model_name}/{config.source_model_version}")
        mlflow.log_param("sample_size", str(config.sample_size))
        mlflow.log_param("quant_type", config.quant_type)
        mlflow.log_metrics(flatten_results_for_mlflow(results))
        mlflow.log_artifact(str(args.report_path))
        for plot in plot_paths:
            mlflow.log_artifact(str(plot))
        mlflow.log_artifacts(str(onnx_int8), artifact_path="onnx-int8")
        mlflow.register_model(
            f"runs:/{run.info.run_id}/onnx-int8", config.onnx_registered_model_name
        )
    logger.info("optimize.benchmark.done", extra={"report": str(args.report_path)})


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_optimize_run_benchmark.py -v`
Expected: 2 passed.

- [ ] **Step 6: Lint + type-check**

Run: `uv run ruff check src/docintel/optimize/run_benchmark.py tests/test_optimize_run_benchmark.py && uv run mypy src/docintel/optimize/run_benchmark.py`
Expected: clean. (If mypy flags `Any`-typed heavy returns, that is acceptable — the heavy libs are in the mypy override list.)

- [ ] **Step 7: Run the full suite to confirm no regressions**

Run: `uv run pytest -q`
Expected: all prior tests + the new optimize tests pass; the slow OCR test stays deselected.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/docintel/optimize/run_benchmark.py tests/test_optimize_run_benchmark.py
git commit -m "feat(optimize): add docintel-benchmark-kie CLI orchestrator

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Controller Final Integration (after Task 6 — produces the deliverable)

The unit tests cover the pure logic; the actual benchmark is the deliverable and is run by the controller on the laptop (analogous to Phase 2's Colab run). Steps:

1. Ensure services: `docker compose up -d mlflow minio` (from `docintel/`).
2. Run end-to-end (override tracking URI for the host; a small sample first to smoke it):
   ```bash
   DOCINTEL_MLFLOW_TRACKING_URI=http://localhost:5000 PYTHONUTF8=1 \
     uv run docintel-benchmark-kie --sample-size 8
   ```
   Then the full run (default sample size) once the smoke run is clean.
3. Verify: `docs/benchmark.md` exists with the 3-config table + plots; an MLflow run under experiment `cord-kie-benchmark` has the flattened metrics + artifacts; `cord-layoutlmv3-onnx-int8` is registered.
4. If `ORTModelForTokenClassification.from_pretrained(..., file_name="model_quantized.onnx")` fails, list the quantized dir to find the real ONNX filename and adjust (Optimum's dynamic output is conventionally `model_quantized.onnx`).
5. Commit `docs/benchmark.md` (and any small plot PNGs) with a `docs(phase-3): add benchmark report` message.

This integration run is **not** a subagent task; the controller performs it and records results in the SDD ledger.

---

## Done When

- `docs/benchmark.md` compares PyTorch fp32 vs ONNX fp32 vs ONNX INT8 on F1 / latency / throughput / size, with tables and plots.
- An MLflow run under `cord-kie-benchmark` records the flattened metrics + report artifacts.
- `cord-layoutlmv3-onnx-int8` is registered in the MLflow model registry for Phase 4.
- ruff, mypy-strict, and the full pytest suite are green.
