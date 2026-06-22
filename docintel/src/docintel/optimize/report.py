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
