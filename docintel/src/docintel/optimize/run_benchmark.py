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
        keys = ("input_ids", "attention_mask", "bbox", "pixel_values", "labels")
        encoded.append({k: enc[k] for k in keys})
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
    ort_int8 = ORTModelForTokenClassification.from_pretrained(
        str(onnx_int8), file_name="model_quantized.onnx"
    )

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
        source_model = f"{config.source_model_name}/{config.source_model_version}"
        mlflow.log_param("source_model", source_model)
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
