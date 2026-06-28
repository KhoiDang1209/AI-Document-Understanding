"""CUAD extractive-QA fine-tuning helpers; run_qa_training executes on Colab GPU.

The pure helpers (training-args mapping, repro params, bundle writing) are
CPU-testable. ``run_qa_training`` assembles them with the Hugging Face Trainer
and runs the actual fine-tune — that executes on Colab GPU. Heavy libraries are
imported inside functions so this module loads cheaply on the laptop.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from docintel.contracts.qa_config import QaTrainingConfig


def resolve_mixed_precision(mode: str) -> tuple[bool, bool]:
    """Return (bf16, fp16) flags for the current GPU.

    A ``"bf16"`` request raises on GPUs without bf16 support rather than silently
    degrading to fp16: DeBERTa-v3's disentangled attention overflows in fp16 and
    produces NaN losses, so the fallback would quietly corrupt the whole run.
    """
    import torch

    if mode == "none" or not torch.cuda.is_available():
        return False, False
    if mode == "fp16":
        return False, True
    bf16_ok = torch.cuda.is_bf16_supported()  # bf16 needs Ampere+ (e.g. A100), not T4
    if not bf16_ok:
        raise RuntimeError(
            "mixed_precision='bf16' requested but this GPU lacks bf16 support. "
            "fp16 produces NaN losses with DeBERTa-v3; use a bf16-capable GPU "
            "(A100/L4) or set mixed_precision='none' (fp32)."
        )
    return True, False


def build_qa_training_arguments(config: QaTrainingConfig, output_dir: str) -> Any:
    """Map QaTrainingConfig to Hugging Face TrainingArguments."""
    from transformers import TrainingArguments

    bf16, fp16 = resolve_mixed_precision(config.mixed_precision)
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        seed=config.seed,
        eval_strategy=config.eval_strategy,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        bf16=bf16,
        fp16=fp16,
        logging_steps=50,
    )


def collect_repro_params(
    config: QaTrainingConfig, dataset_revision: str, git_sha: str
) -> dict[str, str]:
    """Flatten reproducibility-relevant values into MLflow string params."""
    return {
        "model_name": config.model_name,
        "num_train_epochs": str(config.num_train_epochs),
        "learning_rate": str(config.learning_rate),
        "train_batch_size": str(config.train_batch_size),
        "weight_decay": str(config.weight_decay),
        "warmup_ratio": str(config.warmup_ratio),
        "seed": str(config.seed),
        "max_seq_length": str(config.max_seq_length),
        "doc_stride": str(config.doc_stride),
        "mixed_precision": config.mixed_precision,
        "dataset_revision": dataset_revision,
        "git_sha": git_sha,
    }


def find_last_checkpoint(output_dir: str) -> str | None:
    """Return the newest checkpoint dir under output_dir, or None to start fresh.

    Lets a session that timed out resume on a fresh Colab runtime when output_dir
    points at persistent storage (Google Drive).
    """
    from transformers.trainer_utils import get_last_checkpoint

    if not Path(output_dir).is_dir():
        return None
    checkpoint: str | None = get_last_checkpoint(output_dir)  # type: ignore[no-untyped-call]
    return checkpoint


def save_qa_bundle(
    model: Any, tokenizer: Any, metrics: Mapping[str, float], bundle_dir: Path
) -> Path:
    """Write a self-contained model bundle for download + ONNX export.

    Non-finite metrics (NaN/inf from a diverged run) are written as JSON ``null``
    so ``metrics.json`` stays valid JSON rather than the non-standard ``NaN`` token.
    """
    bundle_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(bundle_dir / "model"))
    tokenizer.save_pretrained(str(bundle_dir / "tokenizer"))
    safe_metrics = {k: (v if math.isfinite(v) else None) for k, v in metrics.items()}
    (bundle_dir / "metrics.json").write_text(json.dumps(safe_metrics), encoding="utf-8")
    return bundle_dir


def run_qa_training(
    config: QaTrainingConfig,
    train_dataset: Any,
    eval_dataset: Any,
    tokenizer: Any,
    bundle_dir: Path,
    dataset_revision: str,
    git_sha: str,
    output_dir: str = "outputs",
) -> Path:
    """Fine-tune the QA model, log to MLflow, and save a bundle. Runs on Colab."""
    import mlflow
    from transformers import AutoModelForQuestionAnswering, Trainer, set_seed

    set_seed(config.seed)
    # Pin fp32 master weights: newer transformers can otherwise inherit the
    # checkpoint's dtype, loading DeBERTa-v3 in pure fp16 and yielding NaN losses.
    # bf16 (when enabled) is applied as autocast by the Trainer, not to the weights.
    model = AutoModelForQuestionAnswering.from_pretrained(config.model_name, dtype="float32")
    args = build_qa_training_arguments(config, output_dir)
    trainer = Trainer(
        model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset
    )
    last_checkpoint = find_last_checkpoint(output_dir)
    with mlflow.start_run():
        mlflow.log_params(collect_repro_params(config, dataset_revision, git_sha))
        trainer.train(resume_from_checkpoint=last_checkpoint)
        eval_metrics = trainer.evaluate()
        mlflow.log_metrics({k: v for k, v in eval_metrics.items() if isinstance(v, (int, float))})
        bundle = save_qa_bundle(model, tokenizer, eval_metrics, bundle_dir)
        mlflow.log_artifacts(str(bundle), artifact_path="bundle")
    return bundle
