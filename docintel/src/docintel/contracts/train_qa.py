"""CUAD extractive-QA fine-tuning helpers; run_qa_training executes on Colab GPU.

The pure helpers (training-args mapping, repro params, bundle writing) are
CPU-testable. ``run_qa_training`` assembles them with the Hugging Face Trainer
and runs the actual fine-tune — that executes on Colab GPU. Heavy libraries are
imported inside functions so this module loads cheaply on the laptop.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from docintel.contracts.qa_config import QaTrainingConfig


def build_qa_training_arguments(config: QaTrainingConfig, output_dir: str) -> Any:
    """Map QaTrainingConfig to Hugging Face TrainingArguments."""
    from transformers import TrainingArguments

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
        "dataset_revision": dataset_revision,
        "git_sha": git_sha,
    }


def save_qa_bundle(
    model: Any, tokenizer: Any, metrics: Mapping[str, float], bundle_dir: Path
) -> Path:
    """Write a self-contained model bundle for download + ONNX export."""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(bundle_dir / "model"))
    tokenizer.save_pretrained(str(bundle_dir / "tokenizer"))
    (bundle_dir / "metrics.json").write_text(json.dumps(dict(metrics)), encoding="utf-8")
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
    model = AutoModelForQuestionAnswering.from_pretrained(config.model_name)
    args = build_qa_training_arguments(config, output_dir)
    trainer = Trainer(
        model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset
    )
    with mlflow.start_run():
        mlflow.log_params(collect_repro_params(config, dataset_revision, git_sha))
        trainer.train()
        eval_metrics = trainer.evaluate()
        mlflow.log_metrics({k: v for k, v in eval_metrics.items() if isinstance(v, (int, float))})
        bundle = save_qa_bundle(model, tokenizer, eval_metrics, bundle_dir)
        mlflow.log_artifacts(str(bundle), artifact_path="bundle")
    return bundle
