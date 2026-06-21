"""LayoutLMv3 fine-tuning builder and bundle saver.

The pure helpers (training-args mapping, repro params, bundle writing) are
CPU-testable. ``run_training`` assembles them with the Hugging Face Trainer and
runs the actual fine-tune — that executes on Colab GPU. Heavy libraries are
imported inside functions so this module loads cheaply on the laptop.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from docintel.kie.config import TrainingConfig


def build_training_arguments(config: TrainingConfig, output_dir: str) -> Any:
    """Map ``TrainingConfig`` to Hugging Face ``TrainingArguments``."""
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
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
    )


def collect_repro_params(
    config: TrainingConfig,
    dataset_revision: str,
    git_sha: str,
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
        "dataset_revision": dataset_revision,
        "git_sha": git_sha,
    }


def save_bundle(
    model: Any,
    processor: Any,
    id2label: Mapping[int, str],
    metrics: Mapping[str, float],
    bundle_dir: Path,
) -> Path:
    """Write a self-contained model bundle for download + import."""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(bundle_dir / "model"))
    processor.save_pretrained(str(bundle_dir / "processor"))
    (bundle_dir / "label_map.json").write_text(
        json.dumps({str(k): v for k, v in id2label.items()}), encoding="utf-8"
    )
    (bundle_dir / "metrics.json").write_text(json.dumps(dict(metrics)), encoding="utf-8")
    return bundle_dir


def run_training(
    config: TrainingConfig,
    train_dataset: Any,
    eval_dataset: Any,
    id2label: Mapping[int, str],
    label2id: Mapping[str, int],
    processor: Any,
    bundle_dir: Path,
    dataset_revision: str,
    git_sha: str,
    output_dir: str = "outputs",
) -> Path:
    """Fine-tune LayoutLMv3, log to MLflow, and save a bundle. Runs on Colab.

    Assumes ``mlflow.set_tracking_uri`` has been pointed at the Colab file-store
    by the caller (the notebook). Returns the bundle directory.
    """
    import mlflow  # type: ignore[import-not-found]
    from transformers import (
        AutoModelForTokenClassification,
        Trainer,
        set_seed,
    )

    from docintel.kie.metrics import compute_seqeval_metrics

    set_seed(config.seed)
    model = AutoModelForTokenClassification.from_pretrained(
        config.model_name,
        num_labels=len(id2label),
        id2label=dict(id2label),
        label2id=dict(label2id),
    )
    args = build_training_arguments(config, output_dir)

    def _metrics(eval_pred: Any) -> dict[str, float]:
        predictions, labels = eval_pred
        return compute_seqeval_metrics(predictions, labels, id2label)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=_metrics,
    )

    with mlflow.start_run():
        mlflow.log_params(collect_repro_params(config, dataset_revision, git_sha))
        trainer.train()
        eval_metrics = trainer.evaluate()
        mlflow.log_metrics({k: v for k, v in eval_metrics.items() if isinstance(v, (int, float))})
        bundle = save_bundle(model, processor, id2label, eval_metrics, bundle_dir)
        mlflow.log_artifacts(str(bundle), artifact_path="bundle")
    return bundle
