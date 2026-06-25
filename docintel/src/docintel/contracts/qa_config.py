"""Configuration for fine-tuning the CUAD extractive-QA model (build-time).

A plain data container (Pydantic model) holding the hyperparameters for QA
fine-tuning. Defaults are specified here rather than scattered as literals;
the Colab notebook may override individual fields.
"""

from __future__ import annotations

from pydantic import BaseModel


class QaTrainingConfig(BaseModel):
    """Hyperparameters for CUAD QA fine-tuning."""

    model_name: str = "microsoft/deberta-v3-base"
    num_train_epochs: float = 3.0
    learning_rate: float = 3e-5
    train_batch_size: int = 8
    eval_batch_size: int = 16
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    max_seq_length: int = 512
    doc_stride: int = 128
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
