"""Training hyperparameters for KIE fine-tuning.

A plain data container (not service behavior) holding the knobs for a
LayoutLMv3 fine-tune. Defaults are named here rather than scattered as
literals; the Colab notebook may override individual fields.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from docintel.config import Settings


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for one LayoutLMv3 fine-tuning run."""

    model_name: str
    num_train_epochs: float = 4.0
    learning_rate: float = 1e-5
    train_batch_size: int = 2
    eval_batch_size: int = 2
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    max_seq_length: int = 512
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"

    @classmethod
    def from_settings(cls, settings: Settings) -> TrainingConfig:
        """Build a config whose model name comes from service settings."""
        return cls(model_name=settings.kie_model_name)

    def with_overrides(self, **changes: Any) -> TrainingConfig:
        """Return a copy with the given fields replaced (notebook convenience)."""
        return replace(self, **changes)
