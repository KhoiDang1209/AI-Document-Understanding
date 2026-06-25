"""Tests for CUAD QA fine-tuning module."""

from __future__ import annotations

import json
from pathlib import Path

from docintel.contracts.qa_config import QaTrainingConfig
from docintel.contracts.train_qa import collect_repro_params, save_qa_bundle


def test_collect_repro_params_are_all_strings() -> None:
    cfg = QaTrainingConfig()
    params = collect_repro_params(cfg, dataset_revision="cuad-v1", git_sha="abc123")
    assert params["model_name"] == cfg.model_name
    assert params["dataset_revision"] == "cuad-v1"
    assert params["git_sha"] == "abc123"
    assert all(isinstance(v, str) for v in params.values())


class _FakeSavable:
    def save_pretrained(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "marker").write_text("ok", encoding="utf-8")


def test_save_qa_bundle_writes_artifacts(tmp_path: Path) -> None:
    bundle = save_qa_bundle(_FakeSavable(), _FakeSavable(), {"f1": 0.5}, tmp_path / "bundle")
    assert (bundle / "model" / "marker").exists()
    assert (bundle / "tokenizer" / "marker").exists()
    assert json.loads((bundle / "metrics.json").read_text(encoding="utf-8")) == {"f1": 0.5}
