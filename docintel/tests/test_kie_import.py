"""Tests for importing a bundle into MLflow + registering it."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from docintel.config import Settings
from docintel.kie.import_run import import_bundle


def _make_bundle(bundle_dir: Path) -> Path:
    (bundle_dir / "model").mkdir(parents=True)
    (bundle_dir / "model" / "config.json").write_text("{}", encoding="utf-8")
    (bundle_dir / "processor").mkdir(parents=True)
    (bundle_dir / "processor" / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (bundle_dir / "label_map.json").write_text(
        json.dumps({"0": "O", "1": "B-menu.nm"}), encoding="utf-8"
    )
    (bundle_dir / "metrics.json").write_text(json.dumps({"f1": 0.95}), encoding="utf-8")
    return bundle_dir


def test_import_bundle_registers_model(tmp_path: Path) -> None:
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    bundle = _make_bundle(tmp_path / "bundle")
    settings = Settings(kie_registered_model_name="cord-layoutlmv3")

    version = import_bundle(bundle, settings, tracking_uri=tracking_uri)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    model = client.get_registered_model("cord-layoutlmv3")
    assert model.name == "cord-layoutlmv3"
    assert version == "1"

    # The run logged the f1 metric and the label count param.
    run_id = client.get_model_version("cord-layoutlmv3", version).run_id
    run = client.get_run(run_id)
    assert run.data.metrics["f1"] == 0.95
    assert run.data.params["num_labels"] == "2"
