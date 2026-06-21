"""Tests for the pure, CPU-testable helpers in kie.train."""

from __future__ import annotations

import json
from pathlib import Path

from docintel.kie.config import TrainingConfig
from docintel.kie.train import collect_repro_params, save_bundle


def test_collect_repro_params_is_flat_string_map() -> None:
    config = TrainingConfig(model_name="microsoft/layoutlmv3-base", seed=7)
    params = collect_repro_params(config, dataset_revision="abc123", git_sha="deadbee")
    assert params["seed"] == "7"
    assert params["model_name"] == "microsoft/layoutlmv3-base"
    assert params["dataset_revision"] == "abc123"
    assert params["git_sha"] == "deadbee"
    assert all(isinstance(value, str) for value in params.values())


def test_save_bundle_writes_expected_layout(tmp_path: Path) -> None:
    saved: dict[str, Path] = {}

    class FakeSaveable:
        def __init__(self, name: str) -> None:
            self._name = name

        def save_pretrained(self, path: str) -> None:
            saved[self._name] = Path(path)
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "marker").write_text(self._name, encoding="utf-8")

    bundle_dir = tmp_path / "bundle"
    out = save_bundle(
        model=FakeSaveable("model"),
        processor=FakeSaveable("processor"),
        id2label={0: "O", 1: "B-menu.nm"},
        metrics={"f1": 0.95},
        bundle_dir=bundle_dir,
    )

    assert out == bundle_dir
    assert (bundle_dir / "model" / "marker").read_text(encoding="utf-8") == "model"
    assert (bundle_dir / "processor" / "marker").read_text(encoding="utf-8") == "processor"
    assert json.loads((bundle_dir / "label_map.json").read_text(encoding="utf-8")) == {
        "0": "O",
        "1": "B-menu.nm",
    }
    assert json.loads((bundle_dir / "metrics.json").read_text(encoding="utf-8")) == {"f1": 0.95}
