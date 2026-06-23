"""Tests for the pure token->word aggregation seam."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from docintel.config import Settings
from docintel.kie.backend import resolve_onnx_bundle, words_from_token_logits


def test_first_subword_token_drives_word_label() -> None:
    # 2 words; word 0 -> tokens 1,2 ; word 1 -> token 3. Token 0 is a special token.
    id2label = {0: "O", 1: "B-menu.nm", 2: "I-menu.nm"}
    # logits shape (seq=4, num_labels=3)
    logits = np.array(
        [
            [9.0, 0.0, 0.0],  # special (word_id None)
            [0.0, 9.0, 0.0],  # word 0 first token -> B-menu.nm
            [0.0, 0.0, 9.0],  # word 0 second token (ignored)
            [9.0, 0.0, 0.0],  # word 1 -> O
        ],
        dtype=np.float32,
    )
    word_ids = [None, 0, 0, 1]
    preds = words_from_token_logits(
        logits, word_ids, ["Coke", "x"], [(0, 0, 1, 1), (2, 2, 3, 3)], id2label
    )
    assert [p.label for p in preds] == ["B-menu.nm", "O"]
    assert preds[0].text == "Coke"
    assert 0.99 < preds[0].confidence <= 1.0


def _no_download(*_: object) -> Path:
    raise AssertionError("MLflow download must not be called when local path is set")


def test_resolve_onnx_bundle_prefers_local_path() -> None:
    settings = Settings(kie_onnx_local_path="models/cord-layoutlmv3-onnx-int8")
    assert resolve_onnx_bundle(settings, _no_download) == Path("models/cord-layoutlmv3-onnx-int8")


def test_resolve_onnx_bundle_falls_back_to_mlflow() -> None:
    settings = Settings(kie_onnx_local_path=None)
    seen: dict[str, object] = {}

    def fake_download(name: str, version: str, dest: Path, uri: str | None) -> Path:
        seen.update(name=name, version=version, dest=dest, uri=uri)
        return dest

    result = resolve_onnx_bundle(settings, fake_download)
    assert seen["name"] == settings.kie_onnx_registered_model_name
    assert seen["version"] == settings.kie_onnx_model_version
    assert result == Path(settings.data_dir) / "models" / settings.kie_onnx_registered_model_name
