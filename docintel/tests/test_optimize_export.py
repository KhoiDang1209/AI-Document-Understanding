"""Tests for the pure seam in optimize.export."""

from __future__ import annotations

from docintel.optimize.export import resolve_model_uri


def test_resolve_model_uri_builds_registry_uri() -> None:
    assert resolve_model_uri("cord-layoutlmv3", "1") == "models:/cord-layoutlmv3/1"
    assert resolve_model_uri("m", "3") == "models:/m/3"


def test_export_qa_to_onnx_is_importable() -> None:
    from docintel.optimize.export import export_qa_to_onnx

    assert callable(export_qa_to_onnx)
