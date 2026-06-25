"""Pull the registered model from MLflow and export it to ONNX (fp32).

Heavy libraries (mlflow, optimum) are imported inside the functions so the
module loads cheaply and the pure ``resolve_model_uri`` seam stays testable.
"""

from __future__ import annotations

from pathlib import Path


def resolve_model_uri(name: str, version: str) -> str:
    """MLflow registry URI for a registered model version."""
    return f"models:/{name}/{version}"


def download_registered_model(
    name: str,
    version: str,
    dest: Path,
    tracking_uri: str | None = None,
) -> Path:
    """Download a registered model's bundle artifacts; return the bundle root."""
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    dest.mkdir(parents=True, exist_ok=True)
    local = mlflow.artifacts.download_artifacts(
        artifact_uri=resolve_model_uri(name, version),
        dst_path=str(dest),
    )
    return Path(local)


def export_to_onnx(model_dir: Path, out_dir: Path) -> Path:
    """Export a LayoutLMv3 token-classification model to ONNX (fp32) via Optimum."""
    from optimum.onnxruntime import ORTModelForTokenClassification

    model = ORTModelForTokenClassification.from_pretrained(str(model_dir), export=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    return out_dir


def export_qa_to_onnx(model_dir: Path, out_dir: Path) -> Path:
    """Export a question-answering model to ONNX (fp32) via Optimum."""
    from optimum.onnxruntime import ORTModelForQuestionAnswering

    model = ORTModelForQuestionAnswering.from_pretrained(str(model_dir), export=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    return out_dir
