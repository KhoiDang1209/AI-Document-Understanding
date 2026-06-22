"""Import a Colab-trained model bundle into the local MLflow + MinIO registry.

Run on the laptop after downloading the bundle from Colab::

    docintel-import-kie --bundle-dir ./cord-layoutlmv3-bundle

Logs the run's params + metrics, uploads the bundle artifacts to the MLflow
artifact store (MinIO in docker-compose), and registers the model under the
configured registry name.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path

import mlflow

from docintel.config import Settings, get_settings
from docintel.logging_config import configure_logging

logger = logging.getLogger("docintel.kie.import")


def import_bundle(
    bundle_dir: Path,
    settings: Settings,
    tracking_uri: str | None = None,
) -> str:
    """Log + register a downloaded bundle; return the new model version."""
    uri = tracking_uri or settings.mlflow_tracking_uri
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("cord-kie")

    label_map: dict[str, str] = json.loads(
        (bundle_dir / "label_map.json").read_text(encoding="utf-8")
    )
    metrics: dict[str, float] = json.loads(
        (bundle_dir / "metrics.json").read_text(encoding="utf-8")
    )

    with mlflow.start_run() as run:
        mlflow.log_param("num_labels", str(len(label_map)))
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, int | float)})
        mlflow.log_artifacts(str(bundle_dir), artifact_path="bundle")
        model_uri = f"runs:/{run.info.run_id}/bundle"
        registered = mlflow.register_model(model_uri, settings.kie_registered_model_name)

    logger.info(
        "kie.import.done",
        extra={"model": settings.kie_registered_model_name, "version": registered.version},
    )
    return str(registered.version)


def main() -> None:
    """CLI entry point for ``docintel-import-kie``."""
    # MLflow prints run URLs with an emoji; force UTF-8 so non-UTF-8 consoles
    # (e.g. Windows cp1252) don't crash on the unencodable character.
    if isinstance(sys.stdout, io.TextIOWrapper) and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    settings = get_settings()
    configure_logging(settings.log_level)

    parser = argparse.ArgumentParser(description="Import a KIE bundle into MLflow.")
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        required=True,
        help="Path to the downloaded model bundle directory.",
    )
    args = parser.parse_args()
    version = import_bundle(args.bundle_dir, settings)
    logger.info("kie.import.registered", extra={"version": version})


if __name__ == "__main__":
    main()
