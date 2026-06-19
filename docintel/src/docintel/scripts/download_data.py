"""Download source datasets (CORD, SROIE) into the local data directory.

Requires the optional ``data`` dependency group::

    pip install -e ".[data]"
    docintel-download-data --dataset cord

The Hugging Face ``datasets`` import is deferred so the command can be invoked
(e.g. ``--help``) without the heavy dependency installed.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from docintel.config import get_settings
from docintel.logging_config import configure_logging

logger = logging.getLogger("docintel.download")

# Dataset name -> Hugging Face Hub repository id.
DATASETS: dict[str, str] = {
    "cord": "naver-clova-ix/cord-v2",
    "sroie": "darentang/sroie",
}


def download(dataset: str, out_dir: Path) -> Path:
    """Download ``dataset`` from the Hub and persist it to ``out_dir``."""
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {sorted(DATASETS)}")

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only without extras
        raise SystemExit("The 'data' extra is required: pip install -e \".[data]\"") from exc

    target = out_dir / dataset
    target.mkdir(parents=True, exist_ok=True)
    logger.info("dataset.download.start", extra={"dataset": dataset, "target": str(target)})

    ds = load_dataset(DATASETS[dataset])
    ds.save_to_disk(str(target))

    logger.info("dataset.download.done", extra={"dataset": dataset, "target": str(target)})
    return target


def main() -> None:
    """CLI entrypoint."""
    settings = get_settings()
    configure_logging(settings.log_level)

    parser = argparse.ArgumentParser(description="Download DocIntel source datasets.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="cord",
        help="Dataset to download (default: cord).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(settings.data_dir) / "raw",
        help="Output directory (default: <data_dir>/raw).",
    )
    args = parser.parse_args()
    download(args.dataset, args.out_dir)


if __name__ == "__main__":
    main()
