"""Guards the core docker-compose service set against scope drift."""

from __future__ import annotations

from pathlib import Path

import yaml

COMPOSE_FILE = Path(__file__).resolve().parent.parent / "docker-compose.yml"

# Core = MLOps spine only. Qdrant (GraphRAG) and Prometheus/Grafana/Loki
# (observability) are deliberately out of the Phase 0 core stack.
EXPECTED_SERVICES = {"api", "mlflow", "minio"}


def test_core_services() -> None:
    compose = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    assert set(compose["services"]) == EXPECTED_SERVICES
