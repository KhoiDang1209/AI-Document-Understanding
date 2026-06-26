"""Guards the core docker-compose service set against scope drift."""

from __future__ import annotations

from pathlib import Path

import yaml

COMPOSE_FILE = Path(__file__).resolve().parent.parent / "docker-compose.yml"

# Core MLOps spine, the Phase 5 observability stack, and the C2 Qdrant vector store.
# The Streamlit UI runs locally (not containerised).
EXPECTED_SERVICES = {
    "api",
    "mlflow",
    "minio",
    "prometheus",
    "loki",
    "promtail",
    "grafana",
    "qdrant",
}


def test_core_services() -> None:
    compose = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    assert set(compose["services"]) == EXPECTED_SERVICES
