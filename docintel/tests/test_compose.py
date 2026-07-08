"""Guards the core docker-compose service set against scope drift."""

from __future__ import annotations

from pathlib import Path

import yaml

COMPOSE_FILE = Path(__file__).resolve().parent.parent / "docker-compose.yml"

# Core MLOps spine, the Phase 5 observability stack, the C2 Qdrant vector store, the
# C3 Neo4j knowledge graph, and the C4 self-hosted Langfuse tracing stack. The Streamlit
# UI runs locally (not containerised).
EXPECTED_SERVICES = {
    "api",
    "mlflow",
    "minio",
    "prometheus",
    "loki",
    "promtail",
    "grafana",
    "qdrant",
    "neo4j",
    "langfuse",
    "langfuse-postgres",
}


def test_core_services() -> None:
    compose = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    assert set(compose["services"]) == EXPECTED_SERVICES


def test_api_neo4j_password_matches_neo4j_auth() -> None:
    """The api service must pass the same neo4j password the container is created with,
    or graph-routed /ask queries fail auth and fall back silently."""
    compose = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    _, _, expected_password = compose["services"]["neo4j"]["environment"]["NEO4J_AUTH"].partition(
        "/"
    )
    api_env = compose["services"]["api"]["environment"]
    assert api_env.get("DOCINTEL_NEO4J_PASSWORD") == expected_password
