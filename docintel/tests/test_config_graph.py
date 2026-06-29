from __future__ import annotations

from docintel.config import Settings


def test_graph_settings_defaults() -> None:
    s = Settings()
    assert s.neo4j_uri == "bolt://neo4j:7687"
    assert s.neo4j_user == "neo4j"
    assert s.neo4j_database == "neo4j"
    assert s.graph_enabled is True
    assert s.graph_default_within_days == 90
