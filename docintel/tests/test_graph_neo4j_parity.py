# tests/test_graph_neo4j_parity.py
from __future__ import annotations

import os

import pytest

from docintel.config import Settings
from docintel.graph.schema import ExpirationFact, GraphContract, RenewalFact
from docintel.graph.store import InMemoryGraphStore

pytestmark = pytest.mark.slow


@pytest.mark.skipif(not os.getenv("DOCINTEL_NEO4J_URI"), reason="no live Neo4j")
def test_cypher_matches_fake() -> None:
    from docintel.graph.templates import Neo4jGraphStore

    gc = GraphContract(
        contract_id="p1",
        expiration=ExpirationFact(
            iso_date="2026-02-01", answer_text="exp", char_start=0, char_end=3
        ),
        renewal=RenewalFact(answer_text="ren", char_start=4, char_end=7),
    )
    real = Neo4jGraphStore(Settings())
    fake = InMemoryGraphStore()
    real.upsert_contract(gc)
    fake.upsert_contract(gc)
    params = {"lower": "2026-01-01", "upper": "2026-12-31"}
    real_rows = real.run_template("auto_renewing_expiring_within", params)
    fake_rows = fake.run_template("auto_renewing_expiring_within", params)
    real.close()
    assert {r["contract_id"] for r in real_rows} == {r["contract_id"] for r in fake_rows}
