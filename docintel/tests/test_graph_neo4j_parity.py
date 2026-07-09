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

    def _by_id(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        return sorted(rows, key=lambda r: str(r["contract_id"]))

    # Full-row parity, not just the id set: the real Cypher and the in-memory fake must
    # agree on every projected field — iso_date, the expiration/renewal answer text, and
    # their char offsets — or graph answers would cite different spans between the two.
    assert _by_id(real_rows) == _by_id(fake_rows)
    citation_fields = {"exp_answer", "exp_start", "exp_end", "ren_answer", "ren_start", "ren_end"}
    assert real_rows and all(citation_fields <= r.keys() for r in real_rows)
