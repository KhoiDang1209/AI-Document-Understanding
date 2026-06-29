"""Parameterized Cypher templates and the real Neo4j-backed GraphStore.

Only two deterministic templates are exposed (no Cypher hallucination, no text-to-Cypher).
Citations travel on the EXPIRES_ON / HAS_CLAUSE relationships so answers stay grounded.
"""

from __future__ import annotations

from typing import Any

from docintel.config import Settings
from docintel.graph.normalize import RENEWAL_CLAUSE
from docintel.graph.schema import GraphContract

TEMPLATES: dict[str, str] = {
    "expiring_within": (
        "MATCH (c:Contract)-[r:EXPIRES_ON]->(d:Date) "
        "WHERE d.iso >= $lower AND d.iso <= $upper "
        "RETURN c.id AS contract_id, d.iso AS iso_date, "
        "r.answer_text AS exp_answer, r.char_start AS exp_start, r.char_end AS exp_end "
        "ORDER BY d.iso"
    ),
    "auto_renewing_expiring_within": (
        "MATCH (c:Contract)-[r:EXPIRES_ON]->(d:Date) "
        "MATCH (c)-[hr:HAS_CLAUSE]->(:ClauseType {name: $renewal}) "
        "WHERE d.iso >= $lower AND d.iso <= $upper "
        "RETURN c.id AS contract_id, d.iso AS iso_date, "
        "r.answer_text AS exp_answer, r.char_start AS exp_start, r.char_end AS exp_end, "
        "hr.answer_text AS ren_answer, hr.char_start AS ren_start, hr.char_end AS ren_end "
        "ORDER BY d.iso"
    ),
}

_UPSERT = (
    "MERGE (c:Contract {id: $contract_id}) "
    "WITH c "
    "OPTIONAL MATCH (c)-[old:EXPIRES_ON|HAS_CLAUSE]->() DELETE old "
    "WITH c "
    "FOREACH (_ IN CASE WHEN $iso IS NULL THEN [] ELSE [1] END | "
    "  MERGE (d:Date {iso: $iso}) "
    "  MERGE (c)-[r:EXPIRES_ON]->(d) "
    "  SET r.answer_text = $exp_answer, r.char_start = $exp_start, r.char_end = $exp_end) "
    "FOREACH (_ IN CASE WHEN $ren_answer IS NULL THEN [] ELSE [1] END | "
    "  MERGE (ct:ClauseType {name: $renewal}) "
    "  MERGE (c)-[hr:HAS_CLAUSE]->(ct) "
    "  SET hr.answer_text = $ren_answer, hr.char_start = $ren_start, hr.char_end = $ren_end)"
)


def get_template(name: str) -> str:
    """Return the Cypher for a template name (raises KeyError if unknown)."""
    return TEMPLATES[name]


class Neo4jGraphStore:
    """Real GraphStore backed by the official neo4j driver (lazy-imported)."""

    def __init__(self, settings: Settings) -> None:
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )
        self._database = settings.neo4j_database

    def upsert_contract(self, gc: GraphContract) -> None:
        params: dict[str, Any] = {
            "contract_id": gc.contract_id,
            "renewal": RENEWAL_CLAUSE,
            "iso": gc.expiration.iso_date if gc.expiration else None,
            "exp_answer": gc.expiration.answer_text if gc.expiration else None,
            "exp_start": gc.expiration.char_start if gc.expiration else None,
            "exp_end": gc.expiration.char_end if gc.expiration else None,
            "ren_answer": gc.renewal.answer_text if gc.renewal else None,
            "ren_start": gc.renewal.char_start if gc.renewal else None,
            "ren_end": gc.renewal.char_end if gc.renewal else None,
        }
        with self._driver.session(database=self._database) as session:
            session.run(_UPSERT, **params)

    def run_template(self, name: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        cypher = get_template(name)
        merged = {"renewal": RENEWAL_CLAUSE, **params}
        with self._driver.session(database=self._database) as session:
            return [record.data() for record in session.run(cypher, **merged)]

    def close(self) -> None:
        self._driver.close()
