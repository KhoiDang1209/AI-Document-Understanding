# tests/test_graph_build.py
from __future__ import annotations

from docintel.contracts.schema import ContractDocument, ExtractedClause
from docintel.graph.build import build_contract
from docintel.graph.store import InMemoryGraphStore


def _doc() -> ContractDocument:
    return ContractDocument(
        id="c1",
        source="digital",
        clauses=[
            ExtractedClause(
                clause_type="Expiration Date",
                answer_text="expires on December 31, 2025",
                char_start=0,
                char_end=28,
                confidence=0.9,
            )
        ],
        derived={},
        page_count=1,
        created_at="t",
    )


def test_build_contract_upserts_and_reports_facts() -> None:
    store = InMemoryGraphStore()
    assert build_contract(_doc(), store) is True
    rows = store.run_template("expiring_within", {"lower": "2025-01-01", "upper": "2025-12-31"})
    assert [r["contract_id"] for r in rows] == ["c1"]


def test_build_contract_with_no_facts_returns_false() -> None:
    empty = ContractDocument(
        id="c2", source="digital", clauses=[], derived={}, page_count=1, created_at="t"
    )
    store = InMemoryGraphStore()
    assert build_contract(empty, store) is False
