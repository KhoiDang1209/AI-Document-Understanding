from __future__ import annotations

from docintel.contracts.schema import ContractDocument, ExtractedClause
from docintel.graph.normalize import build_graph_contract, parse_iso_date


def test_parse_iso_date_long_form() -> None:
    assert parse_iso_date("This Agreement shall expire on December 31, 2025.") == "2025-12-31"


def test_parse_iso_date_numeric_and_iso() -> None:
    assert parse_iso_date("term ends 01/05/2026") == "2026-01-05"
    assert parse_iso_date("until 2027-03-09 inclusive") == "2027-03-09"


def test_parse_iso_date_unparseable_returns_none() -> None:
    assert parse_iso_date("expires at the end of the term") is None


def _doc(clauses: list[ExtractedClause]) -> ContractDocument:
    return ContractDocument(
        id="c1", source="digital", clauses=clauses, derived={}, page_count=1, created_at="t"
    )


def test_build_graph_contract_extracts_expiration_and_renewal() -> None:
    doc = _doc(
        [
            ExtractedClause(
                clause_type="Expiration Date",
                answer_text="expire on December 31, 2025",
                char_start=10,
                char_end=37,
                confidence=0.9,
            ),
            ExtractedClause(
                clause_type="Renewal Term",
                answer_text="renews for successive one-year terms",
                char_start=50,
                char_end=86,
                confidence=0.8,
            ),
        ]
    )
    gc = build_graph_contract(doc)
    assert gc.contract_id == "c1"
    assert gc.expiration is not None and gc.expiration.iso_date == "2025-12-31"
    assert gc.renewal is not None and gc.renewal.char_start == 50


def test_build_graph_contract_skips_unparseable_expiration() -> None:
    doc = _doc(
        [
            ExtractedClause(
                clause_type="Expiration Date",
                answer_text="end of the term",
                char_start=0,
                char_end=15,
                confidence=0.9,
            )
        ]
    )
    gc = build_graph_contract(doc)
    assert gc.expiration is None and gc.renewal is None
