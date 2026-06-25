from __future__ import annotations

from docintel.contracts.schema import ContractDocument, ExtractedClause, build_derived


def _clause(t: str, text: str) -> ExtractedClause:
    return ExtractedClause(
        clause_type=t,
        answer_text=text,
        char_start=0,
        char_end=len(text),
        confidence=0.9,
    )


def test_build_derived_groups_by_type() -> None:
    clauses = [
        _clause("Parties", "Acme"),
        _clause("Parties", "Globex"),
        _clause("Governing Law", "NY"),
    ]
    derived = build_derived(clauses)
    assert derived == {
        "Parties": ["Acme", "Globex"],
        "Governing Law": ["NY"],
    }


def test_contract_document_roundtrips() -> None:
    doc = ContractDocument(
        id="abc",
        source="digital",
        clauses=[_clause("Parties", "Acme")],
        derived={"Parties": ["Acme"]},
        page_count=3,
        created_at="2026-06-25T00:00:00+00:00",
    )
    again = ContractDocument.model_validate_json(doc.model_dump_json())
    assert again == doc
