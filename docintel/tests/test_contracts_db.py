from __future__ import annotations

from typing import Any

from docintel.contracts.schema import ContractDocument, ExtractedClause
from docintel.storage.contracts_db import get_contract, init_contracts_db, save_contract


def _doc(cid: str) -> ContractDocument:
    return ContractDocument(
        id=cid,
        source="digital",
        clauses=[
            ExtractedClause(
                clause_type="Parties",
                answer_text="Acme",
                char_start=0,
                char_end=4,
                confidence=0.9,
            )
        ],
        derived={"Parties": ["Acme"]},
        page_count=1,
        created_at="2026-06-25T00:00:00+00:00",
    )


def test_save_and_get_roundtrip(tmp_path: Any) -> None:
    path = str(tmp_path / "c.db")
    init_contracts_db(path)
    save_contract(path, _doc("c1"), "c1.pdf")
    found = get_contract(path, "c1")
    assert found is not None
    doc, key = found
    assert doc == _doc("c1")
    assert key == "c1.pdf"


def test_get_missing_returns_none(tmp_path: Any) -> None:
    path = str(tmp_path / "c.db")
    init_contracts_db(path)
    assert get_contract(path, "nope") is None


def test_save_upserts(tmp_path: Any) -> None:
    path = str(tmp_path / "c.db")
    init_contracts_db(path)
    save_contract(path, _doc("c1"), "old.pdf")
    save_contract(path, _doc("c1"), "new.pdf")
    found = get_contract(path, "c1")
    assert found is not None and found[1] == "new.pdf"
