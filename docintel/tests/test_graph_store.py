"""Tests for GraphStore protocol, in-memory fake, and builder."""

from __future__ import annotations

from docintel.config import Settings
from docintel.graph.schema import ExpirationFact, GraphContract, RenewalFact
from docintel.graph.store import InMemoryGraphStore, build_graph_store


def _gc(cid: str, iso: str, renews: bool) -> GraphContract:
    return GraphContract(
        contract_id=cid,
        expiration=ExpirationFact(iso_date=iso, answer_text=f"exp {iso}", char_start=0, char_end=5),
        renewal=RenewalFact(answer_text="renews", char_start=10, char_end=16) if renews else None,
    )


def test_expiring_within_filters_by_date_window() -> None:
    store = InMemoryGraphStore()
    store.upsert_contract(_gc("a", "2025-02-01", renews=False))
    store.upsert_contract(_gc("b", "2025-12-31", renews=True))
    rows = store.run_template("expiring_within", {"lower": "2025-01-01", "upper": "2025-06-30"})
    assert [r["contract_id"] for r in rows] == ["a"]
    assert rows[0]["exp_answer"] == "exp 2025-02-01"


def test_auto_renewing_requires_renewal_clause() -> None:
    store = InMemoryGraphStore()
    store.upsert_contract(_gc("a", "2025-02-01", renews=False))
    store.upsert_contract(_gc("b", "2025-03-01", renews=True))
    rows = store.run_template(
        "auto_renewing_expiring_within", {"lower": "2025-01-01", "upper": "2025-06-30"}
    )
    assert [r["contract_id"] for r in rows] == ["b"]
    assert rows[0]["ren_answer"] == "renews"


def test_upsert_is_idempotent_per_contract() -> None:
    store = InMemoryGraphStore()
    store.upsert_contract(_gc("a", "2025-02-01", renews=False))
    store.upsert_contract(_gc("a", "2025-02-02", renews=False))
    rows = store.run_template("expiring_within", {"lower": "2025-01-01", "upper": "2025-12-31"})
    assert len(rows) == 1 and rows[0]["iso_date"] == "2025-02-02"


def test_build_graph_store_disabled_returns_none() -> None:
    assert build_graph_store(Settings(graph_enabled=False)) is None


def test_unknown_template_raises_keyerror() -> None:
    import pytest

    store = InMemoryGraphStore()
    with pytest.raises(KeyError):
        store.run_template("nope", {"lower": "2025-01-01", "upper": "2025-12-31"})
