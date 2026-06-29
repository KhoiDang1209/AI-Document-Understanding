# tests/test_graph_query.py
from __future__ import annotations

from datetime import date

from docintel.config import Settings
from docintel.graph.query import run_graph_query
from docintel.graph.schema import ExpirationFact, GraphContract, RenewalFact, RouteDecision
from docintel.graph.store import InMemoryGraphStore


def _store() -> InMemoryGraphStore:
    store = InMemoryGraphStore()
    store.upsert_contract(
        GraphContract(
            contract_id="a",
            expiration=ExpirationFact(
                iso_date="2026-02-01", answer_text="expires 2026-02-01", char_start=0, char_end=10
            ),
            renewal=RenewalFact(answer_text="auto-renews", char_start=20, char_end=31),
        )
    )
    return store


def test_expiring_within_emits_expiration_citation() -> None:
    decision = RouteDecision(target="graph", template="expiring_within", within_days=60)
    chunks = run_graph_query(_store(), decision, Settings(), reference_date=date(2026, 1, 15))
    assert len(chunks) == 1
    c = chunks[0]
    assert c.contract_id == "a" and c.chunk_kind == "graph"
    assert c.clause_type == "Expiration Date" and c.char_start == 0 and c.char_end == 10


def test_auto_renew_emits_two_citations() -> None:
    decision = RouteDecision(
        target="graph", template="auto_renewing_expiring_within", within_days=60
    )
    chunks = run_graph_query(_store(), decision, Settings(), reference_date=date(2026, 1, 15))
    kinds = {c.clause_type for c in chunks}
    assert kinds == {"Expiration Date", "Renewal Term"}


def test_default_window_used_when_within_none() -> None:
    decision = RouteDecision(target="graph", template="expiring_within", within_days=None)
    # graph_default_within_days=90 -> 2026-01-15 .. 2026-04-15 includes 2026-02-01
    chunks = run_graph_query(_store(), decision, Settings(), reference_date=date(2026, 1, 15))
    assert len(chunks) == 1


def test_out_of_window_returns_empty() -> None:
    decision = RouteDecision(target="graph", template="expiring_within", within_days=5)
    chunks = run_graph_query(_store(), decision, Settings(), reference_date=date(2026, 1, 15))
    assert chunks == []
