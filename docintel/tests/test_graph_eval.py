# tests/test_graph_eval.py
from __future__ import annotations

from datetime import date

from docintel.config import Settings
from docintel.graph.eval import evaluate_multihop
from docintel.graph.schema import ExpirationFact, GraphContract, RenewalFact
from docintel.graph.store import InMemoryGraphStore


def _store() -> InMemoryGraphStore:
    store = InMemoryGraphStore()
    store.upsert_contract(
        GraphContract(
            contract_id="a",
            expiration=ExpirationFact(
                iso_date="2026-02-01", answer_text="e", char_start=0, char_end=1
            ),
            renewal=RenewalFact(answer_text="r", char_start=2, char_end=3),
        )
    )
    store.upsert_contract(
        GraphContract(
            contract_id="b",
            expiration=ExpirationFact(
                iso_date="2026-02-01", answer_text="e", char_start=0, char_end=1
            ),
        )
    )
    return store


def test_evaluate_multihop_scores_expected_contracts() -> None:
    cases = [
        ("which contracts expire within 60 days?", {"a", "b"}),
        ("which auto-renewing contracts expire within 60 days?", {"a"}),
    ]
    metrics = evaluate_multihop(_store(), cases, Settings(), reference_date=date(2026, 1, 15))
    assert metrics["multihop_accuracy"] == 1.0
    assert metrics["n"] == 2
