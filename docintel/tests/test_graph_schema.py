"""Test graph schema models."""

from __future__ import annotations

from docintel.graph.schema import ExpirationFact, GraphContract, RenewalFact, RouteDecision


def test_graph_contract_holds_optional_facts() -> None:
    gc = GraphContract(
        contract_id="c1",
        expiration=ExpirationFact(
            iso_date="2025-12-31",
            answer_text="expires on December 31, 2025",
            char_start=10,
            char_end=40,
        ),
        renewal=RenewalFact(answer_text="auto-renews annually", char_start=50, char_end=70),
    )
    assert gc.expiration is not None and gc.expiration.iso_date == "2025-12-31"
    assert gc.renewal is not None and gc.renewal.char_start == 50


def test_graph_contract_defaults_none() -> None:
    gc = GraphContract(contract_id="c2")
    assert gc.expiration is None and gc.renewal is None


def test_route_decision_minimal() -> None:
    d = RouteDecision(target="vector")
    assert d.template is None and d.within_days is None
