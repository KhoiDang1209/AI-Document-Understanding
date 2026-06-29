"""Test the pure rule-based question router."""

from __future__ import annotations

from docintel.graph.router import route


def test_expiry_question_routes_to_graph() -> None:
    d = route("Which contracts expire within 90 days?")
    assert d.target == "graph"
    assert d.template == "expiring_within"
    assert d.within_days == 90


def test_auto_renew_plus_expiry_routes_to_auto_template() -> None:
    d = route("Which auto-renewing contracts expire within 30 days?")
    assert d.template == "auto_renewing_expiring_within"
    assert d.within_days == 30


def test_renewal_wording_detected() -> None:
    assert route("list contracts that renew and expire in 14 days").template == (
        "auto_renewing_expiring_within"
    )


def test_no_day_count_leaves_within_none() -> None:
    d = route("which contracts are expiring soon?")
    assert d.target == "graph" and d.within_days is None


def test_non_graph_question_routes_to_vector() -> None:
    d = route("What is the governing law of this agreement?")
    assert d.target == "vector" and d.template is None
