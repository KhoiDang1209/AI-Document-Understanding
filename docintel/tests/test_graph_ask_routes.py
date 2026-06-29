# tests/test_graph_ask_routes.py
from __future__ import annotations

from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from docintel.api.main import create_app
from docintel.api.routes.ask import get_graph_store, get_rag_llm
from docintel.config import Settings, get_settings
from docintel.graph.schema import ExpirationFact, GraphContract, RenewalFact
from docintel.graph.store import InMemoryGraphStore


def _graph_store() -> InMemoryGraphStore:
    store = InMemoryGraphStore()
    # Far-future date so the default window keeps the fixture valid over time.
    store.upsert_contract(
        GraphContract(
            contract_id="a",
            expiration=ExpirationFact(
                iso_date="2999-02-01", answer_text="expires 2999-02-01", char_start=0, char_end=18
            ),
            renewal=RenewalFact(answer_text="auto-renews", char_start=20, char_end=31),
        )
    )
    return store


def test_graph_route_degrades_without_llm() -> None:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings()
    app.dependency_overrides[get_graph_store] = lambda: _graph_store()
    app.dependency_overrides[get_rag_llm] = lambda: None
    with TestClient(app) as client:
        resp = client.post(
            "/ask", json={"question": "which auto-renewing contracts expire within 400000 days?"}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["generation_skipped"] is True
    kinds = {c["clause_type"] for c in body["citations"]}
    assert kinds == {"Expiration Date", "Renewal Term"}


def test_graph_route_generates_with_llm() -> None:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings()
    app.dependency_overrides[get_graph_store] = lambda: _graph_store()
    app.dependency_overrides[get_rag_llm] = lambda: FakeListChatModel(responses=["One contract."])
    with TestClient(app) as client:
        resp = client.post("/ask", json={"question": "which contracts expire within 400000 days?"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "One contract."
