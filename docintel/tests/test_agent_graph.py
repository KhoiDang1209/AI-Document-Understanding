from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from qdrant_client import QdrantClient

from docintel.agent.graph import AgentDeps, run_agent
from docintel.config import Settings
from docintel.graph.schema import ExpirationFact, GraphContract
from docintel.graph.store import InMemoryGraphStore
from docintel.rag.chunk import build_chunks
from docintel.rag.embed import build_embedder
from docintel.rag.store import build_vector_store, ensure_collection, upsert_chunks


def _vector_store(settings: Settings) -> object:
    client = QdrantClient(":memory:")
    ensure_collection(client, settings.qdrant_collection, settings.rag_embedding_dim)
    store = build_vector_store(settings, build_embedder(settings), client=client)
    chunks = build_chunks("The governing law is the State of New York.", [], 1200, 200)
    upsert_chunks(store, "a", chunks)
    return store


def _graph_store() -> InMemoryGraphStore:
    store = InMemoryGraphStore()
    store.upsert_contract(
        GraphContract(
            contract_id="a",
            expiration=ExpirationFact(
                iso_date="2999-01-01", answer_text="expires 2999-01-01", char_start=0, char_end=18
            ),
        )
    )
    return store


def test_vector_task_generates_answer() -> None:
    settings = Settings()
    deps = AgentDeps(
        settings=settings,
        rag_store=_vector_store(settings),
        graph_store=_graph_store(),
        llm=FakeListChatModel(responses=["Governing law is New York."]),
    )
    resp = run_agent("What is the governing law?", None, deps)
    assert resp.status == "ok"
    assert resp.answer == "Governing law is New York."
    assert resp.citations  # retrieved at least one chunk


def test_graph_task_degrades_without_llm() -> None:
    settings = Settings()
    deps = AgentDeps(
        settings=settings,
        rag_store=_vector_store(settings),
        graph_store=_graph_store(),
        llm=None,
    )
    resp = run_agent("which contracts expire within 400000 days?", None, deps)
    assert resp.status == "degraded"
    assert resp.answer is None
    assert [c.contract_id for c in resp.citations] == ["a"]


def test_graph_empty_falls_back_to_vector_and_recovers() -> None:
    # The point of the retry edge: a graph-routed question whose graph store is empty
    # returns no rows, so the bounded retry must fall back to vector retrieval, recover
    # a citation, and let generation produce an answer.
    settings = Settings()
    deps = AgentDeps(
        settings=settings,
        rag_store=_vector_store(settings),  # holds the governing-law chunk
        graph_store=InMemoryGraphStore(),  # empty -> graph query yields nothing
        llm=FakeListChatModel(responses=["Recovered from vector."]),
    )
    resp = run_agent("which contracts expire within 400000 days?", None, deps)
    assert resp.status == "ok"
    assert resp.answer == "Recovered from vector."
    assert resp.retries == 1
    assert resp.citations  # recovered via the vector fallback
    assert "retrieve:graph:0" in resp.steps
    assert any(s.startswith("retrieve:vector:") for s in resp.steps)


def test_retry_caps_and_marks_degraded_when_nothing_found() -> None:
    settings = Settings()  # agent_max_retries=1
    empty_vector = _vector_store(settings)
    deps = AgentDeps(
        settings=settings, rag_store=empty_vector, graph_store=InMemoryGraphStore(), llm=None
    )
    # A graph question whose store is empty -> graph yields [], fallback to vector also weak.
    resp = run_agent("which contracts expire within 1 day?", None, deps)
    assert resp.retries <= settings.agent_max_retries
    assert resp.status == "degraded"
