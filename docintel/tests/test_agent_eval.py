from __future__ import annotations

from qdrant_client import QdrantClient

from docintel.agent.eval import evaluate_agent
from docintel.agent.graph import AgentDeps
from docintel.config import Settings
from docintel.graph.schema import ExpirationFact, GraphContract
from docintel.graph.store import InMemoryGraphStore
from docintel.rag.chunk import build_chunks
from docintel.rag.embed import build_embedder
from docintel.rag.store import build_vector_store, ensure_collection, upsert_chunks


def _deps() -> AgentDeps:
    settings = Settings()
    client = QdrantClient(":memory:")
    ensure_collection(client, settings.qdrant_collection, settings.rag_embedding_dim)
    store = build_vector_store(settings, build_embedder(settings), client=client)
    upsert_chunks(store, "a", build_chunks("Governing law New York.", [], 1200, 200))
    graph = InMemoryGraphStore()
    graph.upsert_contract(
        GraphContract(
            contract_id="a",
            expiration=ExpirationFact(
                iso_date="2999-01-01", answer_text="expires 2999-01-01", char_start=0, char_end=18
            ),
        )
    )
    return AgentDeps(settings=settings, rag_store=store, graph_store=graph, llm=None)


def test_evaluate_agent_scores_grounding() -> None:
    cases = [("which contracts expire within 400000 days?", None, {"a"})]
    metrics = evaluate_agent(cases, _deps())
    assert metrics["success_rate"] == 1.0 and metrics["n"] == 1.0


def test_evaluate_agent_empty_cases() -> None:
    assert evaluate_agent([], _deps()) == {"success_rate": 0.0, "n": 0.0}
