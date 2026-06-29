from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from docintel.agent.tools import (
    extract_tool,
    generate_tool,
    graph_query_tool,
)
from docintel.config import Settings
from docintel.contracts.schema import ExtractedClause
from docintel.graph.schema import ExpirationFact, GraphContract
from docintel.graph.store import InMemoryGraphStore
from docintel.rag.schema import RetrievedChunk


class _FakeExtractor:
    def extract(self, text: str) -> list[ExtractedClause]:
        return [
            ExtractedClause(
                clause_type="Governing Law",
                answer_text="New York",
                char_start=0,
                char_end=8,
                confidence=0.9,
            )
        ]


def test_extract_tool_builds_document() -> None:
    doc = extract_tool("some contract text", _FakeExtractor())
    assert doc.clauses[0].clause_type == "Governing Law"
    assert doc.derived["Governing Law"] == ["New York"]


def test_graph_query_tool_routes_and_queries() -> None:
    store = InMemoryGraphStore()
    store.upsert_contract(
        GraphContract(
            contract_id="a",
            expiration=ExpirationFact(
                iso_date="2999-01-01", answer_text="expires 2999-01-01", char_start=0, char_end=18
            ),
        )
    )
    chunks = graph_query_tool("which contracts expire within 400000 days?", store, Settings())
    assert [c.contract_id for c in chunks] == ["a"]


def test_graph_query_tool_returns_empty_for_non_graph_question() -> None:
    chunks = graph_query_tool("what is the governing law?", InMemoryGraphStore(), Settings())
    assert chunks == []


def test_generate_tool_degrades_without_llm() -> None:
    cite = RetrievedChunk(
        contract_id="a",
        chunk_index=0,
        chunk_kind="graph",
        clause_type="Governing Law",
        text="New York",
        score=1.0,
        char_start=0,
        char_end=8,
    )
    resp = generate_tool("q", [cite], None, "a")
    assert resp.generation_skipped is True and resp.citations == [cite]
    ok = generate_tool("q", [cite], FakeListChatModel(responses=["A."]), "a")
    assert ok.answer == "A."
