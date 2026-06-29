"""Thin tool adapters that expose C1-C3 capabilities to the agent.

Each tool is a side-effect-free wrapper over an existing function - no retrieval,
generation, or extraction logic is duplicated here. The agent graph composes them.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from docintel.config import Settings
from docintel.contracts.extractor import ContractExtractor
from docintel.contracts.schema import ContractDocument, build_derived
from docintel.graph.query import run_graph_query
from docintel.graph.router import route
from docintel.rag.answer import generate_or_degrade
from docintel.rag.schema import AskResponse, RetrievedChunk
from docintel.rag.store import search


def extract_tool(
    text: str, extractor: ContractExtractor, source: str = "digital"
) -> ContractDocument:
    """Extract clauses from raw contract text and assemble a ContractDocument."""
    clauses = extractor.extract(text)
    return ContractDocument(
        id=uuid4().hex,
        source="digital" if source == "digital" else "ocr",
        clauses=clauses,
        derived=build_derived(clauses),
        page_count=1,
        created_at=datetime.now(UTC).isoformat(),
    )


def vector_retrieve_tool(
    question: str, store: Any, settings: Settings, contract_id: str | None = None
) -> list[RetrievedChunk]:
    """Top-k vector retrieval over indexed contracts (optionally scoped to one)."""
    return search(store, question, settings.rag_top_k, contract_id)


def graph_query_tool(
    question: str, graph_store: Any | None, settings: Settings
) -> list[RetrievedChunk]:
    """Route the question; return cited graph facts, or [] when not a graph question."""
    decision = route(question)
    if decision.target != "graph" or graph_store is None:
        return []
    return run_graph_query(graph_store, decision, settings)


def generate_tool(
    question: str, citations: list[RetrievedChunk], llm: Any | None, contract_id: str | None
) -> AskResponse:
    """Generate a grounded answer from citations, or degrade to citations-only."""
    return generate_or_degrade(question, citations, llm, contract_id)
