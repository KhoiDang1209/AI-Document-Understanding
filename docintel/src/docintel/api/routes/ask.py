"""The /ask endpoint: retrieve cited chunks and generate-or-degrade an answer."""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from docintel.api.metrics import Metrics
from docintel.api.routes.extract import get_metrics
from docintel.config import Settings, get_settings
from docintel.graph.query import run_graph_query
from docintel.graph.router import route
from docintel.graph.store import build_graph_store
from docintel.rag.answer import answer_question, generate_or_degrade
from docintel.rag.embed import build_embedder
from docintel.rag.llm import build_llm
from docintel.rag.schema import AskRequest, AskResponse
from docintel.rag.store import build_vector_store

logger = logging.getLogger("docintel.api.ask")
router = APIRouter(tags=["rag"])


def ensure_rag_store(app: Any, settings: Settings) -> Any:
    """Build the vector store once and cache it on app.state (no network at build time)."""
    store = getattr(app.state, "rag_store", None)
    if store is None:
        store = build_vector_store(settings, build_embedder(settings))
        app.state.rag_store = store
    return store


def get_rag_store(request: Request, settings: Settings = Depends(get_settings)) -> Any:  # noqa: B008
    """Vector store dependency for /ask (propagates build failures)."""
    return ensure_rag_store(request.app, settings)


def get_rag_store_optional(
    request: Request,
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> Any | None:
    """Best-effort vector store for indexing; returns None instead of raising."""
    try:
        return ensure_rag_store(request.app, settings)
    except Exception:
        logger.warning("rag.store.unavailable", exc_info=True)
        return None


def get_rag_llm(request: Request, settings: Settings = Depends(get_settings)) -> Any | None:  # noqa: B008
    """LLM dependency: cached when configured, else None (drives graceful degrade)."""
    llm = getattr(request.app.state, "rag_llm", None)
    if llm is None:
        llm = build_llm(settings)
        request.app.state.rag_llm = llm
    return llm


def ensure_graph_store(app: Any, settings: Settings) -> Any:
    """Build the graph store once and cache it on app.state (None when disabled)."""
    store = getattr(app.state, "graph_store", None)
    if store is None:
        store = build_graph_store(settings)
        app.state.graph_store = store
    return store


def get_graph_store(request: Request, settings: Settings = Depends(get_settings)) -> Any | None:  # noqa: B008
    """Graph store dependency for /ask routing; None when graph is disabled."""
    return ensure_graph_store(request.app, settings)


def get_graph_store_optional(
    request: Request,
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> Any | None:
    """Best-effort graph store for extract-time build; returns None instead of raising."""
    try:
        return ensure_graph_store(request.app, settings)
    except Exception:
        logger.warning("graph.store.unavailable", exc_info=True)
        return None


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question grounded in indexed contracts",
)
def ask(
    req: AskRequest,
    settings: Settings = Depends(get_settings),  # noqa: B008
    store: Any = Depends(get_rag_store),  # noqa: B008
    graph_store: Any | None = Depends(get_graph_store),  # noqa: B008
    llm: Any | None = Depends(get_rag_llm),  # noqa: B008
    metrics: Metrics = Depends(get_metrics),  # noqa: B008
) -> AskResponse:
    """Route to graph or vector retrieval, then generate a grounded answer or degrade."""
    decision = route(req.question)
    target = (
        decision.target if (decision.target == "graph" and graph_store is not None) else "vector"
    )
    metrics.router_decision_total.labels(target=target).inc()
    if target == "graph":
        assert graph_store is not None  # target == "graph" implies a graph store is present
        start = time.perf_counter()
        try:
            citations = run_graph_query(graph_store, decision, settings)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Graph store unavailable."
            ) from exc
        metrics.graph_query_latency.observe(time.perf_counter() - start)
        return generate_or_degrade(req.question, citations, llm, req.contract_id)
    try:
        return answer_question(req.question, store, llm, settings, req.contract_id, req.top_k)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Vector store unavailable."
        ) from exc
