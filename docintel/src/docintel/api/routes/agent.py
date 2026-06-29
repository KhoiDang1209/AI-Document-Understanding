"""The /agent endpoint: run the LangGraph agent over the C1-C3 tools."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from docintel.agent.graph import AgentDeps, run_agent
from docintel.agent.schema import AgentRequest, AgentResponse
from docintel.agent.trace import build_tracer
from docintel.api.metrics import Metrics
from docintel.api.routes.ask import get_graph_store, get_rag_llm, get_rag_store
from docintel.api.routes.extract import get_metrics
from docintel.config import Settings, get_settings

router = APIRouter(tags=["agent"])


def get_agent_tracer(
    request: Request,
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> Any | None:
    """Build the Langfuse tracer once and cache it (None when not configured)."""
    tracer = getattr(request.app.state, "agent_tracer", None)
    if tracer is None:
        tracer = build_tracer(settings)
        request.app.state.agent_tracer = tracer
    return tracer


@router.post("/agent", response_model=AgentResponse, summary="Run the contract agent on a task")
def agent(
    req: AgentRequest,
    settings: Settings = Depends(get_settings),  # noqa: B008
    rag_store: Any = Depends(get_rag_store),  # noqa: B008
    graph_store: Any | None = Depends(get_graph_store),  # noqa: B008
    llm: Any | None = Depends(get_rag_llm),  # noqa: B008
    tracer: Any | None = Depends(get_agent_tracer),  # noqa: B008
    metrics: Metrics = Depends(get_metrics),  # noqa: B008
) -> AgentResponse:
    """Run the agent graph for a compound task and return the grounded (or degraded) result."""
    deps = AgentDeps(
        settings=settings, rag_store=rag_store, graph_store=graph_store, llm=llm, tracer=tracer
    )
    response = run_agent(req.task, req.contract_id, deps)
    metrics.agent_run_total.labels(status=response.status).inc()
    if response.retries:
        metrics.agent_retries.inc(response.retries)
    metrics.agent_steps.observe(len(response.steps))
    return response
