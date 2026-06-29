"""LangGraph state machine that orchestrates the C1-C3 tools.

The graph is deliberately explicit (not a free-form ReAct loop): route -> retrieve ->
generate -> critique, with one bounded retry edge. This keeps every step inspectable and
traceable, and lets the whole flow run deterministically (and testably) without an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, StateGraph

from docintel.agent.schema import AgentResponse, AgentState
from docintel.agent.tools import generate_tool, graph_query_tool, vector_retrieve_tool
from docintel.agent.trace import trace_id_of
from docintel.config import Settings


@dataclass(frozen=True)
class AgentDeps:
    """Dependencies injected into the graph (stores, llm, tracer)."""

    settings: Settings
    rag_store: Any
    graph_store: Any | None
    llm: Any | None
    tracer: Any | None = None


def _route_node(state: AgentState, deps: AgentDeps) -> AgentState:
    from docintel.graph.router import route

    target = route(state["task"]).target
    return {"route_target": target, "steps": [f"route:{target}"]}


def _retrieve_node(state: AgentState, deps: AgentDeps) -> AgentState:
    use_graph = state.get("route_target") == "graph" and not state.get("fallback", False)
    if use_graph:
        citations = graph_query_tool(state["task"], deps.graph_store, deps.settings)
        step = "retrieve:graph"
    else:
        citations = vector_retrieve_tool(
            state["task"], deps.rag_store, deps.settings, state.get("contract_id")
        )
        step = "retrieve:vector"
    return {"citations": citations, "steps": [f"{step}:{len(citations)}"]}


def _generate_node(state: AgentState, deps: AgentDeps) -> AgentState:
    # The tracer is propagated to this node's LLM call via the graph-level invoke config
    # (run_agent passes callbacks once); generate_tool reuses C2's generate_or_degrade.
    resp = generate_tool(
        state["task"], state.get("citations", []), deps.llm, state.get("contract_id")
    )
    return {
        "answer": resp.answer,
        "generation_skipped": resp.generation_skipped,
        "steps": ["generate"],
    }


def _critique_node(state: AgentState, deps: AgentDeps) -> AgentState:
    # Decide retry once, per iteration. do_retry is the single source of truth for the edge,
    # so the loop cannot continue after the cap is reached (fallback alone would never clear).
    no_citations = not state.get("citations")
    can_retry = state.get("retries", 0) < deps.settings.agent_max_retries
    if no_citations and can_retry:
        return {
            "do_retry": True,
            "fallback": True,
            "retries": state.get("retries", 0) + 1,
            "steps": ["critique:retry"],
        }
    return {"do_retry": False, "steps": ["critique:finish"]}


def _should_retry(state: AgentState, deps: AgentDeps) -> str:
    return "retrieve" if state.get("do_retry") else "end"


def build_agent_graph(deps: AgentDeps) -> Any:
    """Compile the route -> retrieve -> generate -> critique graph with a bounded retry."""

    def bind(fn: Any) -> Any:
        return lambda state: fn(state, deps)

    graph = StateGraph(AgentState)
    graph.add_node("route", bind(_route_node))
    graph.add_node("retrieve", bind(_retrieve_node))
    graph.add_node("generate", bind(_generate_node))
    graph.add_node("critique", bind(_critique_node))
    graph.set_entry_point("route")
    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "critique")
    graph.add_conditional_edges(
        "critique", lambda state: _should_retry(state, deps), {"retrieve": "retrieve", "end": END}
    )
    return graph.compile()


def run_agent(task: str, contract_id: str | None, deps: AgentDeps) -> AgentResponse:
    """Run the compiled graph for one task and map the final state to AgentResponse."""
    compiled = build_agent_graph(deps)
    initial: AgentState = {"task": task, "contract_id": contract_id, "retries": 0, "steps": []}
    # Pass the tracer once at graph level; LangChain callback propagation reaches the node LLM call.
    config = {"callbacks": [deps.tracer]} if deps.tracer is not None else None
    final: AgentState = compiled.invoke(initial, config=config)
    answer = final.get("answer")
    skipped = final.get("generation_skipped", True)
    status = "ok" if (answer is not None and not skipped) else "degraded"
    return AgentResponse(
        task=task,
        answer=answer,
        status=status,
        contract_id=contract_id,
        trace_id=trace_id_of(deps.tracer),
        retries=final.get("retries", 0),
        citations=final.get("citations", []),
        steps=final.get("steps", []),
    )
