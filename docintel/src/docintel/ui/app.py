"""Streamlit entrypoint: the Contract Intelligence demo (C1-C4).

Run with ``streamlit run src/docintel/ui/app.py``. The API base URL, request
timeout, and eval directory come from :class:`docintel.config.Settings`
(``DOCINTEL_`` env prefix). Six views walk the platform: Overview (architecture
+ live health), Extract (C1), Ask (C2/C3), Agent (C4), Graph (C3 evidence
network), and Metrics (committed eval JSON). Each API-backed view degrades
gracefully — the API returns citations-only when no LLM is configured, and this
page just reflects that.
"""

from __future__ import annotations

from typing import Any, cast

import streamlit as st

from docintel.config import Settings, get_settings
from docintel.ui.contract_client import (
    ContractApiError,
    ask_question,
    citation_rows,
    clause_rows,
    extract_contract,
    fetch_health,
    graph_dot,
    run_agent,
)
from docintel.ui.eval_report import (
    discover_eval_files,
    load_eval_file,
    ragas_rows,
    retrieval_rows,
)

# Architecture at a glance: the base document pipeline and the C1-C4 flow layered on top.
_ARCHITECTURE_DOT = """
digraph Arch {
  rankdir=LR;
  node [shape=box, style=rounded, fontsize=10];
  subgraph cluster_pipeline {
    label="Document pipeline";
    style=dashed;
    Upload -> Layout -> OCR -> KIE -> Validation;
  }
  subgraph cluster_contract {
    label="Contract Intelligence (C1-C4)";
    style=dashed;
    Ingest -> "C1 Extract" -> "C2 Vector RAG";
    "C1 Extract" -> "C3 GraphRAG";
    "C2 Vector RAG" -> "C4 Agent";
    "C3 GraphRAG" -> "C4 Agent";
  }
  Validation -> Ingest [style=dotted, label="reuse"];
}
"""


def _scope_id() -> str | None:
    """Return the session's extracted contract id, if any."""
    return cast("str | None", st.session_state.get("contract_id"))


def _render_citations(response: dict[str, Any]) -> None:
    rows = citation_rows(response)
    if rows:
        st.subheader("Citations")
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No citations returned.")


def _overview_tab(settings: Settings) -> None:
    st.caption(
        "A production-oriented Contract Intelligence platform: extract clauses (C1), "
        "answer questions over them with vector RAG (C2) and a Neo4j knowledge graph (C3), "
        "and orchestrate the tools with a LangGraph agent (C4)."
    )
    st.graphviz_chart(_ARCHITECTURE_DOT, use_container_width=True)

    st.subheader("Live status")
    try:
        health = fetch_health(settings.ui_api_base_url, settings.ui_request_timeout_s)
        cols = st.columns(4)
        cols[0].metric("API", health.get("status", "—"))
        cols[1].metric("Service", health.get("service", "—"))
        cols[2].metric("Version", health.get("version", "—"))
        cols[3].metric("Environment", health.get("environment", "—"))
    except ContractApiError as exc:
        st.error(f"API unreachable at {settings.ui_api_base_url}: {exc}")

    st.subheader("Configured backends")
    st.dataframe(
        [
            {"Backend": "API", "Target": settings.ui_api_base_url},
            {"Backend": "Qdrant (C2)", "Target": settings.qdrant_url},
            {"Backend": "Neo4j (C3)", "Target": settings.neo4j_uri},
            {
                "Backend": "LLM",
                "Target": settings.llm_base_url or "not configured (degrades to citations)",
            },
            {"Backend": "Graph enabled", "Target": str(settings.graph_enabled)},
            {"Backend": "Agent enabled", "Target": str(settings.agent_enabled)},
        ],
        use_container_width=True,
        hide_index=True,
    )


def _extract_tab(base_url: str, timeout_s: float) -> None:
    st.caption("Upload a contract PDF to extract structured clauses (C1).")
    uploaded = st.file_uploader("Contract PDF", type=["pdf"])
    if uploaded is None:
        return
    if st.button("Extract", key="extract_btn"):
        with st.spinner("Extracting…"):
            try:
                doc = extract_contract(base_url, timeout_s, uploaded.name, uploaded.getvalue())
            except ContractApiError as exc:
                st.error(str(exc))
                return
        st.session_state["contract_id"] = doc["id"]
        st.success(f"Extracted contract {doc['id']} (source: {doc['source']}).")
        rows = clause_rows(doc)
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("No clauses detected.")
        with st.expander("Derived fields"):
            st.json(doc.get("derived", {}))
        with st.expander("Raw JSON"):
            st.json(doc)


def _ask_tab(base_url: str, timeout_s: float) -> None:
    st.caption("Ask a question grounded in the indexed contracts (C2 vector / C3 graph).")
    question = st.text_input("Question", key="ask_q")
    scoped = st.checkbox("Only the last extracted contract", value=True, key="ask_scope")
    if st.button("Ask", key="ask_btn") and question:
        with st.spinner("Answering…"):
            try:
                resp = ask_question(base_url, timeout_s, question, _scope_id() if scoped else None)
            except ContractApiError as exc:
                st.error(str(exc))
                return
        if resp.get("generation_skipped"):
            st.warning("Generation skipped (no LLM configured) — showing citations only.")
        elif resp.get("answer"):
            st.success(resp["answer"])
        _render_citations(resp)


def _agent_tab(base_url: str, timeout_s: float) -> None:
    st.caption("Run the LangGraph agent over a compound task (C4).")
    task = st.text_input("Task", key="agent_task")
    scoped = st.checkbox("Only the last extracted contract", value=True, key="agent_scope")
    if st.button("Run agent", key="agent_btn") and task:
        with st.spinner("Running agent…"):
            try:
                resp = run_agent(base_url, timeout_s, task, _scope_id() if scoped else None)
            except ContractApiError as exc:
                st.error(str(exc))
                return
        status = resp.get("status")
        if status == "ok" and resp.get("answer"):
            st.success(resp["answer"])
        else:
            st.warning("Agent degraded (no grounded answer) — showing citations only.")
        cols = st.columns(3)
        cols[0].metric("Status", str(status))
        cols[1].metric("Retries", resp.get("retries", 0))
        cols[2].metric("Steps", len(resp.get("steps", [])))
        with st.expander("Steps"):
            st.write(resp.get("steps", []))
        if resp.get("trace_id"):
            st.caption(f"Langfuse trace: {resp['trace_id']}")
        _render_citations(resp)


def _graph_tab(base_url: str, timeout_s: float) -> None:
    st.caption(
        "Graph-routed questions (expiration/renewal within a window) hit the Neo4j knowledge "
        "graph (C3). The evidence comes back as cited facts — rendered here as a contract network."
    )
    question = st.text_input(
        "Graph question",
        value="Which contracts expire within 90 days?",
        key="graph_q",
    )
    if st.button("Query graph", key="graph_btn") and question:
        with st.spinner("Querying the graph…"):
            try:
                resp = ask_question(base_url, timeout_s, question, None)
            except ContractApiError as exc:
                st.error(str(exc))
                return
        if resp.get("citations"):
            st.graphviz_chart(graph_dot(resp), use_container_width=True)
        else:
            st.info("No matching contracts in the graph for that window.")
        _render_citations(resp)


def _metrics_tab(settings: Settings) -> None:
    st.caption(
        f"Committed evaluation results from `{settings.ui_eval_dir}` — C2 retrieval recall@k / MRR "
        "and RAGAS faithfulness / answer-relevancy. Provenance (embedder identity) is read from "
        "each run's recorded config."
    )
    files = discover_eval_files(settings.ui_eval_dir)
    if not files:
        st.info(
            "No eval JSON found. Generate some with:\n\n"
            "```\npython -m docintel.scripts.eval_rag --sample 40 --seed 0 --out eval_rag.json\n"
            "```"
        )
        return
    payloads = [load_eval_file(path) for path in files]

    retrieval = retrieval_rows(payloads)
    if retrieval:
        st.subheader("Retrieval (recall@k / MRR)")
        st.dataframe(retrieval, use_container_width=True, hide_index=True)

    ragas = ragas_rows(payloads)
    if ragas:
        st.subheader("RAGAS (faithfulness / answer relevancy)")
        st.dataframe(ragas, use_container_width=True, hide_index=True)

    with st.expander("Per-category recall (per run)"):
        for payload in payloads:
            by_cat = payload.get("recall_at_max_k_by_category")
            if by_cat:
                st.markdown(f"**{payload.get('_name')}**")
                st.dataframe(
                    [{"Category": k, "recall@max_k": v} for k, v in by_cat.items()],
                    use_container_width=True,
                    hide_index=True,
                )


def main() -> None:
    """Build the Streamlit page: Overview / Extract / Ask / Agent / Graph / Metrics."""
    settings = get_settings()
    st.set_page_config(page_title="DocIntel — Contract Intelligence", layout="wide")
    st.title("Contract Intelligence")
    st.caption(f"API: {settings.ui_api_base_url}")

    base_url = settings.ui_api_base_url
    timeout_s = settings.ui_request_timeout_s
    overview, extract, ask, agent, graph, metrics = st.tabs(
        ["Overview", "Extract", "Ask", "Agent", "Graph", "Metrics"]
    )
    with overview:
        _overview_tab(settings)
    with extract:
        _extract_tab(base_url, timeout_s)
    with ask:
        _ask_tab(base_url, timeout_s)
    with agent:
        _agent_tab(base_url, timeout_s)
    with graph:
        _graph_tab(base_url, timeout_s)
    with metrics:
        _metrics_tab(settings)


if __name__ == "__main__":
    main()
