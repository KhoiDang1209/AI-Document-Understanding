"""Streamlit entrypoint: the Contract Intelligence page (C1-C4).

Run with ``streamlit run src/docintel/ui/app.py``. The API base URL and request
timeout come from :class:`docintel.config.Settings` (``DOCINTEL_`` env prefix).
Three tabs walk the pipeline: Extract (C1), Ask (C2/C3), Agent (C4). Each tab
calls the running API and renders its result; the API degrades gracefully to
citations-only when no LLM is configured, and this page just reflects that.
"""

from __future__ import annotations

from typing import Any, cast

import streamlit as st

from docintel.config import get_settings
from docintel.ui.contract_client import (
    ContractApiError,
    ask_question,
    citation_rows,
    clause_rows,
    extract_contract,
    run_agent,
)


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
                resp = ask_question(
                    base_url, timeout_s, question, _scope_id() if scoped else None
                )
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


def main() -> None:
    """Build the Streamlit page with Extract / Ask / Agent tabs."""
    settings = get_settings()
    st.set_page_config(page_title="DocIntel — Contract Intelligence", layout="wide")
    st.title("Contract Intelligence")
    st.caption(f"API: {settings.ui_api_base_url}")

    base_url = settings.ui_api_base_url
    timeout_s = settings.ui_request_timeout_s
    extract, ask, agent = st.tabs(["Extract", "Ask", "Agent"])
    with extract:
        _extract_tab(base_url, timeout_s)
    with ask:
        _ask_tab(base_url, timeout_s)
    with agent:
        _agent_tab(base_url, timeout_s)


if __name__ == "__main__":
    main()
