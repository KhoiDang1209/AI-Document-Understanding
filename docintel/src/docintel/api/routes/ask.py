"""The /ask endpoint: retrieve cited chunks and generate-or-degrade an answer."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from docintel.config import Settings, get_settings
from docintel.rag.answer import answer_question
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


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question grounded in indexed contracts",
)
def ask(
    req: AskRequest,
    settings: Settings = Depends(get_settings),  # noqa: B008
    store: Any = Depends(get_rag_store),  # noqa: B008
    llm: Any | None = Depends(get_rag_llm),  # noqa: B008
) -> AskResponse:
    """Retrieve cited chunks and answer; degrade to citations when no LLM is reachable."""
    try:
        return answer_question(req.question, store, llm, settings, req.contract_id, req.top_k)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Vector store unavailable."
        ) from exc
