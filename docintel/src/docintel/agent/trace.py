"""Self-hosted Langfuse tracing for the agent - strictly best-effort.

A tracing failure (no keys, unreachable Langfuse, library error) must never break
/agent: build_tracer returns None and the graph runs untraced. The callback handler,
when present, is passed to LangChain via config={"callbacks": [tracer]}.
"""

from __future__ import annotations

import logging
from typing import Any

from docintel.config import Settings

logger = logging.getLogger("docintel.agent.trace")


def build_tracer(settings: Settings) -> Any | None:
    """Return a Langfuse CallbackHandler when configured, else None (never raises)."""
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None
    try:
        from langfuse.callback import CallbackHandler

        return CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except Exception:
        logger.warning("agent.trace.unavailable", exc_info=True)
        return None


def trace_id_of(tracer: Any | None) -> str | None:
    """Best-effort read of the handler's last trace id; None on absence or error."""
    if tracer is None:
        return None
    try:
        getter = getattr(tracer, "get_trace_id", None)
        if callable(getter):
            return getter() or None
        value = getattr(tracer, "last_trace_id", None)
        return str(value) if value else None
    except Exception:
        return None
