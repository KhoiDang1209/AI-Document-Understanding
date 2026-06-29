"""Pydantic request/response models and the LangGraph state for the C4 agent."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

from docintel.rag.schema import RetrievedChunk


class AgentRequest(BaseModel):
    """A compound natural-language task, optionally scoped to one contract."""

    task: str = Field(min_length=1)
    contract_id: str | None = None


class AgentResponse(BaseModel):
    """The agent's grounded answer (or null when degraded) plus citations and trace id."""

    task: str
    answer: str | None
    status: Literal["ok", "degraded"]
    contract_id: str | None
    trace_id: str | None
    retries: int
    citations: list[RetrievedChunk] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)


class AgentState(TypedDict, total=False):
    """Mutable state threaded through the LangGraph nodes."""

    task: str
    contract_id: str | None
    route_target: str
    citations: list[RetrievedChunk]
    answer: str | None
    generation_skipped: bool
    retries: int
    fallback: bool
    do_retry: bool
    steps: Annotated[list[str], operator.add]
