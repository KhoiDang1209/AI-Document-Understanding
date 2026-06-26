"""Pydantic models for the /ask request/response and retrieved citations."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """One retrieved chunk used as a grounded citation."""

    contract_id: str
    chunk_index: int
    chunk_kind: str
    clause_type: str | None
    text: str
    score: float
    char_start: int
    char_end: int


class AskRequest(BaseModel):
    """A natural-language question, optionally scoped to one contract."""

    question: str
    contract_id: str | None = None
    top_k: int | None = None


class AskResponse(BaseModel):
    """The grounded answer (or null when degraded) plus its citations."""

    question: str
    answer: str | None
    generation_skipped: bool
    contract_id: str | None
    citations: list[RetrievedChunk] = Field(default_factory=list)
