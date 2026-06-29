"""Pydantic models for the C3 graph layer: normalized facts and the router decision."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ExpirationFact(BaseModel):
    """A contract's expiration date, normalized to ISO, with its citation span."""

    iso_date: str
    answer_text: str
    char_start: int
    char_end: int


class RenewalFact(BaseModel):
    """Presence of a renewal/auto-renewal clause, with its citation span."""

    answer_text: str
    char_start: int
    char_end: int


class GraphContract(BaseModel):
    """The minimal, normalized projection of a ContractDocument upserted into the graph."""

    contract_id: str
    expiration: ExpirationFact | None = None
    renewal: RenewalFact | None = None


class RouteDecision(BaseModel):
    """Where /ask should send a question, plus the chosen template and parameters."""

    target: Literal["graph", "vector"]
    template: str | None = None
    within_days: int | None = None
