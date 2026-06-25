"""Structured contract record schema for the C1 extraction path."""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, Field


class ExtractedClause(BaseModel):
    """One extracted clause span with its char offsets into the ingested text."""

    clause_type: str
    answer_text: str
    char_start: int
    char_end: int
    confidence: float


class ContractDocument(BaseModel):
    """Structured extraction result for one contract."""

    id: str
    source: Literal["digital", "ocr"]
    clauses: list[ExtractedClause] = Field(default_factory=list)
    derived: dict[str, list[str]] = Field(default_factory=dict)
    page_count: int
    created_at: str


def build_derived(clauses: list[ExtractedClause]) -> dict[str, list[str]]:
    """Group clause answer texts by clause type, preserving extraction order."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for clause in clauses:
        grouped[clause.clause_type].append(clause.answer_text)
    return dict(grouped)
