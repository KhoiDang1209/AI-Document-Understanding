"""Pure chunking of a contract into clause chunks + sliding paragraph chunks.

Clause chunks (one per ExtractedClause) are precise and citation-ready; paragraph
chunks are overlapping char windows over the full ingested text, giving coverage
for questions outside the 41 clause types. No model or I/O — fully CPU-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from docintel.contracts.schema import ExtractedClause


@dataclass(frozen=True)
class TextChunk:
    """One indexable chunk with its char offsets into the ingested text."""

    text: str
    char_start: int
    char_end: int
    chunk_index: int
    chunk_kind: Literal["clause", "paragraph"]
    clause_type: str | None


def _paragraph_chunks(text: str, size: int, overlap: int, start_index: int) -> list[TextChunk]:
    """Overlapping char windows over ``text``, indexed from ``start_index``."""
    step = max(size - overlap, 1)
    chunks: list[TextChunk] = []
    index = start_index
    position = 0
    length = len(text)
    while position < length:
        window = text[position : position + size]
        if window.strip():
            chunks.append(
                TextChunk(
                    text=window,
                    char_start=position,
                    char_end=min(position + size, length),
                    chunk_index=index,
                    chunk_kind="paragraph",
                    clause_type=None,
                )
            )
            index += 1
        position += step
    return chunks


def build_chunks(
    text: str, clauses: list[ExtractedClause], size: int, overlap: int
) -> list[TextChunk]:
    """Build clause chunks (first) then paragraph chunks, with unique sequential indices."""
    chunks: list[TextChunk] = []
    index = 0
    for clause in clauses:
        if not clause.answer_text.strip():
            continue
        chunks.append(
            TextChunk(
                text=clause.answer_text,
                char_start=clause.char_start,
                char_end=clause.char_end,
                chunk_index=index,
                chunk_kind="clause",
                clause_type=clause.clause_type,
            )
        )
        index += 1
    chunks.extend(_paragraph_chunks(text, size, overlap, index))
    return chunks
