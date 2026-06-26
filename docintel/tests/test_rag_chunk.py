from __future__ import annotations

from docintel.contracts.schema import ExtractedClause
from docintel.rag.chunk import TextChunk, build_chunks


def _clause() -> ExtractedClause:
    return ExtractedClause(
        clause_type="Governing Law",
        answer_text="New York",
        char_start=0,
        char_end=8,
        confidence=0.9,
    )


def test_clause_chunks_come_first_and_carry_type() -> None:
    chunks = build_chunks("New York law applies here.", [_clause()], size=10, overlap=2)
    assert chunks[0] == TextChunk(
        text="New York",
        char_start=0,
        char_end=8,
        chunk_index=0,
        chunk_kind="clause",
        clause_type="Governing Law",
    )


def test_paragraph_chunks_cover_text_with_unique_indices() -> None:
    chunks = build_chunks("abcdefghij", [], size=4, overlap=1)
    para = [c for c in chunks if c.chunk_kind == "paragraph"]
    assert [c.text for c in para] == ["abcd", "defg", "ghij", "j"]
    assert [c.chunk_index for c in para] == [0, 1, 2, 3]
    assert para[0].clause_type is None


def test_indices_are_sequential_across_both_kinds() -> None:
    chunks = build_chunks("abcdefghij", [_clause()], size=4, overlap=1)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].chunk_kind == "clause"
    assert chunks[1].chunk_kind == "paragraph"


def test_blank_clause_is_skipped() -> None:
    blank = ExtractedClause(
        clause_type="X", answer_text="   ", char_start=0, char_end=3, confidence=0.1
    )
    chunks = build_chunks("abcd", [blank], size=4, overlap=1)
    assert all(c.chunk_kind == "paragraph" for c in chunks)
