from __future__ import annotations

import numpy as np

from docintel.contracts.aggregate import (
    WindowSpan,
    aggregate_clause,
    best_spans_from_window,
)


def test_best_spans_picks_highest_scoring_valid_span() -> None:
    # 4 tokens; token 0 is the CLS/no-answer slot (offset (0,0)).
    start = np.array([0.1, 2.0, 0.0, 0.0])
    end = np.array([0.1, 0.0, 3.0, 0.0])
    offsets = [(0, 0), (0, 5), (6, 11), (12, 16)]
    spans = best_spans_from_window(start, end, offsets, n_best=3, max_answer_length=20)
    top = spans[0]
    assert (top.start_char, top.end_char) == (0, 11)
    assert top.score == 5.0


def test_aggregate_clause_applies_no_answer_threshold() -> None:
    text = "alpha beta gamma"
    windows = [WindowSpan(0, 5, score=4.0), WindowSpan(6, 10, score=-1.0)]
    clauses = aggregate_clause("Parties", text, windows, n_best=2, no_answer_threshold=0.0)
    assert len(clauses) == 1
    assert clauses[0].clause_type == "Parties"
    assert clauses[0].answer_text == "alpha"
    assert clauses[0].char_start == 0 and clauses[0].char_end == 5
