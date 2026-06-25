"""Turn per-window QA logits into char-offset clause spans (SQuAD-style decode)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from docintel.contracts.schema import ExtractedClause


@dataclass(frozen=True)
class WindowSpan:
    """A candidate answer span in document char offsets with its logit score."""

    start_char: int
    end_char: int
    score: float


def best_spans_from_window(
    start_logits: Any,
    end_logits: Any,
    offset_mapping: Sequence[tuple[int, int]],
    n_best: int,
    max_answer_length: int,
) -> list[WindowSpan]:
    """Return the top-``n_best`` (start, end) spans for one tokenized window.

    Tokens whose offset is ``(0, 0)`` (special/question tokens) are skipped as
    span endpoints. Scores are ``start_logit + end_logit``.
    """
    start = np.asarray(start_logits)
    end = np.asarray(end_logits)
    starts = np.argsort(start)[::-1][:n_best]
    ends = np.argsort(end)[::-1][:n_best]
    spans: list[WindowSpan] = []
    for s in starts:
        for e in ends:
            if e < s or (e - s + 1) > max_answer_length:
                continue
            s_off = offset_mapping[int(s)]
            e_off = offset_mapping[int(e)]
            if s_off == (0, 0) or e_off == (0, 0):
                continue
            spans.append(
                WindowSpan(
                    start_char=int(s_off[0]),
                    end_char=int(e_off[1]),
                    score=float(start[s] + end[e]),
                )
            )
    spans.sort(key=lambda span: span.score, reverse=True)
    return spans[:n_best]


def aggregate_clause(
    clause_type: str,
    text: str,
    windows: Sequence[WindowSpan],
    n_best: int,
    no_answer_threshold: float,
) -> list[ExtractedClause]:
    """Rank spans across windows; keep those above threshold, up to ``n_best``."""
    ranked = sorted(windows, key=lambda span: span.score, reverse=True)
    clauses: list[ExtractedClause] = []
    for span in ranked[:n_best]:
        if span.score < no_answer_threshold:
            continue
        clauses.append(
            ExtractedClause(
                clause_type=clause_type,
                answer_text=text[span.start_char : span.end_char],
                char_start=span.start_char,
                char_end=span.end_char,
                confidence=float(1.0 / (1.0 + np.exp(-span.score))),
            )
        )
    return clauses
