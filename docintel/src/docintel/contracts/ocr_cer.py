"""Character Error Rate for the OCR ingestion path."""

from __future__ import annotations

from docintel.contracts.eval import _levenshtein


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate = edit distance / len(reference); 0.0 when both empty."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _levenshtein(reference, hypothesis) / len(reference)
