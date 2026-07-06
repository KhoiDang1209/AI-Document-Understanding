"""Rewrite CUAD-template questions into focused retrieval queries.

The CUAD question template ("Highlight the parts (if any) of this contract related to
'X' ... Details: ...") is ~90% boilerplate shared by all 41 categories — a weak embedding
signal. ``focus_query`` keeps only the category name and the Details sentence; natural
questions pass through unchanged, so it is safe to apply to any retrieval query.
"""

from __future__ import annotations

import re

_CATEGORY = re.compile(r'related to "([^"]+)"')
_DETAILS = re.compile(r"Details:\s*(.+)\s*$", re.DOTALL)


def focus_query(question: str) -> str:
    """Return a focused retrieval query for CUAD-template questions, else the question."""
    category_match = _CATEGORY.search(question)
    if category_match is None:
        return question
    category = category_match.group(1)
    details_match = _DETAILS.search(question)
    if details_match is None:
        return category
    return f"{category}: {details_match.group(1).strip()}"
