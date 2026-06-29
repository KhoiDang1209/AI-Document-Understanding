"""Rule-based normalization: a ContractDocument's clauses → a minimal GraphContract.

Date parsing is intentionally rule-based (a small set of explicit formats); unparseable
expiration text is skipped, never blocking the build. Party/governing-law normalization is
out of scope for C3.
"""

from __future__ import annotations

import re
from datetime import datetime

from docintel.contracts.schema import ContractDocument, ExtractedClause
from docintel.graph.schema import ExpirationFact, GraphContract, RenewalFact

# Exact CUAD category names (reference data; see contracts/questions.py).
EXPIRATION_CLAUSE = "Expiration Date"
RENEWAL_CLAUSE = "Renewal Term"

# (compiled regex, strptime format) pairs tried in order. ISO is handled first verbatim.
_ISO = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_DATE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b((?:January|February|March|April|May|June|July|August|September|October"
            r"|November|December)\s+\d{1,2},\s+\d{4})\b"
        ),
        "%B %d, %Y",
    ),
    (re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b"), "%m/%d/%Y"),
)


def parse_iso_date(text: str) -> str | None:
    """Return the first date in ``text`` as ``YYYY-MM-DD``, or None if none parse."""
    iso = _ISO.search(text)
    if iso is not None:
        try:
            return datetime.strptime(iso.group(1), "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            pass
    for pattern, fmt in _DATE_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        try:
            return datetime.strptime(match.group(1), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _first(clauses: list[ExtractedClause], clause_type: str) -> ExtractedClause | None:
    return next((c for c in clauses if c.clause_type == clause_type), None)


def build_graph_contract(doc: ContractDocument) -> GraphContract:
    """Project a ContractDocument onto the minimal graph node set (expiration + renewal)."""
    expiration: ExpirationFact | None = None
    exp_clause = _first(doc.clauses, EXPIRATION_CLAUSE)
    if exp_clause is not None:
        iso = parse_iso_date(exp_clause.answer_text)
        if iso is not None:
            expiration = ExpirationFact(
                iso_date=iso,
                answer_text=exp_clause.answer_text,
                char_start=exp_clause.char_start,
                char_end=exp_clause.char_end,
            )

    renewal: RenewalFact | None = None
    ren_clause = _first(doc.clauses, RENEWAL_CLAUSE)
    if ren_clause is not None:
        renewal = RenewalFact(
            answer_text=ren_clause.answer_text,
            char_start=ren_clause.char_start,
            char_end=ren_clause.char_end,
        )

    return GraphContract(contract_id=doc.id, expiration=expiration, renewal=renewal)
