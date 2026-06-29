"""Pure rule-based router: classify a question to a graph template or the vector path.

No LLM and no settings are consulted, so routing is deterministic and works in the degraded
(LLM-down) path. Parameter extraction is regex-only.
"""

from __future__ import annotations

import re

from docintel.graph.schema import RouteDecision

_EXPIRY = re.compile(r"\bexpir", re.IGNORECASE)
_RENEWAL = re.compile(r"\b(?:auto[\s-]?renew|renew)", re.IGNORECASE)
_WITHIN_DAYS = re.compile(r"(\d+)\s*day", re.IGNORECASE)


def route(question: str) -> RouteDecision:
    """Map a question to {graph template | vector}, extracting an N-day window if present."""
    if not _EXPIRY.search(question):
        return RouteDecision(target="vector")
    days_match = _WITHIN_DAYS.search(question)
    within_days = int(days_match.group(1)) if days_match else None
    template = "auto_renewing_expiring_within" if _RENEWAL.search(question) else "expiring_within"
    return RouteDecision(target="graph", template=template, within_days=within_days)
