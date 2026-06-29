"""C3 GraphRAG eval: multi-hop date-routing accuracy over the CUAD gold graph.

Builds the date-centric graph from CUAD gold clauses (all contracts, via the real
``build_graph_contract`` normalizer), then runs a set of diverse natural-language date
questions through the rule-based router + graph query. Expected contract sets are computed
independently from the gold dates, so accuracy reflects router + template correctness
(not a tautology against the store). No services required: the in-memory store is exact.

Reproduce::

    python -m docintel.scripts.eval_graph
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import date, timedelta
from typing import Any

from docintel.config import get_settings
from docintel.contracts.schema import ContractDocument, ExtractedClause
from docintel.graph.eval import evaluate_multihop
from docintel.graph.normalize import build_graph_contract
from docintel.graph.store import InMemoryGraphStore

_CATEGORY = re.compile(r'related to "([^"]+)"')
_REFERENCE_DATE = date(2019, 1, 1)


def _category(question: str) -> str | None:
    match = _CATEGORY.search(question)
    return match.group(1) if match else None


def _gold_clauses(dataset: Any) -> dict[str, dict[str, str]]:
    """title -> {category -> first gold answer text} for the date-relevant categories."""
    wanted = {"Expiration Date", "Renewal Term"}
    gold: dict[str, dict[str, str]] = defaultdict(dict)
    for ex in dataset:
        category = _category(ex["question"])
        if category not in wanted:
            continue
        answers = ex["answers"]["text"]
        if answers and category not in gold[ex["title"]]:
            gold[ex["title"]][category] = answers[0]
    return gold


def build_gold_graph(dataset: Any) -> tuple[InMemoryGraphStore, dict[str, tuple[str, bool]]]:
    """Build the in-memory graph and a parallel {title -> (iso_date, has_renewal)} projection."""
    store = InMemoryGraphStore()
    projection: dict[str, tuple[str, bool]] = {}
    for title, cats in _gold_clauses(dataset).items():
        clauses: list[ExtractedClause] = []
        if "Expiration Date" in cats:
            clauses.append(
                ExtractedClause(
                    clause_type="Expiration Date",
                    answer_text=cats["Expiration Date"],
                    char_start=0,
                    char_end=len(cats["Expiration Date"]),
                    confidence=1.0,
                )
            )
        if "Renewal Term" in cats:
            clauses.append(
                ExtractedClause(
                    clause_type="Renewal Term",
                    answer_text=cats["Renewal Term"],
                    char_start=0,
                    char_end=len(cats["Renewal Term"]),
                    confidence=1.0,
                )
            )
        doc = ContractDocument(
            id=title, source="digital", clauses=clauses, derived={}, page_count=1, created_at="t"
        )
        gc = build_graph_contract(doc)
        store.upsert_contract(gc)
        if gc.expiration is not None:
            projection[title] = (gc.expiration.iso_date, gc.renewal is not None)
    return store, projection


def _expected(
    projection: dict[str, tuple[str, bool]], within_days: int, auto_renew: bool
) -> set[str]:
    """Contracts expiring in [ref, ref+within_days], optionally requiring a renewal clause."""
    upper = (_REFERENCE_DATE + timedelta(days=within_days)).isoformat()
    lower = _REFERENCE_DATE.isoformat()
    out = set()
    for title, (iso, has_renewal) in projection.items():
        if lower <= iso <= upper and (not auto_renew or has_renewal):
            out.add(title)
    return out


def build_cases(projection: dict[str, tuple[str, bool]]) -> list[tuple[str, set[str]]]:
    """Diverse NL date questions paired with independently-computed expected contract sets."""
    default_days = get_settings().graph_default_within_days
    # (question, within_days_used, auto_renew) — phrasings stress within_days + auto-renew parsing.
    specs: list[tuple[str, int, bool]] = [
        ("Which contracts expire within 365 days?", 365, False),
        ("List the agreements expiring within 730 days.", 730, False),
        ("Which contracts will expire within 1095 days?", 1095, False),
        ("Show contracts expiring within 1825 days.", 1825, False),
        ("which contracts are expiring soon?", default_days, False),  # no N -> default
        ("What agreements expire in the next 1460 days?", 1460, False),
        ("Which auto-renewing contracts expire within 730 days?", 730, True),
        ("List auto-renew contracts expiring within 1095 days.", 1095, True),
        ("Which contracts that renew expire within 1825 days?", 1825, True),
        ("auto-renewing agreements expiring within 3650 days?", 3650, True),
    ]
    return [(q, _expected(projection, n, ar)) for (q, n, ar) in specs]


def run() -> dict[str, Any]:
    """Build the gold graph and evaluate multi-hop date routing."""
    from datasets import load_dataset

    settings = get_settings()
    dataset = load_dataset("theatticusproject/cuad-qa", split="train", trust_remote_code=True)
    store, projection = build_gold_graph(dataset)
    cases = build_cases(projection)
    metrics = evaluate_multihop(store, cases, settings, reference_date=_REFERENCE_DATE)

    per_case = []
    for question, expected in cases:
        from docintel.graph.query import run_graph_query
        from docintel.graph.router import route

        decision = route(question)
        got = {
            c.contract_id
            for c in run_graph_query(store, decision, settings, reference_date=_REFERENCE_DATE)
        }
        per_case.append(
            {
                "question": question,
                "template": decision.template,
                "within_days": decision.within_days,
                "expected": len(expected),
                "got": len(got),
                "correct": got == expected,
            }
        )

    return {
        "reference_date": _REFERENCE_DATE.isoformat(),
        "contracts_with_parsable_expiration": len(projection),
        "contracts_with_renewal_in_graph": sum(1 for _, r in projection.values() if r),
        "multihop_accuracy": metrics["multihop_accuracy"],
        "n_cases": int(metrics["n"]),
        "per_case": per_case,
    }


def main() -> None:
    """CLI entrypoint."""
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
