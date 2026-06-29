"""Run a routed template against a GraphStore and return cited RetrievedChunks.

Date bounds are computed in Python (reference_date .. reference_date + N days) and passed as
ISO strings, keeping the Cypher a pure string comparison. Citations reuse C2's RetrievedChunk
shape so /ask returns one consistent citation contract across vector and graph answers.
"""

from __future__ import annotations

from datetime import date, timedelta

from docintel.config import Settings
from docintel.graph.normalize import EXPIRATION_CLAUSE, RENEWAL_CLAUSE
from docintel.graph.schema import RouteDecision
from docintel.graph.store import GraphStore
from docintel.rag.schema import RetrievedChunk


def run_graph_query(
    store: GraphStore,
    decision: RouteDecision,
    settings: Settings,
    reference_date: date | None = None,
) -> list[RetrievedChunk]:
    """Execute the routed template and map result rows to cited RetrievedChunks.

    Graph templates are corpus-wide by design, so any per-question ``contract_id`` is not
    applied here (the two date templates answer cross-contract questions).
    """
    assert decision.template is not None  # router guarantees this for target == "graph"
    start = reference_date or date.today()
    within = decision.within_days or settings.graph_default_within_days
    params = {"lower": start.isoformat(), "upper": (start + timedelta(days=within)).isoformat()}
    rows = store.run_template(decision.template, params)

    chunks: list[RetrievedChunk] = []
    for row in rows:
        chunks.append(
            RetrievedChunk(
                contract_id=row["contract_id"],
                chunk_index=0,
                chunk_kind="graph",
                clause_type=EXPIRATION_CLAUSE,
                text=row["exp_answer"],
                score=1.0,
                char_start=row["exp_start"],
                char_end=row["exp_end"],
            )
        )
        if "ren_answer" in row:
            chunks.append(
                RetrievedChunk(
                    contract_id=row["contract_id"],
                    chunk_index=0,
                    chunk_kind="graph",
                    clause_type=RENEWAL_CLAUSE,
                    text=row["ren_answer"],
                    score=1.0,
                    char_start=row["ren_start"],
                    char_end=row["ren_end"],
                )
            )
    return chunks
