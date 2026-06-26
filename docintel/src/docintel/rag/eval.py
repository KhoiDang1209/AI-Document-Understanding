"""Pure retrieval-quality metrics for the C2 vector index (logged to MLflow by the
eval notebook). RAGAS answer-quality metrics need a live LLM judge and run separately.
"""

from __future__ import annotations


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant ids found within the top-k retrieved ids."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Reciprocal rank of the first relevant id (0.0 if none retrieved)."""
    for rank, identifier in enumerate(retrieved_ids, start=1):
        if identifier in relevant_ids:
            return 1.0 / rank
    return 0.0
