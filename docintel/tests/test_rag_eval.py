from __future__ import annotations

from docintel.rag.eval import mrr, recall_at_k


def test_recall_at_k_counts_relevant_in_top_k() -> None:
    assert recall_at_k(["a", "b", "c", "d"], {"b", "z"}, k=2) == 0.5  # b found, z not
    assert recall_at_k(["a", "b"], set(), k=2) == 0.0
    assert recall_at_k([], {"a"}, k=3) == 0.0


def test_mrr_uses_first_relevant_rank() -> None:
    assert mrr(["a", "b", "c"], {"b"}) == 0.5  # rank 2
    assert mrr(["a", "b"], {"a"}) == 1.0
    assert mrr(["a", "b"], {"z"}) == 0.0
