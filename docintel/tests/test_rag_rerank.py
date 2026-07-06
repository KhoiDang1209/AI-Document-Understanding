from __future__ import annotations

from docintel.config import Settings
from docintel.rag.rerank import ChunkReranker, build_reranker, rerank_chunks
from docintel.rag.schema import RetrievedChunk


class _FakeEncoder:
    """Scores each text by a fixed lookup, mimicking TextCrossEncoder.rerank."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        return [self._scores[text] for text in texts]


def _chunk(index: int, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        contract_id="c1",
        chunk_index=index,
        chunk_kind="paragraph",
        clause_type=None,
        text=text,
        score=0.0,
        char_start=0,
        char_end=len(text),
    )


def test_rerank_orders_by_cross_encoder_score_and_truncates() -> None:
    reranker = ChunkReranker(_FakeEncoder({"a": 0.1, "b": 0.9, "c": 0.5}))
    chunks = [_chunk(0, "a"), _chunk(1, "b"), _chunk(2, "c")]
    top = rerank_chunks(reranker, "query", chunks, top_k=2)
    assert [c.text for c in top] == ["b", "c"]
    assert [c.score for c in top] == [0.9, 0.5]


def test_rerank_empty_input_returns_empty() -> None:
    reranker = ChunkReranker(_FakeEncoder({}))
    assert rerank_chunks(reranker, "query", [], top_k=5) == []


def test_build_reranker_returns_none_when_model_unset() -> None:
    assert build_reranker(Settings(rag_rerank_model="")) is None
