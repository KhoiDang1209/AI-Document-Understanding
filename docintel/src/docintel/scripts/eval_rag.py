"""C2 retrieval-quality eval: within-contract recall@k / MRR over CUAD gold spans.

Indexes paragraph chunks (only) of a CUAD contract sample into an in-memory Qdrant,
then for each gold-answered clause question measures whether the passage covering the
gold answer is retrieved in the top-k. Clause chunks are intentionally excluded so the
metric reflects semantic passage retrieval, not trivial gold-span lookup. Retrieval is
the production stack: hybrid dense+BM25 search, focused query rewriting (see
``rag.query``), and cross-encoder reranking — each toggleable for ablation via
``--raw-query`` / ``--no-rerank``. LLM-dependent RAGAS metrics are out of scope here
(need the intermittent Colab judge); run separately.

Reproduce::

    python -m docintel.scripts.eval_rag --sample 40 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from typing import Any

from docintel.config import get_settings
from docintel.rag.answer import retrieve_citations
from docintel.rag.chunk import build_chunks
from docintel.rag.eval import mrr, recall_at_k
from docintel.rag.query import focus_query
from docintel.rag.rerank import build_reranker
from docintel.rag.store import build_vector_store, ensure_collection, upsert_chunks

_CATEGORY = re.compile(r'related to "([^"]+)"')


def _category(question: str) -> str | None:
    match = _CATEGORY.search(question)
    return match.group(1) if match else None


def _covering_chunk_indices(chunks: list[Any], answer_start: int) -> set[int]:
    """Indices of paragraph chunks whose char span contains ``answer_start``."""
    return {c.chunk_index for c in chunks if c.char_start <= answer_start < c.char_end}


def _parse_top_ks(raw: str) -> tuple[int, ...]:
    return tuple(int(part) for part in raw.split(","))


def _sample_contracts(dataset: Any, sample: int, seed: int) -> list[str]:
    titles = sorted({ex["title"] for ex in dataset})
    random.Random(seed).shuffle(titles)
    return titles[:sample]


def run(
    sample: int,
    seed: int,
    top_ks: tuple[int, ...] = (1, 3, 5),
    rerank: bool = True,
    focused: bool = True,
) -> dict[str, Any]:
    """Index a CUAD sample and return aggregate retrieval metrics."""
    from datasets import load_dataset

    settings = get_settings()
    reranker = build_reranker(settings) if rerank else None
    dataset = load_dataset("theatticusproject/cuad-qa", split="train", trust_remote_code=True)
    keep = set(_sample_contracts(dataset, sample, seed))

    # Group the kept contracts: full context once, plus each answered question's gold starts.
    contexts: dict[str, str] = {}
    queries: dict[str, list[tuple[str, str, list[int]]]] = defaultdict(list)
    for ex in dataset:
        title = ex["title"]
        if title not in keep:
            continue
        contexts.setdefault(title, ex["context"])
        starts = ex["answers"]["answer_start"]
        if starts:
            queries[title].append((_category(ex["question"]) or "?", ex["question"], list(starts)))

    # Index paragraph chunks of every kept contract into one in-memory Qdrant.
    from qdrant_client import QdrantClient

    from docintel.rag.embed import build_embedder

    client = QdrantClient(":memory:")
    ensure_collection(client, settings.qdrant_collection, settings.rag_embedding_dim)
    store = build_vector_store(settings, build_embedder(settings), client=client)
    chunks_by_contract: dict[str, list[Any]] = {}
    for title, text in contexts.items():
        chunks = build_chunks(text, [], settings.rag_chunk_size, settings.rag_chunk_overlap)
        chunks_by_contract[title] = chunks
        upsert_chunks(store, title, chunks)

    # Evaluate every gold-answered question against its own contract.
    max_k = max(top_ks)
    recalls: dict[int, list[float]] = {k: [] for k in top_ks}
    reciprocal: list[float] = []
    per_category: dict[str, list[float]] = defaultdict(list)
    evaluated = 0
    for title, items in queries.items():
        chunks = chunks_by_contract[title]
        for category, question, starts in items:
            relevant = set()
            for start in starts:
                relevant |= _covering_chunk_indices(chunks, start)
            if not relevant:
                continue
            query = focus_query(question) if focused else question
            results = retrieve_citations(store, query, max_k, title, settings, reranker)
            retrieved = [str(r.chunk_index) for r in results]
            relevant_ids = {str(i) for i in relevant}
            evaluated += 1
            for k in top_ks:
                recalls[k].append(recall_at_k(retrieved, relevant_ids, k))
            reciprocal.append(mrr(retrieved, relevant_ids))
            per_category[category].append(recall_at_k(retrieved, relevant_ids, max_k))

    def mean(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    return {
        "sample_contracts": len(contexts),
        "evaluated_queries": evaluated,
        "total_chunks": sum(len(c) for c in chunks_by_contract.values()),
        "recall_at_k": {str(k): mean(recalls[k]) for k in top_ks},
        "mrr": mean(reciprocal),
        "recall_at_max_k_by_category": {
            cat: mean(vals) for cat, vals in sorted(per_category.items())
        },
        "config": {
            "sample": sample,
            "seed": seed,
            "top_ks": list(top_ks),
            "rerank": rerank,
            "focused_query": focused,
            "rerank_model": settings.rag_rerank_model if rerank else None,
            "sparse_model": settings.rag_sparse_model,
        },
    }


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="C2 retrieval recall@k / MRR over CUAD.")
    parser.add_argument("--sample", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-rerank", action="store_true", help="Skip cross-encoder reranking.")
    parser.add_argument(
        "--raw-query", action="store_true", help="Use the raw CUAD template question."
    )
    parser.add_argument(
        "--top-ks", type=str, default="1,3,5", help="Comma-separated recall@k cutoffs."
    )
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    metrics = run(
        args.sample,
        args.seed,
        top_ks=_parse_top_ks(args.top_ks),
        rerank=not args.no_rerank,
        focused=not args.raw_query,
    )
    print(json.dumps(metrics, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)


if __name__ == "__main__":
    main()
