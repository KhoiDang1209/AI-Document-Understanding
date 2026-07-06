"""Retrieve-then-(generate-or-degrade) orchestration for /ask."""

from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import StrOutputParser

from docintel.config import Settings
from docintel.rag.llm import build_prompt, format_context
from docintel.rag.rerank import ChunkReranker, rerank_chunks
from docintel.rag.schema import AskResponse, RetrievedChunk
from docintel.rag.store import search


def generate_answer(llm: Any, question: str, context: str) -> str:
    """Run the LCEL chain prompt | llm | parser and return the answer text."""
    chain = build_prompt() | llm | StrOutputParser()
    return str(chain.invoke({"question": question, "context": context}))


def generate_or_degrade(
    question: str,
    citations: list[RetrievedChunk],
    llm: Any | None,
    contract_id: str | None,
) -> AskResponse:
    """Generate a grounded answer from citations, or degrade to citations-only."""
    if llm is None:
        return AskResponse(
            question=question,
            answer=None,
            generation_skipped=True,
            contract_id=contract_id,
            citations=citations,
        )
    try:
        answer = generate_answer(llm, question, format_context(citations))
    except Exception:
        return AskResponse(
            question=question,
            answer=None,
            generation_skipped=True,
            contract_id=contract_id,
            citations=citations,
        )
    return AskResponse(
        question=question,
        answer=answer,
        generation_skipped=False,
        contract_id=contract_id,
        citations=citations,
    )


def retrieve_citations(
    store: Any,
    query: str,
    top_k: int,
    contract_id: str | None,
    settings: Settings,
    reranker: ChunkReranker | None = None,
) -> list[RetrievedChunk]:
    """Top-k hybrid search; with a reranker, rescore a wider candidate pool first."""
    if reranker is None:
        return search(store, query, top_k, contract_id)
    pool = max(settings.rag_rerank_candidates, top_k)
    candidates = search(store, query, pool, contract_id)
    return rerank_chunks(reranker, query, candidates, top_k)


def answer_question(
    question: str,
    store: Any,
    llm: Any | None,
    settings: Settings,
    contract_id: str | None = None,
    top_k: int | None = None,
    reranker: ChunkReranker | None = None,
    retrieval_query: str | None = None,
) -> AskResponse:
    """Retrieve top-k chunks, then generate a grounded answer or degrade to citations.

    ``retrieval_query`` overrides the text used for retrieval (e.g. a focused rewrite
    of a template question); the original ``question`` is still what the LLM answers.
    """
    citations = retrieve_citations(
        store,
        retrieval_query or question,
        top_k or settings.rag_top_k,
        contract_id,
        settings,
        reranker,
    )
    return generate_or_degrade(question, citations, llm, contract_id)
