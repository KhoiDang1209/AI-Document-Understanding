"""Retrieve-then-(generate-or-degrade) orchestration for /ask."""

from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import StrOutputParser

from docintel.config import Settings
from docintel.rag.llm import build_prompt, format_context
from docintel.rag.schema import AskResponse
from docintel.rag.store import search


def generate_answer(llm: Any, question: str, context: str) -> str:
    """Run the LCEL chain prompt | llm | parser and return the answer text."""
    chain = build_prompt() | llm | StrOutputParser()
    return str(chain.invoke({"question": question, "context": context}))


def answer_question(
    question: str,
    store: Any,
    llm: Any | None,
    settings: Settings,
    contract_id: str | None = None,
    top_k: int | None = None,
) -> AskResponse:
    """Retrieve top-k chunks, then generate a grounded answer or degrade to citations."""
    citations = search(store, question, top_k or settings.rag_top_k, contract_id)
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
