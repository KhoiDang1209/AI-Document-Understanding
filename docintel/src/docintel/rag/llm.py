"""LLM client (OpenAI-compatible) + grounded prompt for /ask.

``build_llm`` returns ``None`` when no endpoint is configured, which drives the
graceful-degrade path. The same ``ChatOpenAI`` points at the Colab/ngrok server now
and the OpenAI API later — only settings change. ``langchain_openai`` is imported lazily.
"""

from __future__ import annotations

from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from docintel.config import Settings
from docintel.rag.schema import RetrievedChunk

# Grounding instruction (reference text, not a tunable knob).
RAG_SYSTEM_PROMPT = (
    "You are a contract analysis assistant. Answer the question using ONLY the provided "
    "context. Cite the clause types you relied on. If the context does not contain the "
    "answer, say you do not have enough information."
)


def build_llm(settings: Settings) -> Any | None:
    """Return an OpenAI-compatible chat model, or None when no endpoint is configured."""
    if not settings.llm_base_url:
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key or "EMPTY",
        model=settings.llm_model,
        timeout=settings.llm_timeout_s,
        temperature=0,
    )


def build_prompt() -> ChatPromptTemplate:
    """Grounded chat prompt with {question} and {context} slots."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            ("human", "Question: {question}\n\nContext:\n{context}"),
        ]
    )


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as a numbered, labelled context block."""
    lines = []
    for position, chunk in enumerate(chunks, start=1):
        label = chunk.clause_type or "Excerpt"
        lines.append(f"[{position}] ({label}, contract {chunk.contract_id}): {chunk.text}")
    return "\n".join(lines)
