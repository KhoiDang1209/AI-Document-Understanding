"""C2/C4 answer-quality eval: RAGAS faithfulness + answer_relevancy over /ask.

Runs the *real* retrieve-then-generate pipeline (``answer_question``) over a small CUAD
sample using the live LLM, then scores each generated answer with RAGAS. Both metrics are
reference-free — they need only (question, answer, retrieved contexts) — so no gold labels
are required. The LLM judge is the same OpenAI-compatible endpoint the API uses
(``DOCINTEL_LLM_BASE_URL``); answer_relevancy embeddings reuse the local fastembed model.

Retrieval matches production: hybrid dense+BM25 search, focused query rewriting, and
cross-encoder reranking, over clause chunks (real C1 extractor, when its model bundle is
available) plus paragraph chunks. Because ``answer_relevancy`` hard-zeros noncommittal
answers, results are also split into answered vs refused subsets with a refusal rate, so
"grounded refusal on a retrieval miss" is visible instead of blended into one low number.

Requires a configured LLM endpoint (bring up the Colab notebook first); it errors out
clearly if none is set. Each sample triggers several LLM calls, so keep the sample small.

Reproduce::

    python -m docintel.scripts.eval_ragas --contracts 5 --questions 40 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Any

from docintel.config import Settings, get_settings
from docintel.rag.answer import answer_question
from docintel.rag.chunk import build_chunks
from docintel.rag.embed import build_embedder
from docintel.rag.llm import build_llm
from docintel.rag.query import focus_query
from docintel.rag.rerank import build_reranker
from docintel.rag.store import build_vector_store, ensure_collection, upsert_chunks

# Markers of the grounded-refusal answer the RAG prompt mandates on retrieval misses.
_REFUSAL_MARKERS = ("enough information", "cannot answer", "unable to answer")


def _is_refusal(answer: str) -> bool:
    """True when the answer is the grounded 'not enough information' refusal."""
    lowered = answer.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


def _extract_clauses(contexts: dict[str, str], settings: Settings) -> dict[str, list[Any]]:
    """Run the real C1 extractor per contract; empty lists when it is unavailable."""
    try:
        from docintel.contracts.extractor import CuadQaOnnxExtractor

        extractor = CuadQaOnnxExtractor.load(settings)
    except Exception as exc:
        print(f"warning: C1 extractor unavailable ({exc}); indexing paragraph chunks only")
        return {title: [] for title in contexts}
    clauses: dict[str, list[Any]] = {}
    for title, text in contexts.items():
        print(f"extracting clauses: {title}")
        clauses[title] = extractor.extract(text)
    return clauses


def _sample_questions(
    dataset: Any, contracts: int, questions: int, seed: int
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Return (context-by-title, [(title, question)]) for gold-answered clause questions."""
    rng = random.Random(seed)
    titles = sorted({ex["title"] for ex in dataset})
    rng.shuffle(titles)
    keep = set(titles[:contracts])

    contexts: dict[str, str] = {}
    answered: list[tuple[str, str]] = []
    for ex in dataset:
        title = ex["title"]
        if title not in keep:
            continue
        contexts.setdefault(title, ex["context"])
        if ex["answers"]["answer_start"]:
            answered.append((title, ex["question"]))

    rng.shuffle(answered)
    return contexts, answered[:questions]


def run(
    contracts: int,
    questions: int,
    seed: int,
    max_workers: int,
    timeout: int,
    samples_csv: str = "",
) -> dict[str, Any]:
    """Generate answers over a CUAD sample and score them with RAGAS."""
    from datasets import load_dataset
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, faithfulness
    from ragas.run_config import RunConfig

    settings: Settings = get_settings()
    llm = build_llm(settings)
    if llm is None:
        raise SystemExit(
            "No LLM configured. Set DOCINTEL_LLM_BASE_URL (bring up the Colab notebook) "
            "and restart, then re-run this eval."
        )

    embedder = build_embedder(settings)
    dataset = load_dataset("theatticusproject/cuad-qa", split="train", trust_remote_code=True)
    contexts, asked = _sample_questions(dataset, contracts, questions, seed)

    # Index clause chunks (real C1 extractor, when available) + paragraph chunks of the
    # sampled contracts into one in-memory Qdrant.
    from qdrant_client import QdrantClient

    clauses_by_title = _extract_clauses(contexts, settings)
    client = QdrantClient(":memory:")
    ensure_collection(client, settings.qdrant_collection, settings.rag_embedding_dim)
    store = build_vector_store(settings, embedder, client=client)
    for title, text in contexts.items():
        chunks = build_chunks(
            text, clauses_by_title[title], settings.rag_chunk_size, settings.rag_chunk_overlap
        )
        upsert_chunks(store, title, chunks)

    # Run the real /ask pipeline (retrieve + rerank + generate) per question.
    reranker = build_reranker(settings)
    samples: list[SingleTurnSample] = []
    refusals: list[bool] = []
    for title, question in asked:
        response = answer_question(
            question,
            store,
            llm,
            settings,
            contract_id=title,
            reranker=reranker,
            retrieval_query=focus_query(question),
        )
        if response.generation_skipped or not response.answer:
            continue
        samples.append(
            SingleTurnSample(
                user_input=question,
                response=response.answer,
                retrieved_contexts=[c.text for c in response.citations],
            )
        )
        refusals.append(_is_refusal(response.answer))

    if not samples:
        raise SystemExit("No answers were generated; cannot score. Check the LLM endpoint.")

    # The Colab LLM serves one request at a time on a single GPU; serialize RAGAS
    # (max_workers=1) so concurrent judge calls don't queue past the timeout.
    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[faithfulness, answer_relevancy],
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(embedder),
        run_config=RunConfig(max_workers=max_workers, timeout=timeout),
    )

    frame = result.to_pandas()
    frame["refusal"] = refusals
    answered = frame[~frame["refusal"]]
    if samples_csv:
        frame.to_csv(samples_csv, index=False)

    def mean(subset: Any, column: str) -> float | None:
        return round(float(subset[column].mean()), 4) if len(subset) else None

    return {
        "sample_contracts": len(contexts),
        "scored_answers": len(samples),
        "refusal_rate": round(sum(refusals) / len(refusals), 4),
        "faithfulness": mean(frame, "faithfulness"),
        "answer_relevancy": mean(frame, "answer_relevancy"),
        "answered_only": {
            "count": len(answered),
            "faithfulness": mean(answered, "faithfulness"),
            "answer_relevancy": mean(answered, "answer_relevancy"),
        },
        "clause_chunks_indexed": any(clauses_by_title.values()),
        "config": {"contracts": contracts, "questions": questions, "seed": seed},
    }


def log_to_mlflow(metrics: dict[str, Any], experiment: str = "ragas-eval") -> None:
    """Log answer-quality metrics to MLflow (lazy import; called only via --mlflow)."""
    import mlflow

    mlflow.set_experiment(experiment)
    with mlflow.start_run():
        mlflow.log_params(metrics["config"])
        mlflow.log_metric("faithfulness", metrics["faithfulness"])
        mlflow.log_metric("answer_relevancy", metrics["answer_relevancy"])
        mlflow.log_metric("scored_answers", metrics["scored_answers"])
        mlflow.log_metric("refusal_rate", metrics["refusal_rate"])


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="RAGAS faithfulness + answer_relevancy over CUAD.")
    parser.add_argument("--contracts", type=int, default=5)
    parser.add_argument("--questions", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=1, help="RAGAS concurrency (1=serial).")
    parser.add_argument("--timeout", type=int, default=300, help="Per-job timeout (s).")
    parser.add_argument("--samples-csv", type=str, default="", help="Write per-sample scores CSV.")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--mlflow", action="store_true", help="Log metrics to MLflow.")
    args = parser.parse_args()

    metrics = run(
        args.contracts, args.questions, args.seed, args.max_workers, args.timeout, args.samples_csv
    )
    print(json.dumps(metrics, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
    if args.mlflow:
        log_to_mlflow(metrics)


if __name__ == "__main__":
    main()
