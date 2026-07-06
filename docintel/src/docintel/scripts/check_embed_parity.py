"""Gate: the local ONNX bundle must reproduce the trained model's embeddings.

The Colab notebook writes ``parity.json`` (fixed sentences + their vectors from the
fine-tuned sentence-transformers model) into the bundle. This script embeds the same
sentences through the production ``build_embedder`` path — both the document and the
query call — and fails if any cosine similarity drops below the threshold. Catches
pooling (bge = CLS), normalization, and tokenizer drift in the export.

Reproduce::

    python -m docintel.scripts.check_embed_parity --bundle models/rag-embed-cuad
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import TypedDict

_DEFAULT_THRESHOLD = 0.999


class ParityData(TypedDict):
    """Structure of parity.json."""

    sentences: list[str]
    vectors: list[list[float]]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def check(
    reference: ParityData, produced: list[list[float]], threshold: float
) -> list[tuple[str, float]]:
    """Return (sentence, cosine) for every produced vector below the threshold."""
    failures: list[tuple[str, float]] = []
    for sentence, expected, got in zip(
        reference["sentences"], reference["vectors"], produced, strict=True
    ):
        cosine = _cosine(expected, got)
        if cosine < threshold:
            failures.append((sentence, cosine))
    return failures


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Embedding bundle parity gate.")
    parser.add_argument("--bundle", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=_DEFAULT_THRESHOLD)
    args = parser.parse_args()

    from docintel.config import Settings
    from docintel.rag.embed import build_embedder

    reference = json.loads((Path(args.bundle) / "parity.json").read_text(encoding="utf-8"))
    embedder = build_embedder(Settings(rag_embedding_local_path=args.bundle))
    sentences = reference["sentences"]

    doc_failures = check(reference, embedder.embed_documents(sentences), args.threshold)
    query_failures = check(
        reference, [embedder.embed_query(sentence) for sentence in sentences], args.threshold
    )
    for label, failures in (("documents", doc_failures), ("query", query_failures)):
        for sentence, cosine in failures:
            print(f"FAIL [{label}] cosine={cosine:.6f}  {sentence[:80]!r}")
    if doc_failures or query_failures:
        sys.exit(1)
    print(f"parity OK: {len(sentences)} sentences, both paths, threshold {args.threshold}")


if __name__ == "__main__":
    main()
