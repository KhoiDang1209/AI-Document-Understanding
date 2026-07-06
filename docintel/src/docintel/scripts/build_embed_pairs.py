"""Build CUAD (focused query -> covering paragraph window) pairs for embedder fine-tuning.

Positives are the production 1200/200 paragraph windows covering each gold span (built
with ``rag.chunk.build_chunks``), so the model trains on the exact text distribution it
retrieves over at serve time. The 40 seed-0 eval contracts (the same
``_sample_contracts`` call ``eval_rag`` uses) are excluded; a further ``--dev-contracts``
slice of the training pool becomes the dev set for during-training IR validation.

Reproduce::

    python -m docintel.scripts.build_embed_pairs --out-dir data/processed/embed_pairs
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from docintel.config import get_settings
from docintel.rag.chunk import build_chunks
from docintel.rag.query import focus_query
from docintel.scripts.eval_rag import _covering_chunk_indices, _sample_contracts


def build_pairs(
    dataset: Any, exclude_titles: set[str], size: int, overlap: int
) -> list[dict[str, str]]:
    """(focused query, covering paragraph window) per gold answer, minus held-out titles."""
    contexts: dict[str, str] = {}
    questions: dict[str, list[tuple[str, list[int]]]] = defaultdict(list)
    for ex in dataset:
        title = ex["title"]
        if title in exclude_titles:
            continue
        contexts.setdefault(title, ex["context"])
        starts = ex["answers"]["answer_start"]
        if starts:
            questions[title].append((ex["question"], list(starts)))

    pairs: list[dict[str, str]] = []
    for title, items in questions.items():
        chunks = build_chunks(contexts[title], [], size, overlap)
        by_index = {chunk.chunk_index: chunk for chunk in chunks}
        for question, starts in items:
            covering: set[int] = set()
            for start in starts:
                covering |= _covering_chunk_indices(chunks, start)
            query = focus_query(question)
            for index in sorted(covering):
                pairs.append({"query": query, "positive": by_index[index].text, "title": title})
    return pairs


def split_dev(
    pairs: list[dict[str, str]], dev_contracts: int, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Deterministic contract-disjoint train/dev split of the built pairs."""
    titles = sorted({pair["title"] for pair in pairs})
    random.Random(seed).shuffle(titles)
    dev_titles = set(titles[:dev_contracts])
    train = [pair for pair in pairs if pair["title"] not in dev_titles]
    dev = [pair for pair in pairs if pair["title"] in dev_titles]
    return train, dev


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="CUAD embedding fine-tune pair builder.")
    parser.add_argument("--out-dir", type=str, default="data/processed/embed_pairs")
    parser.add_argument("--holdout-sample", type=int, default=40)
    parser.add_argument("--holdout-seed", type=int, default=0)
    parser.add_argument("--dev-contracts", type=int, default=20)
    parser.add_argument("--dev-seed", type=int, default=0)
    args = parser.parse_args()

    from datasets import load_dataset

    settings = get_settings()
    dataset = load_dataset("theatticusproject/cuad-qa", split="train", trust_remote_code=True)
    holdout = set(_sample_contracts(dataset, args.holdout_sample, args.holdout_seed))
    pairs = build_pairs(dataset, holdout, settings.rag_chunk_size, settings.rag_chunk_overlap)
    train, dev = split_dev(pairs, args.dev_contracts, args.dev_seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "train.jsonl", train)
    _write_jsonl(out_dir / "dev.jsonl", dev)
    meta = {
        "train_pairs": len(train),
        "dev_pairs": len(dev),
        "train_contracts": len({p["title"] for p in train}),
        "dev_contracts": len({p["title"] for p in dev}),
        "holdout_titles": sorted(holdout),
        "chunk_size": settings.rag_chunk_size,
        "chunk_overlap": settings.rag_chunk_overlap,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "holdout_titles"}, indent=2))


if __name__ == "__main__":
    main()
