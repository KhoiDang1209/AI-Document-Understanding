"""The eval provenance block must pin the embedder identity.

Without this, a stock and a fine-tuned run produce byte-identical config and are
only told apart by filename — the exact ambiguity the review flagged.
"""

from __future__ import annotations

from docintel.config import Settings
from docintel.scripts.eval_rag import _run_config


def test_run_config_records_stock_embedder_identity() -> None:
    settings = Settings(rag_embedding_local_path="")
    cfg = _run_config(40, 0, (1, 3, 5), True, True, settings)
    assert cfg["embedding_model"] == settings.rag_embedding_model
    assert cfg["embedding_local_path"] is None


def test_run_config_records_finetuned_bundle_path() -> None:
    settings = Settings(rag_embedding_local_path="/models/bge-small-cuad")
    cfg = _run_config(40, 0, (1, 3, 5), True, True, settings)
    assert cfg["embedding_local_path"] == "/models/bge-small-cuad"
