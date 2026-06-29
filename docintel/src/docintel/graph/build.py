"""Upsert one ContractDocument's subgraph into a GraphStore (called best-effort at extract)."""

from __future__ import annotations

from docintel.contracts.schema import ContractDocument
from docintel.graph.normalize import build_graph_contract
from docintel.graph.store import GraphStore


def build_contract(doc: ContractDocument, store: GraphStore) -> bool:
    """Normalize and upsert the contract; return True if any fact (expiration/renewal) existed."""
    gc = build_graph_contract(doc)
    store.upsert_contract(gc)
    return gc.expiration is not None or gc.renewal is not None
