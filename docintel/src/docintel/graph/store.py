"""GraphStore abstraction: a Protocol plus an in-memory fake for CPU-only tests.

The fake re-implements the two date templates in Python so unit tests never need a live
Neo4j. The real Neo4jGraphStore (Task 5) runs the equivalent Cypher; a deselected parity
test guards against drift. build_graph_store returns None when the graph path is disabled.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from docintel.config import Settings
from docintel.graph.schema import GraphContract

_AUTO_RENEW = "auto_renewing_expiring_within"
_EXPIRING = "expiring_within"


@runtime_checkable
class GraphStore(Protocol):
    """Minimal store interface: upsert one contract's subgraph; run a named template."""

    def upsert_contract(self, gc: GraphContract) -> None: ...

    def run_template(self, name: str, params: dict[str, Any]) -> list[dict[str, Any]]: ...


class InMemoryGraphStore:
    """Dict-backed fake; runs the two date templates in Python. Upsert is keyed by id."""

    def __init__(self) -> None:
        self._data: dict[str, GraphContract] = {}

    def upsert_contract(self, gc: GraphContract) -> None:
        self._data[gc.contract_id] = gc

    def run_template(self, name: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        lower, upper = params["lower"], params["upper"]
        rows: list[dict[str, Any]] = []
        for gc in self._data.values():
            if gc.expiration is None or not (lower <= gc.expiration.iso_date <= upper):
                continue
            if name == _AUTO_RENEW and gc.renewal is None:
                continue
            row: dict[str, Any] = {
                "contract_id": gc.contract_id,
                "iso_date": gc.expiration.iso_date,
                "exp_answer": gc.expiration.answer_text,
                "exp_start": gc.expiration.char_start,
                "exp_end": gc.expiration.char_end,
            }
            if name == _AUTO_RENEW and gc.renewal is not None:
                row["ren_answer"] = gc.renewal.answer_text
                row["ren_start"] = gc.renewal.char_start
                row["ren_end"] = gc.renewal.char_end
            rows.append(row)
        return rows


def build_graph_store(settings: Settings) -> GraphStore | None:
    """Return a GraphStore, or None when the graph path is disabled."""
    if not settings.graph_enabled:
        return None
    from docintel.graph.templates import Neo4jGraphStore

    return Neo4jGraphStore(settings)
