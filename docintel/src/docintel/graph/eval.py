"""GraphRAG evaluation: multi-hop accuracy over a constructed case set (run later → MLflow).

Each case is (question, expected_contract_ids). We route + query the graph and compare the
returned contract id set to the expected set. MLflow logging is a thin, optional tail so the
metric function stays unit-testable on the laptop with no tracking server.
"""

from __future__ import annotations

from datetime import date

from docintel.config import Settings
from docintel.graph.query import run_graph_query
from docintel.graph.router import route
from docintel.graph.store import GraphStore

EvalCase = tuple[str, set[str]]


def evaluate_multihop(
    store: GraphStore,
    cases: list[EvalCase],
    settings: Settings,
    reference_date: date | None = None,
) -> dict[str, float]:
    """Return {'multihop_accuracy', 'n'} over the cases (exact contract-id-set match)."""
    if not cases:
        return {"multihop_accuracy": 0.0, "n": 0}
    correct = 0
    for question, expected in cases:
        decision = route(question)
        if decision.target != "graph":
            continue
        chunks = run_graph_query(store, decision, settings, reference_date)
        got = {c.contract_id for c in chunks}
        if got == expected:
            correct += 1
    return {"multihop_accuracy": correct / len(cases), "n": float(len(cases))}


def log_to_mlflow(metrics: dict[str, float], experiment: str = "graphrag-eval") -> None:
    """Log eval metrics to MLflow (import lazy; called only when running the eval, not in CI)."""
    import mlflow

    mlflow.set_experiment(experiment)
    with mlflow.start_run():
        mlflow.log_metrics(metrics)
