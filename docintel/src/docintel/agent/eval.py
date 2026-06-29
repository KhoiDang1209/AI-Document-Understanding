"""Compound-task eval for the agent: citation-grounding success rate (run later -> MLflow).

Each case is (task, contract_id, expected_citation_contract_ids). Success is exact match of
the cited contract id set, so the metric is LLM-independent and runs on the laptop. The
MLflow tail is optional and lazy, matching graph/eval.py.
"""

from __future__ import annotations

from docintel.agent.graph import AgentDeps, run_agent

EvalCase = tuple[str, str | None, set[str]]


def evaluate_agent(cases: list[EvalCase], deps: AgentDeps) -> dict[str, float]:
    """Return {'success_rate', 'n'} over the cases (exact cited-contract-id-set match)."""
    if not cases:
        return {"success_rate": 0.0, "n": 0.0}
    correct = 0
    for task, contract_id, expected in cases:
        response = run_agent(task, contract_id, deps)
        got = {c.contract_id for c in response.citations}
        if got == expected:
            correct += 1
    return {"success_rate": correct / len(cases), "n": float(len(cases))}


def log_to_mlflow(metrics: dict[str, float], experiment: str = "agent-eval") -> None:
    """Log eval metrics to MLflow (lazy import; called only when running the eval, not in CI)."""
    import mlflow

    mlflow.set_experiment(experiment)
    with mlflow.start_run():
        mlflow.log_metrics(metrics)
