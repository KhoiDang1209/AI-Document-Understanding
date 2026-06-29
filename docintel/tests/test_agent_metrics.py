from __future__ import annotations

from prometheus_client import CollectorRegistry

from docintel.api.metrics import build_metrics


def test_agent_metrics_present() -> None:
    m = build_metrics(CollectorRegistry())
    m.agent_run_total.labels(status="ok").inc()
    m.agent_retries.inc()
    m.agent_steps.observe(4)
