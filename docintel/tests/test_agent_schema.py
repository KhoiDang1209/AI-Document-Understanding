from __future__ import annotations

from docintel.agent.schema import AgentRequest, AgentResponse


def test_agent_request_requires_task() -> None:
    req = AgentRequest(task="When does contract X expire?")
    assert req.contract_id is None


def test_agent_response_defaults() -> None:
    resp = AgentResponse(
        task="t", answer=None, status="degraded", contract_id=None, trace_id=None, retries=0
    )
    assert resp.citations == [] and resp.steps == [] and resp.status == "degraded"
