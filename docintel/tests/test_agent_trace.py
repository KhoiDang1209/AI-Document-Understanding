from __future__ import annotations

from docintel.agent.trace import build_tracer, trace_id_of
from docintel.config import Settings


def test_build_tracer_disabled_without_keys() -> None:
    assert build_tracer(Settings()) is None


def test_trace_id_of_handles_none_and_missing() -> None:
    assert trace_id_of(None) is None

    class _NoId:
        pass

    assert trace_id_of(_NoId()) is None


def test_trace_id_of_reads_attribute() -> None:
    class _WithId:
        last_trace_id = "abc123"

    assert trace_id_of(_WithId()) == "abc123"
