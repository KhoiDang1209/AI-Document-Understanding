from __future__ import annotations

from docintel.config import Settings


def test_agent_settings_defaults() -> None:
    s = Settings()
    assert s.langfuse_host == "http://langfuse:3000"
    assert s.langfuse_public_key is None
    assert s.langfuse_secret_key is None
    assert s.agent_enabled is True
    assert s.agent_max_retries == 1
