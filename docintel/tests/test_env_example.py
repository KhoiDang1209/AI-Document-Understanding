"""Ensures .env.example keys map to real Settings fields (no drift)."""

from __future__ import annotations

from pathlib import Path

from docintel.config import Settings

ENV_EXAMPLE = Path(__file__).resolve().parent.parent / ".env.example"
PREFIX = "DOCINTEL_"


def _env_keys() -> list[str]:
    keys: list[str] = []
    for raw in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        keys.append(line.split("=", 1)[0].strip())
    return keys


def test_env_example_exists() -> None:
    assert ENV_EXAMPLE.is_file()


def test_env_keys_are_valid_settings_fields() -> None:
    valid = set(Settings.model_fields)
    for key in _env_keys():
        assert key.startswith(PREFIX), f"{key} missing {PREFIX} prefix"
        field = key[len(PREFIX) :].lower()
        assert field in valid, f"{key} is not a Settings field"
