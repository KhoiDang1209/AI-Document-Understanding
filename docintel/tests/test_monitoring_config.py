"""Validates the in-repo monitoring config files parse and hold key invariants."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

MON = Path(__file__).resolve().parent.parent / "monitoring"


def test_prometheus_scrapes_the_api() -> None:
    cfg = yaml.safe_load((MON / "prometheus.yml").read_text(encoding="utf-8"))
    targets = cfg["scrape_configs"][0]["static_configs"][0]["targets"]
    assert "api:8000" in targets


def test_promtail_ships_to_loki() -> None:
    cfg = yaml.safe_load((MON / "promtail-config.yml").read_text(encoding="utf-8"))
    assert cfg["clients"][0]["url"] == "http://loki:3100/loki/api/v1/push"


def test_loki_config_parses() -> None:
    cfg = yaml.safe_load((MON / "loki-config.yml").read_text(encoding="utf-8"))
    assert cfg["server"]["http_listen_port"] == 3100


def test_grafana_datasources_define_prometheus_and_loki() -> None:
    cfg = yaml.safe_load(
        (MON / "grafana/provisioning/datasources/datasources.yml").read_text(encoding="utf-8")
    )
    uids = {ds["uid"] for ds in cfg["datasources"]}
    assert {"prometheus", "loki"} <= uids


def test_dashboard_json_is_valid() -> None:
    dash = json.loads((MON / "grafana/dashboards/docintel.json").read_text(encoding="utf-8"))
    assert dash["title"] == "DocIntel"
    assert len(dash["panels"]) >= 1
