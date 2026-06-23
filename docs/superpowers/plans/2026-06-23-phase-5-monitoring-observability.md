# Phase 5 — Monitoring & Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose Prometheus metrics from the FastAPI service (HTTP request/latency/error + custom KIE-confidence and validation metrics) and add a provisioned Grafana/Loki/Promtail observability stack to `docker-compose.yml`, reproducible from a clean checkout.

**Architecture:** `prometheus-fastapi-instrumentator` adds `/metrics` with the standard HTTP metrics; a small `metrics.py` module defines two custom `prometheus_client` metrics bound to a **per-app `CollectorRegistry`** (so the test suite, which builds the app many times, never hits a duplicate-registration error). The `/extract` route records the domain metrics after validation. Compose gains `prometheus`, `loki`, `promtail`, `grafana` services with in-repo config under `monitoring/`; Grafana is provisioned from files so it boots fully wired.

**Tech Stack:** FastAPI, prometheus-fastapi-instrumentator, prometheus_client, Prometheus, Grafana, Loki, Promtail, Docker Compose, pytest.

## Global Constraints

- Python 3.12+, full type hints; `from __future__ import annotations` at top of every module (matches existing files).
- Prefer functional style; keep functions small. No hardcoded constants in app code — config lives in `Settings` (`DOCINTEL_` env prefix). (Monitoring service config lives in static YAML/JSON files under `monitoring/`, mirroring how `prometheus.yml`-style infra config is conventionally kept.)
- Pin all Docker image tags (no `latest`), consistent with the existing `mlflow`/`minio` pins.
- New runtime dependency goes in the **`serve`** extra of `docintel/pyproject.toml`.
- Quality gate for every code task: `ruff check .`, `ruff format --check .`, `mypy src` (strict per project config), `pytest` — all green.
- All paths below are relative to the repo root; the Python package lives under `docintel/`. Run Python/test/uv commands from the `docintel/` directory (where `pyproject.toml` lives).
- **Env note (uv-sync gotcha):** `uv sync` replaces the environment. To keep the full test suite runnable after adding the new dependency, sync **all** extras: `uv sync --all-extras`.
- Never commit automatically without the explain/summarize/confirm step — but each task below ends with a commit the executor should make once its tests are green (the human approves at task review).

---

## File Structure

**Create:**
- `docintel/src/docintel/api/metrics.py` — custom metrics + `build_metrics` + `record_extraction`.
- `docintel/tests/test_metrics.py` — unit tests for the metrics module + `/metrics` endpoint + record-on-extract flow.
- `docintel/monitoring/prometheus.yml` — Prometheus scrape config.
- `docintel/monitoring/loki-config.yml` — single-binary Loki config.
- `docintel/monitoring/promtail-config.yml` — Promtail → Loki, Docker service discovery.
- `docintel/monitoring/grafana/provisioning/datasources/datasources.yml` — Prometheus + Loki datasources.
- `docintel/monitoring/grafana/provisioning/dashboards/dashboards.yml` — file-based dashboard provider.
- `docintel/monitoring/grafana/dashboards/docintel.json` — the DocIntel dashboard.
- `docintel/tests/test_monitoring_config.py` — parses every config file (YAML/JSON valid; key invariants hold).
- `docs/phases/phase5/report_phase5.md` — completion report (final task).

**Modify:**
- `docintel/pyproject.toml` — add `prometheus-fastapi-instrumentator` to the `serve` extra.
- `docintel/src/docintel/api/main.py` — build the per-app registry, custom metrics, and instrumentator in `create_app`.
- `docintel/src/docintel/api/routes/extract.py` — add `get_metrics` dependency and call `record_extraction`.
- `docintel/docker-compose.yml` — add `prometheus`, `loki`, `promtail`, `grafana` services + volumes.
- `docintel/tests/test_compose.py` — update `EXPECTED_SERVICES` and the comment.
- `docs/phases/phase5/phase5.md` + `docs/phases/README.md` — mark status (final task).

---

## Task 1: Metrics module + dependency

**Files:**
- Modify: `docintel/pyproject.toml` (the `serve` extra)
- Create: `docintel/src/docintel/api/metrics.py`
- Test: `docintel/tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `build_metrics(registry: CollectorRegistry) -> Metrics` — `Metrics` is a frozen dataclass with `kie_field_confidence: Histogram` and `validation_total: Counter`, both registered on `registry`.
  - `record_extraction(metrics: Metrics, document: Document) -> None` — observes each value in `document.field_confidence` into the histogram and increments `validation_total` with label `outcome="ok"|"failed"` from `document.validation.ok`.
  - Exposed metric names: `docintel_kie_field_confidence` (histogram) and `docintel_validation_total` (counter; `prometheus_client` appends `_total` to the base name `docintel_validation`).

- [ ] **Step 1: Add the dependency**

In `docintel/pyproject.toml`, change the `serve` extra from:

```toml
serve = [
    "onnxruntime>=1.16",
    "transformers>=4.40,<5",
]
```

to:

```toml
serve = [
    "onnxruntime>=1.16",
    "transformers>=4.40,<5",
    "prometheus-fastapi-instrumentator>=7.0",
]
```

- [ ] **Step 2: Sync the environment (all extras)**

Run (from `docintel/`): `uv sync --all-extras`
Expected: resolves and installs `prometheus-fastapi-instrumentator` and `prometheus-client`; exit 0.

- [ ] **Step 3: Write the failing unit test**

Create `docintel/tests/test_metrics.py`:

```python
"""Tests for the Prometheus metrics module and /metrics endpoint."""

from __future__ import annotations

from prometheus_client import CollectorRegistry

from docintel.api.metrics import build_metrics, record_extraction
from docintel.schema import Document, ValidationReport


def _doc(*, ok: bool, confidences: dict[str, float]) -> Document:
    return Document(
        id="d1",
        currency="IDR",
        field_confidence=confidences,
        validation=ValidationReport(ok=ok),
        created_at="2026-06-23T00:00:00+00:00",
    )


def test_record_extraction_observes_confidences_and_outcome() -> None:
    registry = CollectorRegistry()
    metrics = build_metrics(registry)

    record_extraction(metrics, _doc(ok=True, confidences={"total": 0.9, "subtotal": 0.8}))

    assert registry.get_sample_value("docintel_kie_field_confidence_count") == 2.0
    assert (
        registry.get_sample_value("docintel_validation_total", {"outcome": "ok"}) == 1.0
    )


def test_record_extraction_counts_failed_outcome() -> None:
    registry = CollectorRegistry()
    metrics = build_metrics(registry)

    record_extraction(metrics, _doc(ok=False, confidences={}))

    assert (
        registry.get_sample_value("docintel_validation_total", {"outcome": "failed"})
        == 1.0
    )
    # No confidences observed -> histogram count is 0 (metric still present).
    assert registry.get_sample_value("docintel_kie_field_confidence_count") == 0.0
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'docintel.api.metrics'`.

- [ ] **Step 5: Implement the metrics module**

Create `docintel/src/docintel/api/metrics.py`:

```python
"""Prometheus metrics for the DocIntel API.

Custom domain metrics are bound to a caller-supplied ``CollectorRegistry`` so each
FastAPI app instance owns its own metrics. This keeps the test suite — which builds
the app many times — free of duplicate-registration errors on the global registry.
"""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Histogram

from docintel.schema import Document

_CONFIDENCE_BUCKETS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


@dataclass(frozen=True)
class Metrics:
    """Custom DocIntel metrics bound to a single registry."""

    kie_field_confidence: Histogram
    validation_total: Counter


def build_metrics(registry: CollectorRegistry) -> Metrics:
    """Create the custom metrics against ``registry`` (one set per app instance)."""
    return Metrics(
        kie_field_confidence=Histogram(
            "docintel_kie_field_confidence",
            "Per-field KIE confidence observed on /extract.",
            buckets=_CONFIDENCE_BUCKETS,
            registry=registry,
        ),
        validation_total=Counter(
            "docintel_validation",
            "Documents processed by /extract, labelled by validation outcome.",
            labelnames=("outcome",),
            registry=registry,
        ),
    )


def record_extraction(metrics: Metrics, document: Document) -> None:
    """Record one extracted document: field confidences + validation outcome."""
    for value in document.field_confidence.values():
        metrics.kie_field_confidence.observe(value)
    outcome = "ok" if document.validation.ok else "failed"
    metrics.validation_total.labels(outcome=outcome).inc()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: PASS (2 tests).

- [ ] **Step 7: Quality gate**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy src`
Expected: all clean.

- [ ] **Step 8: Commit**

```bash
git add docintel/pyproject.toml docintel/uv.lock docintel/src/docintel/api/metrics.py docintel/tests/test_metrics.py
git commit -m "feat(metrics): add KIE confidence + validation prometheus metrics"
```

---

## Task 2: Wire the instrumentator + /metrics endpoint

**Files:**
- Modify: `docintel/src/docintel/api/main.py`
- Test: `docintel/tests/test_metrics.py` (add endpoint test)

**Interfaces:**
- Consumes: `build_metrics` from Task 1.
- Produces: `app.state.metrics: Metrics` is set in `create_app`; `GET /metrics` returns Prometheus exposition text (HTTP metrics + custom metrics), backed by a per-app `CollectorRegistry`.

- [ ] **Step 1: Write the failing endpoint test**

Append to `docintel/tests/test_metrics.py`:

```python
from fastapi.testclient import TestClient  # noqa: E402

from docintel.api.main import create_app  # noqa: E402


def test_metrics_endpoint_exposes_custom_and_http_metrics() -> None:
    with TestClient(create_app()) as client:
        client.get("/health")  # generate one HTTP sample
        resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    # Custom metric is registered at app build time, so it is always present.
    assert "docintel_kie_field_confidence" in body
    # Instrumentator default HTTP metric.
    assert "http_request_duration_seconds" in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metrics.py::test_metrics_endpoint_exposes_custom_and_http_metrics -v`
Expected: FAIL — `GET /metrics` returns 404 (route not registered yet).

- [ ] **Step 3: Wire the instrumentator in create_app**

Edit `docintel/src/docintel/api/main.py`. Add imports near the existing imports:

```python
from prometheus_client import CollectorRegistry
from prometheus_fastapi_instrumentator import Instrumentator

from docintel.api.metrics import build_metrics
```

Then in `create_app`, after `app = FastAPI(...)` and before `app.include_router(...)`, insert:

```python
    registry = CollectorRegistry()
    app.state.metrics = build_metrics(registry)
    Instrumentator(registry=registry).instrument(app).expose(app)
```

(The per-app `registry` is passed to `Instrumentator` so both the default HTTP metrics and our custom metrics share one registry, and rebuilding the app in tests never collides on the global registry.)

- [ ] **Step 4: Run the endpoint test to verify it passes**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the full API test set (no regressions)**

Run: `uv run pytest tests/test_health.py tests/test_documents.py tests/test_extract.py -v`
Expected: all PASS (app builds repeatedly across tests with no duplicate-registration error).

- [ ] **Step 6: Quality gate**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy src`
Expected: all clean.

- [ ] **Step 7: Commit**

```bash
git add docintel/src/docintel/api/main.py docintel/tests/test_metrics.py
git commit -m "feat(api): expose /metrics via prometheus-fastapi-instrumentator"
```

---

## Task 3: Record metrics in the /extract route

**Files:**
- Modify: `docintel/src/docintel/api/routes/extract.py`
- Test: `docintel/tests/test_metrics.py` (add flow test)

**Interfaces:**
- Consumes: `record_extraction`, `Metrics` (Task 1); `app.state.metrics` (Task 2); the stubbed-app helpers `_make_stubbed_app`, `_png_bytes` from `tests/test_extract.py`.
- Produces: each successful `POST /extract` increments `docintel_validation_total` and observes confidences.

- [ ] **Step 1: Write the failing flow test**

Append to `docintel/tests/test_metrics.py`:

```python
from typing import Any  # noqa: E402

from tests.test_extract import _make_stubbed_app, _png_bytes  # noqa: E402


def test_extract_records_validation_metric(tmp_path: Any) -> None:
    from docintel.config import Settings

    app = _make_stubbed_app(Settings(sqlite_path=str(tmp_path / "db.sqlite")))
    with TestClient(app) as client:
        assert (
            client.post(
                "/extract", files={"file": ("r.png", _png_bytes(), "image/png")}
            ).status_code
            == 200
        )
        body = client.get("/metrics").text
    app.dependency_overrides.clear()
    assert "docintel_validation_total{outcome=" in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metrics.py::test_extract_records_validation_metric -v`
Expected: FAIL — the counter has no samples yet (route does not record), so the labelled line is absent from `/metrics`.

- [ ] **Step 3: Add the dependency and call in the route**

Edit `docintel/src/docintel/api/routes/extract.py`.

Add to the imports:

```python
from docintel.api.metrics import Metrics, record_extraction
```

Add a dependency provider after `get_s3_client` (around line 56):

```python
def get_metrics(request: Request) -> Metrics:
    """Return the per-app metrics set built in create_app."""
    return request.app.state.metrics
```

Add the parameter to the `extract` signature (after the `s3` dependency):

```python
    metrics: Metrics = Depends(get_metrics),  # noqa: B008
```

Add the recording call immediately before `return document` (after the `logger.info("extract.complete", ...)` block):

```python
    record_extraction(metrics, document)
    return document
```

- [ ] **Step 4: Run the flow test to verify it passes**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Run the existing extract tests (no regressions)**

Run: `uv run pytest tests/test_extract.py -v`
Expected: all PASS (the new dependency resolves from `app.state.metrics`, set by `create_app`, with no override needed).

- [ ] **Step 6: Quality gate**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy src`
Expected: all clean.

- [ ] **Step 7: Commit**

```bash
git add docintel/src/docintel/api/routes/extract.py docintel/tests/test_metrics.py
git commit -m "feat(api): record KIE confidence + validation metrics on /extract"
```

---

## Task 4: Monitoring config files

**Files:**
- Create: `docintel/monitoring/prometheus.yml`
- Create: `docintel/monitoring/loki-config.yml`
- Create: `docintel/monitoring/promtail-config.yml`
- Create: `docintel/monitoring/grafana/provisioning/datasources/datasources.yml`
- Create: `docintel/monitoring/grafana/provisioning/dashboards/dashboards.yml`
- Create: `docintel/monitoring/grafana/dashboards/docintel.json`
- Test: `docintel/tests/test_monitoring_config.py`

**Interfaces:**
- Produces: in-repo config consumed by the compose services in Task 5. Prometheus scrapes `api:8000`; Grafana datasources have fixed uids `prometheus` and `loki` referenced by the dashboard.

- [ ] **Step 1: Write the failing config test**

Create `docintel/tests/test_monitoring_config.py`:

```python
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
        (MON / "grafana/provisioning/datasources/datasources.yml").read_text(
            encoding="utf-8"
        )
    )
    uids = {ds["uid"] for ds in cfg["datasources"]}
    assert {"prometheus", "loki"} <= uids


def test_dashboard_json_is_valid() -> None:
    dash = json.loads(
        (MON / "grafana/dashboards/docintel.json").read_text(encoding="utf-8")
    )
    assert dash["title"] == "DocIntel"
    assert len(dash["panels"]) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_monitoring_config.py -v`
Expected: FAIL — config files do not exist yet (`FileNotFoundError`).

- [ ] **Step 3: Create `docintel/monitoring/prometheus.yml`**

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: docintel-api
    metrics_path: /metrics
    static_configs:
      - targets: ["api:8000"]
```

- [ ] **Step 4: Create `docintel/monitoring/loki-config.yml`**

```yaml
auth_enabled: false

server:
  http_listen_port: 3100

common:
  instance_addr: 127.0.0.1
  path_prefix: /loki
  storage:
    filesystem:
      chunks_directory: /loki/chunks
      rules_directory: /loki/rules
  replication_factor: 1
  ring:
    kvstore:
      store: inmemory

schema_config:
  configs:
    - from: 2020-10-24
      store: tsdb
      object_store: filesystem
      schema: v13
      index:
        prefix: index_
        period: 24h
```

- [ ] **Step 5: Create `docintel/monitoring/promtail-config.yml`**

Promtail discovers containers via the Docker socket and streams their logs to Loki (no host log-path mount needed — refines the spec's "tail container stdout" to the socket-based Docker SD, which is the portable approach on Docker Desktop):

```yaml
server:
  http_listen_port: 9080

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: docker
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
        refresh_interval: 5s
    relabel_configs:
      - source_labels: ["__meta_docker_container_name"]
        regex: "/(.*)"
        target_label: container
```

- [ ] **Step 6: Create `docintel/monitoring/grafana/provisioning/datasources/datasources.yml`**

```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    uid: prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
  - name: Loki
    uid: loki
    type: loki
    access: proxy
    url: http://loki:3100
```

- [ ] **Step 7: Create `docintel/monitoring/grafana/provisioning/dashboards/dashboards.yml`**

```yaml
apiVersion: 1

providers:
  - name: docintel
    type: file
    allowUiUpdates: true
    options:
      path: /etc/grafana/dashboards
      foldersFromFilesStructure: false
```

- [ ] **Step 8: Create `docintel/monitoring/grafana/dashboards/docintel.json`**

```json
{
  "title": "DocIntel",
  "uid": "docintel",
  "schemaVersion": 39,
  "version": 1,
  "time": { "from": "now-1h", "to": "now" },
  "refresh": "10s",
  "templating": { "list": [] },
  "annotations": { "list": [] },
  "panels": [
    {
      "id": 1,
      "title": "Request rate (req/s)",
      "type": "timeseries",
      "datasource": { "type": "prometheus", "uid": "prometheus" },
      "gridPos": { "h": 8, "w": 12, "x": 0, "y": 0 },
      "targets": [
        {
          "refId": "A",
          "expr": "sum(rate(http_requests_total[5m]))"
        }
      ]
    },
    {
      "id": 2,
      "title": "p95 latency (s)",
      "type": "timeseries",
      "datasource": { "type": "prometheus", "uid": "prometheus" },
      "gridPos": { "h": 8, "w": 12, "x": 12, "y": 0 },
      "targets": [
        {
          "refId": "A",
          "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_highr_seconds_bucket[5m])) by (le))"
        }
      ]
    },
    {
      "id": 3,
      "title": "Error rate (5xx req/s)",
      "type": "timeseries",
      "datasource": { "type": "prometheus", "uid": "prometheus" },
      "gridPos": { "h": 8, "w": 12, "x": 0, "y": 8 },
      "targets": [
        {
          "refId": "A",
          "expr": "sum(rate(http_requests_total{status=~\"5..\"}[5m]))"
        }
      ]
    },
    {
      "id": 4,
      "title": "Validation outcomes",
      "type": "timeseries",
      "datasource": { "type": "prometheus", "uid": "prometheus" },
      "gridPos": { "h": 8, "w": 12, "x": 12, "y": 8 },
      "targets": [
        {
          "refId": "A",
          "expr": "sum(docintel_validation_total) by (outcome)",
          "legendFormat": "{{outcome}}"
        }
      ]
    },
    {
      "id": 5,
      "title": "KIE field confidence (heatmap)",
      "type": "heatmap",
      "datasource": { "type": "prometheus", "uid": "prometheus" },
      "gridPos": { "h": 8, "w": 12, "x": 0, "y": 16 },
      "targets": [
        {
          "refId": "A",
          "format": "heatmap",
          "expr": "sum(rate(docintel_kie_field_confidence_bucket[5m])) by (le)",
          "legendFormat": "{{le}}"
        }
      ]
    },
    {
      "id": 6,
      "title": "API logs",
      "type": "logs",
      "datasource": { "type": "loki", "uid": "loki" },
      "gridPos": { "h": 8, "w": 12, "x": 12, "y": 16 },
      "targets": [
        {
          "refId": "A",
          "expr": "{container=\"docintel-api\"}"
        }
      ]
    }
  ]
}
```

- [ ] **Step 9: Run the config test to verify it passes**

Run: `uv run pytest tests/test_monitoring_config.py -v`
Expected: PASS (5 tests).

- [ ] **Step 10: Commit**

```bash
git add docintel/monitoring docintel/tests/test_monitoring_config.py
git commit -m "feat(monitoring): add prometheus, loki, promtail, grafana config"
```

---

## Task 5: Add observability services to docker-compose

**Files:**
- Modify: `docintel/tests/test_compose.py`
- Modify: `docintel/docker-compose.yml`

**Interfaces:**
- Consumes: config files from Task 4; the `/metrics` endpoint from Task 2.
- Produces: `docker compose up` brings up api + mlflow + minio + prometheus + loki + promtail + grafana.

- [ ] **Step 1: Update the compose guard test first (TDD)**

Edit `docintel/tests/test_compose.py`. Replace the comment and `EXPECTED_SERVICES`:

```python
# Core MLOps spine plus the Phase 5 observability stack. Qdrant (GraphRAG)
# remains out of scope until the advancements.
EXPECTED_SERVICES = {
    "api",
    "mlflow",
    "minio",
    "prometheus",
    "loki",
    "promtail",
    "grafana",
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_compose.py -v`
Expected: FAIL — assertion mismatch; compose still has only `{api, mlflow, minio}`.

- [ ] **Step 3: Add the services to `docintel/docker-compose.yml`**

Insert these four services after the `minio` service (before the top-level `volumes:` key):

```yaml
  prometheus:
    image: prom/prometheus:v2.54.1
    container_name: docintel-prometheus
    command:
      - --config.file=/etc/prometheus/prometheus.yml
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"
    depends_on:
      - api

  loki:
    image: grafana/loki:3.1.1
    container_name: docintel-loki
    command: -config.file=/etc/loki/loki-config.yml
    volumes:
      - ./monitoring/loki-config.yml:/etc/loki/loki-config.yml:ro
      - loki-data:/loki
    ports:
      - "3100:3100"

  promtail:
    image: grafana/promtail:3.1.1
    container_name: docintel-promtail
    command: -config.file=/etc/promtail/promtail-config.yml
    volumes:
      - ./monitoring/promtail-config.yml:/etc/promtail/promtail-config.yml:ro
      - /var/run/docker.sock:/var/run/docker.sock:ro
    depends_on:
      - loki

  grafana:
    image: grafana/grafana:11.2.0
    container_name: docintel-grafana
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
      GF_AUTH_ANONYMOUS_ENABLED: "true"
      GF_AUTH_ANONYMOUS_ORG_ROLE: Viewer
    volumes:
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
      - ./monitoring/grafana/dashboards:/etc/grafana/dashboards:ro
      - grafana-data:/var/lib/grafana
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
      - loki
```

Add the new named volumes to the existing top-level `volumes:` block:

```yaml
volumes:
  mlflow-data:
  minio-data:
  prometheus-data:
  loki-data:
  grafana-data:
```

- [ ] **Step 4: Run the compose test to verify it passes**

Run: `uv run pytest tests/test_compose.py -v`
Expected: PASS.

- [ ] **Step 5: Validate the compose file parses (Docker)**

Run (from `docintel/`): `docker compose config --quiet`
Expected: exit 0, no error. (If Docker is unavailable in the execution environment, skip and note it; the YAML is also covered by the passing `test_compose.py`.)

- [ ] **Step 6: Commit**

```bash
git add docintel/docker-compose.yml docintel/tests/test_compose.py
git commit -m "feat(compose): add prometheus, loki, promtail, grafana services"
```

---

## Task 6: Full verification, live stack check, and report

**Files:**
- Create: `docs/phases/phase5/report_phase5.md`
- Modify: `docs/phases/phase5/phase5.md` (check the metrics/dashboard task boxes; note CI deferred)
- Modify: `docs/phases/README.md` (Phase 5 status)

- [ ] **Step 1: Run the full quality gate**

Run (from `docintel/`):
```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest
```
Expected: ruff clean, format clean, mypy clean, full suite green (the one pre-existing slow real-OCR test is deselected by default config; note its status).

- [ ] **Step 2: Live stack verification (manual / integration)**

This proves the "done when" criterion; it needs Docker and the MLflow-registered ONNX model (as in the Phase 4 report). Run from `docintel/`:

```bash
docker compose up -d --build
```

Then verify:
- `curl -s localhost:8000/metrics | grep docintel_` shows the custom metrics.
- POST a CORD test receipt to `localhost:8000/extract` (as in the Phase 4 report) and re-check `/metrics` — `docintel_validation_total{outcome=...}` increments.
- Prometheus `localhost:9090/targets` shows the `docintel-api` target **UP**.
- Grafana `localhost:3000` (anonymous viewer) → the **DocIntel** dashboard renders request/latency/error + KIE confidence + validation panels.
- The dashboard **API logs** panel (Loki) shows `extract.complete` log lines from `docintel-api`.

Record the actual observations (and any metric-name corrections to the dashboard JSON) in the report. If a dashboard query shows no data, confirm the live metric name via `/metrics` and adjust `docintel.json` accordingly, then re-commit Task 4's dashboard.

Tear down when done: `docker compose down`.

- [ ] **Step 3: Write `docs/phases/phase5/report_phase5.md`**

Mirror the structure of `report_phase4.md`: What Was Built (metrics module, instrumentator wiring, /extract recording, monitoring stack), End-to-End Verification (the Step 2 observations with real numbers), Key Decisions (per-app registry; instrumentator + custom metric; Promtail Docker SD; CI build/push deferred), Deviations (Promtail socket-based SD vs host log mount; any dashboard query fixes), Test & Quality Status (counts), Done When (tick the metrics+dashboard criteria; note CI build/push is deferred so the phase's third task remains open).

- [ ] **Step 4: Update phase status docs**

In `docs/phases/phase5/phase5.md`: check the boxes for the Prometheus `/metrics` task and the Grafana/Loki task; leave the GitHub Actions build+push box unchecked with a note "(deferred — see report)".
In `docs/phases/README.md`: change the Phase 5 row Status from **Next** to a partial-complete note, e.g. `🟡 Metrics + dashboards complete — [report](phase5/report_phase5.md); CI image build/push deferred`.

- [ ] **Step 5: Commit**

```bash
git add docs/phases/phase5/report_phase5.md docs/phases/phase5/phase5.md docs/phases/README.md
git commit -m "docs(phase-5): add report; mark metrics + dashboards complete"
```

---

## Self-Review Notes

- **Spec coverage:** metrics module + custom KIE/validation metrics (Task 1), `/metrics` via instrumentator (Task 2), recording on `/extract` (Task 3), Prometheus/Loki/Promtail/Grafana config + provisioned dashboard (Task 4), compose services (Task 5), tests at every seam + report and done-when verification (Tasks 1–6). CI build/push is explicitly out of scope per the approved spec.
- **Per-app registry** is the one non-obvious correctness point (tests build the app repeatedly) and is enforced in Tasks 1–2.
- **Dashboard metric names** (`http_requests_total`, `http_request_duration_highr_seconds_bucket`) are the documented prometheus-fastapi-instrumentator defaults; Task 6 Step 2 verifies them against a live `/metrics` and corrects the JSON if a name differs in the installed version.
