# Phase 5 Design — Monitoring & Observability (metrics + dashboards)

> Approved design. Part of the [Build Roadmap](../../plan.md) → [Phase 5 brief](../../phases/phase5/phase5.md).
> **Scope this session:** Prometheus metrics + Grafana/Loki dashboards. CI image build/push is **deferred**.

## Goal

Production hygiene for the Phase 4 service: expose Prometheus metrics from the FastAPI
app, ship container logs to Loki, and surface request/latency/error + model-quality
signals on a provisioned Grafana dashboard — all reproducible from a clean checkout via
`docker compose up`.

## Decisions (locked during brainstorming)

- **Scope:** metrics + dashboards only. The CI `build + push image` step from the brief
  is deferred to a follow-up; `ci.yml` is left unchanged this session.
- **Metrics library:** `prometheus-fastapi-instrumentator` for HTTP metrics (request
  count, latency histograms, error rate, `/metrics` endpoint) **plus** a small custom
  `prometheus_client` metric for KIE confidence. Least code; standard HTTP metrics; still
  covers the domain signal.
- **Log shipping:** Promtail scrapes Docker container stdout → Loki. **Zero app code
  change** — the existing structured stdout logs flow straight through. Decoupled and
  reproducible (no host Docker plugin required).

## Architecture

```
                      ┌─────────────┐   scrape /metrics    ┌────────────┐
  POST /extract  ───▶ │  api (8000) │ ───────────────────▶ │ prometheus │ ─┐
                      │  FastAPI    │                       └────────────┘  │
                      │  + instrum. │                                       ▼
                      └──────┬──────┘                                 ┌──────────┐
                             │ stdout (structured logs)               │ grafana  │
                             ▼                                        │ (3000)   │
                      ┌────────────┐   push     ┌──────┐  query       └──────────┘
                      │  promtail  │ ─────────▶ │ loki │ ◀──────────────────┘
                      └────────────┘            └──────┘
```

Grafana is provisioned from files (datasources + dashboard) so it boots fully wired.

## Components

### 1. Metrics (`docintel/src/docintel/api/metrics.py`, new)

Custom domain metrics via `prometheus_client`, plus a recorder used by the route:

- `KIE_FIELD_CONFIDENCE` — `Histogram` with buckets across `0.0–1.0`; observes each
  field's confidence value per `/extract` call.
- `VALIDATION_TOTAL` — `Counter` named `docintel_validation` (prometheus_client appends
  `_total`, exposing it as `docintel_validation_total`), labeled `outcome={ok,failed}`;
  incremented once per document from `document.validation.ok`.
- `record_extraction(document) -> None` — iterates `document.field_confidence` observing
  each value into `KIE_FIELD_CONFIDENCE`, and increments `VALIDATION_TOTAL` with the
  outcome label. Pure aside from touching the module-level metrics (testable against the
  registry).

### 2. App wiring

- `api/main.py` — in `create_app`, attach the instrumentator:
  `Instrumentator().instrument(app).expose(app)`. This adds `/metrics` and the default
  HTTP request-count / latency-histogram / in-progress / error metrics.
- `api/routes/extract.py` — one call to `record_extraction(document)` immediately before
  returning the `Document` (after validation, before the response).

### 3. Dependency

- Add `prometheus-fastapi-instrumentator` to the **`serve`** extra in `pyproject.toml`
  (it pulls `prometheus_client` transitively). No new top-level dep beyond that.

### 4. Observability stack (`docintel/docker-compose.yml` + `docintel/monitoring/`)

New compose services and config files (config kept in-repo for reproducibility):

| Service | Image | Port | Config |
|---------|-------|------|--------|
| `prometheus` | `prom/prometheus` | 9090 | `monitoring/prometheus.yml` |
| `loki` | `grafana/loki` | 3100 | `monitoring/loki-config.yml` |
| `promtail` | `grafana/promtail` | — | `monitoring/promtail-config.yml` |
| `grafana` | `grafana/grafana` | 3000 | `monitoring/grafana/provisioning/**` |

- `monitoring/prometheus.yml` — single scrape job targeting `api:8000/metrics`.
- `monitoring/loki-config.yml` — minimal single-binary (filesystem) Loki.
- `monitoring/promtail-config.yml` — tails `/var/lib/docker/containers/*/*log` (mounted
  read-only) with the docker socket, ships to `loki:3100`.
- `monitoring/grafana/provisioning/datasources/datasources.yml` — Prometheus + Loki
  datasources.
- `monitoring/grafana/provisioning/dashboards/dashboards.yml` — file-based dashboard
  provider pointing at the dashboards dir.
- `monitoring/grafana/dashboards/docintel.json` — one dashboard with panels:
  request rate, p95 latency, error rate, KIE-confidence heatmap, validation ok/failed,
  and a Loki logs panel.

Pinned image tags (no `latest`) consistent with the existing mlflow/minio pins.

## Data flow

1. Client calls `POST /extract` → existing Phase-4 pipeline produces a validated
   `Document`.
2. Route calls `record_extraction(document)`; the instrumentator already recorded the
   HTTP request/latency/error.
3. Prometheus scrapes `api:8000/metrics` on its interval.
4. The API's structured logs go to stdout → Docker captures them → Promtail tails and
   pushes to Loki.
5. Grafana queries Prometheus + Loki for the dashboard.

## Error handling

- `record_extraction` must never break a successful extraction: it operates on an
  already-built `Document` and only does arithmetic/label increments. A missing or empty
  `field_confidence` simply observes nothing; the validation counter always increments.
- Monitoring services are independent of the API: if `prometheus`/`grafana`/`loki` are
  down, `/extract` and `/metrics` still work. The API does not depend on them at runtime.

## Testing

- `/metrics` returns `200` with Prometheus exposition text including our custom metric
  names (`docintel_kie_field_confidence`, `docintel_validation_total`).
- `record_extraction` unit test: build a small `Document`, call it, assert the histogram
  observed the expected count and the counter incremented the right `outcome` label
  (read back from the registry).
- Existing `/extract` → persist → retrieve test stays green (the recorder is additive).
- Quality gate: `ruff check`, `ruff format --check`, `mypy --strict`, `pytest` all green.

## Done when

- `/metrics` exposes HTTP request/latency/error + the custom KIE-confidence and
  validation metrics.
- `docker compose up` brings up api + mlflow + minio + prometheus + loki + promtail +
  grafana; the provisioned Grafana dashboard shows live request/latency/error and model
  metrics, plus API logs from Loki.
- ruff, ruff-format, mypy-strict, and pytest are green.
- `report_phase5.md` added to `docs/phases/phase5/`.

## Out of scope (deferred)

- CI: extend GitHub Actions to build + push the image (the brief's third task).
- Alerting / Alertmanager, multi-service tracing, retention tuning.
