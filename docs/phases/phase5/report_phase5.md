# Phase 5 Report — Monitoring & Observability

**Status:** 🟡 Metrics + dashboards complete and verified · CI image build/push deferred
**Location:** `docintel/src/docintel/api/metrics.py`, `docintel/monitoring/`, `docintel/docker-compose.yml`
**Date:** 2026-06-23

Phase 5 adds the observability layer to the serving stack from Phases 0–4. The FastAPI
service now exposes Prometheus metrics at `/metrics` — the standard HTTP request /
latency / error metrics plus two custom domain metrics (KIE field confidence and
validation outcome) — and `docker compose up` brings up a fully provisioned
Prometheus + Loki + Promtail + Grafana stack that scrapes those metrics, ships
container logs, and renders a ready-made **DocIntel** dashboard from a clean checkout.
See [`phase5.md`](phase5.md) for the brief and
[`../../superpowers/specs/2026-06-23-phase-5-monitoring-observability-design.md`](../../superpowers/specs/2026-06-23-phase-5-monitoring-observability-design.md)
for the approved design.

**Scope note:** the approved spec narrowed this session to **metrics + dashboards**.
The third Phase 5 task — GitHub Actions lint/test/build+push — is intentionally
**deferred** to a follow-up and is not part of this delivery.

---

## 1. What Was Built

### Metrics module (`api/metrics.py`)
- `build_metrics(registry: CollectorRegistry) -> Metrics` — a frozen `Metrics` dataclass holding two `prometheus_client` collectors bound to a **caller-supplied registry**: `kie_field_confidence` (`Histogram`, fixed 0.1–1.0 buckets) and `validation_total` (`Counter`, base name `docintel_validation`, label `outcome`; `prometheus_client` appends `_total`).
- `record_extraction(metrics, document)` — observes every value in `document.field_confidence` into the histogram and increments `validation_total{outcome="ok"|"failed"}` from `document.validation.ok`.
- **Per-app registry** is the one non-obvious correctness point: the test suite rebuilds the app many times, so binding metrics to the default global registry would raise duplicate-registration errors. Each `create_app` owns its own `CollectorRegistry`.

### Instrumentator wiring (`api/main.py`)
- In `create_app`, after the app is built: a per-app `CollectorRegistry` is passed to both `build_metrics` (stored on `app.state.metrics`) and `Instrumentator(registry=...)`, which `.instrument(app).expose(app)` to add `GET /metrics`. Both the default HTTP metrics and the custom metrics share the one registry.

### Recording on `/extract` (`api/routes/extract.py`)
- A `get_metrics(request) -> Metrics` dependency returns `request.app.state.metrics` (annotated local fixes mypy `no-any-return`), mirroring the existing `get_ocr_engine` / `get_kie_backend` lazy-singleton pattern. `record_extraction(metrics, document)` runs after validation, immediately before the response — so every successful extraction updates the confidence histogram and validation counter.

### Monitoring config (`monitoring/`)
- `prometheus.yml` — scrapes `api:8000` at `/metrics`, 15 s interval.
- `loki-config.yml` — single-binary Loki, filesystem storage, `http_listen_port: 3100`.
- `promtail-config.yml` — Docker service discovery via `unix:///var/run/docker.sock`, ships to `http://loki:3100/loki/api/v1/push`; a relabel rule maps `__meta_docker_container_name` → `container`. **Zero application code change** for log shipping.
- `grafana/provisioning/datasources/datasources.yml` — Prometheus (uid `prometheus`, default) + Loki (uid `loki`).
- `grafana/provisioning/dashboards/dashboards.yml` — file provider pointing at `/etc/grafana/dashboards`.
- `grafana/dashboards/docintel.json` — the **DocIntel** dashboard, 6 panels: request rate, p95 latency, 5xx error rate, validation outcomes (`sum(docintel_validation_total) by (outcome)`), KIE confidence heatmap (`docintel_kie_field_confidence_bucket`), and an API logs panel (Loki `{container="docintel-api"}`).

### Compose (`docker-compose.yml`)
- Four new services pinned by tag (no `latest`, consistent with the existing `mlflow`/`minio` pins): `prom/prometheus:v2.54.1`, `grafana/loki:3.1.1`, `grafana/promtail:3.1.1`, `grafana/grafana:11.2.0`, each mounting its in-repo config read-only, plus named volumes `prometheus-data` / `loki-data` / `grafana-data`. Grafana runs with anonymous Viewer access so the dashboard opens without login.

### Packaging
- `prometheus-fastapi-instrumentator>=7.0` added to the **`serve`** extra (resolved to 8.0.1; pulls `prometheus-client` 0.25.0). No new hardcoded constants — buckets/metric names live in the module; service config lives in the static `monitoring/` files, mirroring how infra config is conventionally kept.

---

## 2. End-to-End Verification

The unit + integration suite proves the in-process wiring; a live `docker compose`
run proves the stack boots fully provisioned from a clean checkout. The four
observability services were brought up standalone (`docker compose up -d --no-deps
prometheus loki promtail grafana`) — the heavy API image (torch + baked docTR weights)
was not rebuilt, since the `/metrics` and `/extract`-recording wiring is already
covered green by the test suite.

- **`docker compose config --quiet` → exit 0** — the full 7-service compose file is valid.
- **All four services reached `running`**; `docker compose up` created the three named volumes from scratch.
- **Prometheus** — `/-/ready` → `200`; the `docintel-api` target is present with `scrapeUrl http://api:8000/metrics` (health `down` only because the API was intentionally not started under `--no-deps`) — confirms the scrape config targets the API correctly.
- **Loki** — `/ready` → `200` after startup.
- **Grafana** — `/api/health` → `{"database":"ok","version":"11.2.0"}`; `/api/datasources` returns both provisioned datasources (`uid: prometheus`, `uid: loki`, `readOnly: true` = file-provisioned); `/api/search` shows the **DocIntel** dashboard (`uid: docintel`, `type: dash-db`) loaded.
- **Promtail → Loki** — Loki's `container` label already lists all four running containers (`docintel-grafana`, `docintel-loki`, `docintel-prometheus`, `docintel-promtail`), proving the Docker-socket service discovery and log push work end to end. With the API running, `docintel-api` joins the set and the dashboard's logs panel (`{container="docintel-api"}`) populates.

The custom-metric path itself is verified by the test suite: `GET /metrics` exposes
`docintel_kie_field_confidence` + `http_request_duration_seconds`, and a stubbed
`POST /extract` produces a `docintel_validation_total{outcome=...}` sample.

---

## 3. Key Decisions

- **Per-app `CollectorRegistry`** — the correctness keystone; lets the test suite rebuild the app freely with no duplicate-registration collisions on the global registry.
- **`prometheus-fastapi-instrumentator` + one custom module** — the library gives the standard HTTP metrics for free; a small `metrics.py` adds only the two domain metrics, keeping app code minimal.
- **Promtail Docker service discovery** — tails container stdout via the Docker socket, so log shipping needs **no application code change** and no host log-path mounts.
- **Provisioned-from-file Grafana** — datasources and the dashboard are baked in as repo files, so the stack boots fully wired and reproducible; nothing is configured by hand.
- **CI build/push deferred** — per the approved spec, this session ships metrics + dashboards only.

---

## 4. Deviations from the Plan

- **Promtail uses Docker-socket service discovery** rather than mounting a host log path — the portable approach on Docker Desktop; refines the spec's "tail container stdout" without weakening it.
- **No dashboard query corrections were needed** — the documented `prometheus-fastapi-instrumentator` default metric names (`http_requests_total`, `http_request_duration_highr_seconds_bucket`, `http_request_duration_seconds`) matched the installed version (8.0.1), so `docintel.json` was left as planned.
- **Live verification scoped to the monitoring stack** (`--no-deps`) instead of a full `--build`, to avoid rebuilding the multi-GB API image; the API-integrated metric path is covered by the test suite instead.

---

## 5. Test & Quality Status

- **93 passed, 1 deselected** (the slow real-OCR test), up from the 84-test Phase 4 baseline. New tests: `test_metrics.py` (4 — record-on-extract, the two registry-sample assertions, and the `/metrics` endpoint), `test_monitoring_config.py` (5 — every config file parses and holds its key invariant), and the updated `test_compose.py` guard now asserting all 7 services.
- `ruff check .` clean, `mypy src` clean (40 source files). `ruff format --check .` flags only two **pre-existing** files (`notebooks/phase2_kie_layoutlmv3.ipynb`, `tests/test_kie_dataset.py`) that predate Phase 5 and are out of scope; all Phase 5 files are formatted.

---

## 6. Done When

- ✅ `/metrics` exposes live request / latency / error metrics plus the custom KIE-confidence and validation metrics, recorded on every `/extract`.
- ✅ `docker compose up` brings up a provisioned Prometheus + Loki + Promtail + Grafana stack from a clean checkout — Prometheus targets the API, Promtail ships container logs to Loki, Grafana renders the DocIntel dashboard with both datasources wired.
- ✅ ruff, mypy-strict, and the full pytest suite are green.
- ⏳ **Deferred:** GitHub Actions lint / test / build + push image — tracked for a follow-up per the approved spec.
