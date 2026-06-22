# Phase 5 — Monitoring, Observability & CI/CD

> Phase brief. Part of the [Build Roadmap](../../plan.md). Status: **Not started**.

**Goal:** Production hygiene — metrics, logs, dashboards, automated checks. Completes the core MLOps story.

## Research 🔬
- [ ] Prometheus metrics to expose (request count, latency histograms, error rate, KIE confidence).
- [ ] Grafana dashboard layout; Loki log pipeline.

## Tasks
- [ ] Instrument FastAPI with the Prometheus client; `/metrics` endpoint.
- [ ] Grafana dashboards + Loki log shipping in compose.
- [ ] GitHub Actions: lint, test, build + push image.

## Done when 📦
- [ ] Dashboards show live request/latency/error + model metrics; CI green on PRs.
- [ ] **Core complete:** `docker compose up` runs the full pipeline — fine-tuned model pulled from the registry, served on CPU, monitored — reproducible from a clean checkout.

## Report
On completion, add `report_phase5.md` to this folder.
