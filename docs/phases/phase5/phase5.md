# Phase 5 — Monitoring, Observability & CI/CD

> Phase brief. Part of the [Build Roadmap](../../plan.md). Status: 🟡 **Metrics + dashboards complete** — [report](report_phase5.md); CI image build/push deferred.

**Goal:** Production hygiene — metrics, logs, dashboards, automated checks. Completes the core MLOps story.

## Research 🔬
- [x] Prometheus metrics to expose (request count, latency histograms, error rate, KIE confidence).
- [x] Grafana dashboard layout; Loki log pipeline.

## Tasks
- [x] Instrument FastAPI with the Prometheus client; `/metrics` endpoint.
- [x] Grafana dashboards + Loki log shipping in compose.
- [ ] GitHub Actions: lint, test, build + push image. *(deferred — see report)*

## Done when 📦
- [x] Dashboards show live request/latency/error + model metrics. *(CI green on PRs — deferred with the GitHub Actions task.)*
- [x] **Core complete:** `docker compose up` runs the full pipeline — fine-tuned model pulled from the registry, served on CPU, monitored — reproducible from a clean checkout.

## Report
On completion, add `report_phase5.md` to this folder.
