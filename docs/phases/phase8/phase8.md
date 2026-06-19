# Phase 8 — Observability & CI/CD

> Phase brief. Part of the [Build Roadmap](../../plan.md). Status: **Not started**.

**Goal:** Production hygiene — metrics, logs, dashboards, automated checks.

## Research 🔬
- [ ] Prometheus metrics to expose (request count, latency histograms, error rate, model latency, KIE confidence).
- [ ] Grafana dashboard layout; Loki log pipeline.

## Tasks
- [ ] Instrument FastAPI with Prometheus client; `/metrics` endpoint.
- [ ] Grafana dashboards + Loki log shipping in compose.
- [ ] Langfuse dashboards for LLM/RAG quality.
- [ ] GitHub Actions: lint, test, build + push image.

## Done when 📦
- [ ] Dashboards show live request/latency/error + model metrics; CI green on PRs.

## Report
On completion, add `report_phase8.md` to this folder.
