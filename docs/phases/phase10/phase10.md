# Phase 10 (Optional) — On-Demand LLM KIE Backend

> Phase brief. Part of the [Build Roadmap](../../plan.md). Status: **Not started** (optional).

**Goal:** Showcase LLM fine-tuning + on-demand GPU serving without a persistent service.

## Research 🔬
- [ ] QLoRA recipe for Qwen2.5-3B on instruction-style KIE.
- [ ] On-demand/serverless deployment option + budget.

## Tasks
- [ ] ☁️ QLoRA fine-tune Qwen2.5-3B; log to MLflow.
- [ ] Deploy as on-demand endpoint behind the `KIEBackend` interface.
- [ ] Extend benchmark report: LLM vs encoder KIE (accuracy / latency / cost).

## Done when 📦
- [ ] `/extract?backend=llm` works on-demand; trade-offs documented.

## Report
On completion, add `report_phase10.md` to this folder.
