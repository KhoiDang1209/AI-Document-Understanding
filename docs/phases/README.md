# Phases

Per-phase working folders for the DocIntel [Build Roadmap](../plan.md). Each folder holds the phase **brief** (`phaseN.md` / `aN.md`) and, once delivered, the completion **report** (`report_*.md`).

The roadmap is **core-first**: Phases 0–5 build the MLOps spine. The **Advancements** (A1–A5) begin only once the core is green.

## Core — the MLOps spine

| Phase | Title | Status |
|-------|-------|--------|
| [0](phase0/phase0.md) | Foundations & Environment | ✅ Complete — [report](phase0/report_phase0.md) |
| [1](phase1/phase1.md) | OCR Baseline + `/extract` | ✅ Complete — [report](phase1/report_phase1.md) |
| [2](phase2/phase2.md) | KIE Fine-tune (LayoutLMv3) + MLflow | ✅ Complete — [report](phase2/report_phase2.md) |
| [3](phase3/phase3.md) | Optimization: ONNX + INT8 + Benchmark | ✅ Complete — [report](phase3/report_phase3.md) · [benchmark](../benchmark.md) |
| [4](phase4/phase4.md) | Serving + Validation + Schema + Persistence | ✅ Complete — [report](phase4/report_phase4.md) |
| [5](phase5/phase5.md) | Monitoring, Observability & CI/CD | 🟡 Metrics + dashboards complete — [report](phase5/report_phase5.md); CI image build/push deferred |

## Advancements — after the core is green

| Item | Title | Status |
|------|-------|--------|
| [A1](a1/a1.md) | Layout Detection | Not started |
| [A2](a2/a2.md) | GraphRAG over Extracted Data + `/ask` | Not started |
| [A3](a3/a3.md) | Agent Orchestration (LangGraph) + `/agent` | Not started |
| [A4](a4/a4.md) | Kubernetes (kind) & Packaging | Not started |
| [A5](a5/a5.md) | On-Demand LLM KIE Backend (optional) | Not started |

Each brief mirrors the corresponding section of `plan.md` (goal, research, tasks, done-when) as a standalone working doc.
