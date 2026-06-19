# Project: Document AI Platform

## Tech Stack

* Python 3.12+
* uv
* FastAPI
* PyTorch + ONNX Runtime
* Pydantic
* MLflow
* Docker
* pytest
* ruff + mypy

## What We Build

A production-oriented Document AI pipeline:

```
Document
→ Layout Detection
→ OCR
→ Information Extraction
→ Validation
→ API
```

Build for modularity, maintainability, and reproducibility.

---

# Coding Standard

* Python with full type hints.
* Prefer **functional components over classes**.
* Keep functions small and focused.
* No hardcoded constants.
* Configuration lives in settings/env.
* Follow existing project style.

---

# Commit Standard

**Never commit automatically.**

Before every commit:

* Explain what changed.
* Summarize affected files.
* Ask for confirmation.
* Commit only after approval.

---

# Phase-by-Phase Development

Every phase follows the same workflow:

1. **Explore** – understand requirements and existing code.
2. **Plan** – propose architecture and trade-offs.
3. **Code** – implement the minimum required solution.
4. **Test** – format, lint, type-check, and verify behavior.
5. **Review** – summarize changes and identify improvements.

Each phase is a consecutive implementation of the project.

Example:

* Phase 0: Planning & project setup
* Phase 1: Repository initialization
* Phase 2: Layout Detection
* Phase 3: OCR
* Phase 4: Information Extraction
* ...

---

# Engineering Rules

### Don't Assume

* State assumptions explicitly.
* If uncertain, ask.
* Present multiple interpretations instead of silently choosing one.
* Surface trade-offs.
* Stop when requirements are unclear.

### Keep It Simple

* Write the minimum code that solves the problem.
* No speculative features.
* No unnecessary abstractions.
* No configurability unless requested.
* If a solution can be simpler, prefer the simpler one.

### Minimal Changes

When modifying existing code:

* Touch only what is necessary.
* Don't refactor unrelated code.
* Match the existing style.
* Remove only imports or variables made unused by your own changes.
* Mention unrelated issues instead of fixing them.

Every changed line should directly support the requested task.
