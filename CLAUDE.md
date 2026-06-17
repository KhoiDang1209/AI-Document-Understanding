# CLAUDE.md

# Project Overview

This project is a production-oriented Document AI system that transforms unstructured documents into structured, validated, and queryable information.

The architecture includes:

* Layout Detection
* OCR Pipeline
* Document Understanding / Key Information Extraction
* Vision-Language Models
* Validation Layer
* FastAPI Serving
* MLflow Registry
* Docker Deployment
* CI/CD
* Monitoring & Logging

The objective is to build a modular, maintainable, and production-ready AI system rather than a notebook prototype.

---

# Development Principles

* Prioritize simplicity over cleverness.
* Write maintainable and modular code.
* Favor composition over tightly coupled components.
* Keep business logic separate from model inference.
* Every component should be independently testable.

---

# Coding Standards

* Use Python 3.12+
* Follow PEP8 and type hints
* Prefer async APIs when appropriate
* Use dataclasses or Pydantic models for structured data
* Avoid hardcoded constants
* Write descriptive function and variable names

---

# Architecture

Always think in terms of pipelines and services instead of scripts.

Prefer:

Document
→ Layout
→ OCR
→ Information Extraction
→ Validation
→ API
→ Monitoring

instead of monolithic implementations.

---

# AI Engineering Principles

When proposing solutions, always consider:

* Accuracy
* Latency
* GPU Cost
* Throughput
* Scalability
* Reliability
* Maintainability

Do not optimize for model accuracy alone.

Always discuss production trade-offs.

---

# Collaboration Style

Before implementing:

1. Understand the requirement.
2. Explain the proposed approach.
3. Identify potential risks.
4. Suggest a clean architecture.
5. Implement incrementally.

Avoid making large refactors without justification.

---

# Response Style

Be concise and technical.

Prefer concrete implementation details over generic explanations.

When multiple solutions exist, compare their advantages, disadvantages, and production implications.

Challenge assumptions when appropriate instead of automatically agreeing.

---

# Project Goal

Build an enterprise-grade Document AI platform that demonstrates real-world AI engineering practices, including model orchestration, deployment, observability, validation, and scalable system design.
