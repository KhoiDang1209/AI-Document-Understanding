# C3 Completion Report — GraphRAG (Neo4j) + routed `/ask`

**Date:** 2026-06-29
**Status:** ✅ Code complete & merged. ✅ Multi-hop routing accuracy measured (this report). ✅ Deterministic (no LLM needed for the headline metric).
**Spec:** [`docs/superpowers/specs/2026-06-25-contract-intelligence-c3-graphrag-design.md`](../../superpowers/specs/2026-06-25-contract-intelligence-c3-graphrag-design.md)
**Plan:** [`docs/superpowers/plans/2026-06-29-contract-intelligence-c3-graphrag.md`](../../superpowers/plans/2026-06-29-contract-intelligence-c3-graphrag.md)

## What C3 delivers

A deterministic, date-centric knowledge graph over the C1 extractions, with `/ask` routed between vector (C2) and graph:

```
ContractDocument
 → normalize (Expiration Date → ISO date; Renewal Term → clause) [rule-based]
 → graph: (Contract)-[EXPIRES_ON]->(Date), (Contract)-[HAS_CLAUSE]->(ClauseType)
   built best-effort at extract time (beside Qdrant indexing)
 → route(question): rule-based → {expiring_within | auto_renewing_expiring_within | vector}
 → run Cypher template → cited facts (reuse C2 RetrievedChunk) → C2 generate-or-degrade
```

Modules: `graph/{schema,normalize,store,templates,router,query,build,eval}.py`. `GraphStore` Protocol with an in-memory fake (CPU-testable) + `Neo4jGraphStore`; a deselected parity test guards Cypher drift. Two deterministic Cypher templates only — **no text-to-Cypher, no hallucination**.

## Headline metric — multi-hop routing accuracy

**Method.** The gold graph is built from CUAD gold clauses over **all contracts** via the real `build_graph_contract` normalizer. Ten **diverse natural-language** date questions are run through the rule-based router + graph query; each question's expected contract set is computed **independently** from the gold ISO dates (so accuracy reflects router + template correctness, not a tautology against the store). Reference date fixed at 2019-01-01.

| Metric | Value |
|---|---|
| Contracts with parsable absolute expiration date (in graph) | 65 |
| Contracts with a Renewal Term clause (in graph) | 26 |
| NL date questions evaluated | 10 |
| **Multi-hop accuracy (exact contract-set match)** | **1.00 (10/10)** |

Every phrasing routed correctly and returned the exact expected set across varied day-windows (expected sizes 2–19): explicit windows (365 / 730 / 1095 / 1460 / 1825 / 3650 days), the no-count case ("expiring soon" → default 90-day window), and auto-renew detection ("auto-renewing", "auto-renew", "that renew" → `auto_renewing_expiring_within`).

**Honest reading.** This is a **correctness validation of a deterministic pipeline**, not a noisy ML score — 1.00 means the rule-based router + parameterized Cypher reliably translate diverse NL into the right multi-hop traversal. The genuine *quality* ceiling here is **coverage**, not routing (see below).

Reproduce: `python -m docintel.scripts.eval_graph`

## Key finding & limitation — date coverage

The substantive GraphRAG limitation is the rule-based date normalizer: across CUAD, **only ~18% (70/384) of "Expiration Date" gold answers are absolute, parsable dates** — most are *relative* ("ten (10) years from the Effective Date", "the earlier of …"). So the date graph covers **65/408 contracts**. This is a deliberate C3 scope decision (deterministic normalization, no inference), and it bounds recall of the date templates. Closing it would require resolving relative terms against the Effective/Agreement Date — a candidate C3.1 or an agent-tool reasoning step in C4.

## Verification

- New eval runner `src/docintel/scripts/eval_graph.py` reuses `graph.eval.evaluate_multihop`, the real `route` / `run_graph_query` / `build_graph_contract`; pure helpers unit-tested.
- Gates: `ruff check` / `ruff format --check` / `mypy src` clean.
