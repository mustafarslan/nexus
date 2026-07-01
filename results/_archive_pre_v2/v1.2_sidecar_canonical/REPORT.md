# V1.2 Hybrid (Routing Sidecar + Compressed Text IR) — Empirical Report

**Date:** 2026-06-21 · **Host:** Apple-Silicon (`.venv/bin/python`, llama_cpp 0.2.83) · **Model:** Qwen2.5-14B Q4_K_M · **Tools:** first-10 GitHub MCP.
**Provenance:** `results/v1.2_sidecar_canonical/` only. **`results/v1.1_canonical/` NOT touched.**

## Architecture (pivot from V1.1)
Routing is pure-Python retrieval (Dense+CE on the query embedding — never touches the deep llama context). Arguments are generated in the main context grounded on a **compressed TEXT IR** of the schema (`_compress_schema_to_ir`), NOT spliced KV. The ATB splice is removed from the request path. Concurrency: one coarse `_ctx_lock` per turn (single-threaded per llama_context).

## Verdict: MAJOR IMPROVEMENT, but the literal ≥95% arg-fidelity gate is NOT met.

| Check | V1.1 splice-sidecar | **V1.2 hybrid** | Gate |
|---|---|---|---|
| Routing accuracy (n=30) | 0.867 | **0.867** | ✓ ~ceiling |
| Arg JSON-valid rate | 0.77 | **0.97** | ✓ |
| **Specified-field consistency** vs Oracle (n=30) | — | **0.90** | ✗ (<0.95) |
| Oracle exact parsed-equal | 0.00 | **0.13** | ✗ (structurally capped — see below) |
| mean field agreement (incl. underspecified) | 0.24 | **0.60** | — |
| **Coreference — tool routing** (n=5) | 1.00 | **1.00** | ✓ |
| **Coreference — argument** (n=5) | 0.80 | **1.00** | ✓ (text IR fixed it) |
| V5 leak (40 serial) | ~3 MB | **0 MB** growth | ✓ |
| **V6 concurrency** (4 threads) | **SIGSEGV** | **24/24, 0 err, deterministic** | ✓ (coarse lock fixed it) |
| TTFT→first arg token P50 | — | **378 ms** (vs 677 ms full-schema) | ✓ IR ~1.8× faster, IR p50=19 tok |

## What the pivot fixed
- **Argument grounding:** text IR restores `owner`/`repo` extraction that spliced-KV got wrong; JSON-valid 0.77→0.97; coreference args 0.80→1.00.
- **Concurrency:** coarse `_ctx_lock` over the whole turn → the 4-thread SIGSEGV became clean serialized, fully deterministic execution.
- **Latency:** compressed IR (p50 19 tokens vs full schema) → ~1.8× faster time-to-first-arg-token.

## Why the ≥95% gate is still not met (honest)
1. **Oracle EXACT consistency is structurally capped.** Underspecified fields (`content`, `message`, `sha` — values absent from the query) are hallucinated by BOTH arms and differ → parsed-equal can't approach 95% on these queries regardless of architecture. This is a metric limitation (no ground-truth args), not a Hybrid defect.
2. **Specified-field consistency = 0.90**, not ≥0.95: on fields whose value IS in the query, the model still mis-slots ~10% (e.g. putting the repo name in `owner` when `owner` is unspecified). A genuine residual arg-gen imperfection, smaller than the splice path's.

## Options to close the gap (for decision)
- Add **one-line field descriptions** to the IR (still ≪ full schema) — likely lifts specified-consistency toward the full-schema Oracle, modest token cost.
- Gate on **specified-field consistency** (the meaningful metric) rather than exact parsed-equal, with an agreed threshold.
- Accept 0.90 specified / 0.97 valid as the V1.2 result and document; do not promote to canonical.

## Provenance
- `accuracy.json` (n=30 hybrid-vs-oracle records + coref). Harnesses: `test/bench_sidecar_accuracy.py`, `test/stress_sidecar_concurrency.py`. Canonical baseline untouched; **not promoted** (gate not cleared).
