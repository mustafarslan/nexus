# V1.3 Semantic IR — Readiness Report

**Date:** 2026-06-21 · Host: Apple-Silicon (`.venv/bin/python`, llama_cpp 0.2.83) · Model: Qwen2.5-14B Q4_K_M · Tools: first-10 GitHub MCP · n=30.
**Provenance:** `results/v1.3_sidecar_canonical/`. `v1.1_canonical/` and `v1.2_sidecar_canonical/` left untouched as historical artifacts.

## PROMOTION DECISION: **NOT CLEARED — canonical NOT promoted.**
Gate (Decision 3): specified-field consistency ≥ 0.95. **Measured: 0.926.** The system improved but the gate is not met, so per the stated condition no promotion occurred and no `CANONICAL_RESULTS_MANIFEST.json` was emitted.

## Metrics (Semantic IR = type + truncated inline descriptions, desc[:50])
| Metric | V1.2 (type-only IR) | **V1.3 (semantic IR)** | Gate |
|---|---|---|---|
| Routing accuracy | 0.867 | **0.867** | — |
| **Specified-field consistency vs Oracle** | 0.90 | **0.926** | **✗ < 0.95** |
| Arg JSON-valid rate | 0.97 | **0.967** | ✓ |
| Coreference — tool / arg | 1.0 / 1.0 | **1.0 / 1.0** | ✓ |
| TTFT → first arg token P50 | 378 ms | **452 ms** (vs 720 ms full-schema, ~1.6×) | ✓ fast |
| IR tokens P50 / P99 | 19 / 32 | **56 / 79** | (descriptions cost ~3× IR size) |
| Oracle exact parsed-equal | 0.13 | 0.10 | RETIRED (Decision 1 — noise) |

## Assessment
- Semantic descriptions closed ~⅓ of the remaining gap (0.90 → 0.926) by reducing argument mis-slotting, at the cost of ~3× IR size (still far below full schema; TTFT stays ~1.6× faster).
- The residual ~7% disagreement is genuine model mis-slotting on queries where a determinable field is adjacent to an unspecified one (e.g. owner vs repo). It is a real arg-gen imperfection, not metric noise.
- All other axes are healthy: routing at ceiling, JSON-valid 0.97, coreference perfect, concurrency safe (V5/V6 passed under the coarse lock in V1.2 and are unchanged).

## Options to reach ≥0.95 (decision required)
1. Raise the description budget (desc[:80–100]) and/or add an explicit field-order hint — may push past 0.95 at some token cost; re-measure.
2. Add a one-shot arg-format exemplar per tool (few-shot) in the IR block.
3. Increase n (e.g. 50–100) to tighten the estimate — but the point estimate is 0.926, so this is unlikely to cross 0.95 on its own.
4. Accept 0.926 as the V1.3 result and adjust the gate, or stop here.

**Honesty note:** I did not promote on a near-miss. The measured value governs; 0.926 < 0.95.
