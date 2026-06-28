# Nexus v1.0 Release Notes

**Release date:** 2026-06-14  
**Status:** Codebase frozen. No new features for v1.0.

Nexus v1.0 is a **Hybrid C++23 KV-Cache MMU and ML-gated agentic routing engine** on llama.cpp. It pre-compiles MCP tool schemas into offline `.atb` KV blocks, routes tools via quantized retrieval and a radix trie FSM, and splices KV into the active context within strict physical boundaries.

---

## Headline metrics

> **⚠️ Corrected 2026-06-20 (v1.1 canonical regeneration).** Headlines below were regenerated on the
> original host; the committed 237.4 ms / 171 ms figures **did not reproduce**. Canonical bundle:
> [`results/v1.1_canonical/`](results/v1.1_canonical/).

| Metric | Value (v1.1 canonical) | Artifact |
|--------|-------|----------|
| Gateway **routing+splice** latency P50 | **160.2 ms** (n=500, std 0.6 ms) | `results/v1.1_canonical/raw/gateway_run[1-5].json` |
| Tool-routing accuracy | **0.89** | same |
| End-to-end TTFT (`N1x`, incl. first token) | **457.5 ms** median (B3 `B_RP` = 1131.4 ms → **2.47×, 95% CI [2.41, 2.72]**, n=100) | `results/v1.1_canonical/phaseH/h3_e2e_n100.json` |
| L0 warm-tier copy P50 | **~3.06 µs** (69.5% hit) | `results/v1.1_canonical/raw/l0_seed_[0-9].json` |
| CE Recall@1 @ 20% fire rate | **0.88** (target 0.90 not met) | `results/v1.1_canonical/raw/recall_miss_analysis.json` |

**Metric note:** "Gateway TTFT" in v1.0 was a misnomer — it measured routing+splice only (`decode_us=0`), not time-to-first-token. The honest like-for-like speedup over retrieve-and-prefill (`B3`) is **2.47×, 95% CI [2.41, 2.72]** (457.5 ms vs 1131.4 ms true TTFT, serial n=100; Phase H, supersedes the earlier n=30 2.34×), not 6.3×. Accuracy: no detectable N1x-vs-B3 gap at n=100 (Δ = −0.010, McNemar p = 1.0) — not "equivalent," CI ±~5 pts.
**Do not cite:** the 153× ratio against a 12,500-token bloat strawman, or B1 as a fair comparison baseline.

---

## The journey: full-context prefill → ~160 ms routing

### Phase 0 — The prefill wall

Standard agentic systems append full MCP tool schemas on every turn. The B1 strawman (all schemas as text, ~12.5k tokens) measured **12.35 s** TTFT P50. This anchor is useful for motivation only.

### Phase 1 — Honest baseline (B3)

Retrieve + single-schema text prefill: **1.50 s** TTFT, **0.92** tool-hit. This is the fair comparison point.

### Phase 2 — Data-oriented C++ (Path A)

Pre-compile schemas to `.atb` KV blocks. At $P \le 256$, POSIX-mapped splice + **5% suffix recompute** avoids full quadratic prefill. N1 fast path: **466 ms** (2.8× vs B3).

### Phase 3 — Exact-token L0 radix (Path B)

At $P > 256$, RoPE phase drift blocks ATB splice. Production falls back to a **zero-allocation exact-token LCP radix tree** (`node_arena_` + `token_arena_`, LCRS layout) with `llama_kv_cache_seq_cp` warm copies at **~3 µs** P50.

### Phase M3 — Strict ML gating

- **Transient intent signatures** enrich dense/CE documents (not BM25 or generative prompts).
- **P20 adversarial margin gate** (~0.014 threshold) preserves ~**20%** CE invocation budget.
- **0.15 static clamp dropped from routing** (retained in calibration audit only).
- **Synthetic-only CE v3** — zero E2E leakage in training data.

### Gateway production path

`NexusAgent.route_with_retrieval` with calibrated CE gate: **160.2 ms** routing+splice P50 (excludes first-token decode), **0.89** accuracy (n=500, regenerated 2026-06-20).

---

## What ships

```
Hybrid Dense Retrieval → P20 Margin-Gated CE v3 Rerank →
  P ≤ 256? → ATB KV Splice + 5% Suffix Recompute
  P > 256? → Exact-Token L0 Radix Copy / Text Prefill →
  Trie FSM Logit-Masked Decode → GBNF Argument Generation
```

See [`docs/architecture.md`](docs/architecture.md) for full system design.

---

## What does not ship (Graveyard)

| Approach | Why |
|----------|-----|
| **ColBERT MaxSim** | 709 µs P50 (>>5 µs budget), Recall@1 0.72 |
| **LegoLink partial recompute** | Deep/off-anchor splice degraded; rejected (legacy "KL 5.72" figure retired — not regenerated, no tensor-KL harness) |
| **Blockmask N1m orchestrator** | Tensor KL=0 but E2E accuracy 0.0 |
| **FNV-1a 32-token chunking** | 0% L0 hit rate, cache fragmentation (pre-DOD refactor) |
| **P>256 ATB splice** | RoPE physics boundary |

---

## Local setup (models not in Git)

Model weights, `.atb` binaries, and finetuned checkpoints are **gitignored**. Generate locally:

```sh
# Build C++ engine
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# Train CE v3 (synthetic-only, no E2E leakage)
python3 scripts/train_cross_encoder.py --output results/tool_cross_encoder_finetuned_v3

# Calibrate P20 margin gate
python3 scripts/calibrate_margin_threshold.py

# Run gateway benchmark
python3 test/bench_gateway_e2e.py --output results/bench_gateway_e2e_v2.json
```

Calibration constants are committed in [`src/nexus_calibration.py`](src/nexus_calibration.py).

---

## Validation anchor

- Git (canonical regen): `3ea7441` · llama.cpp `cb2463bb` · model_hash `a09ea5e7…`
- Model: Qwen2.5-14B-Instruct Q4_K_M
- Platform: Apple Silicon (Darwin 25.5.0 arm64)
- Canonical evidence: `results/v1.1_canonical/` · Audit: `docs/AUDIT_2026-06-20_LIVE.md`
- Single host / single model — generality not yet established.
