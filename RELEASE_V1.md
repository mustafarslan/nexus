# Nexus v1.0 Release Notes

**Release date:** 2026-06-14  
**Status:** Codebase frozen. No new features for v1.0.

Nexus v1.0 is a **Hybrid C++23 KV-Cache MMU and ML-gated agentic routing engine** on llama.cpp. It pre-compiles MCP tool schemas into offline `.atb` KV blocks, routes tools via quantized retrieval and a radix trie FSM, and splices KV into the active context within strict physical boundaries.

---

## Headline metrics

| Metric | Value | Artifact |
|--------|-------|----------|
| Gateway TTFT P50 | **171 ms** | `results/bench_gateway_e2e_v2.json` |
| E2E tool-hit accuracy | **0.91** | same |
| Honest baseline B3 TTFT | **1.33 s** | `results/bench_e2e_v2.json` |
| Bloat strawman B1 (retired) | **9.48 s** | `results/bench_e2e_v2.json` |
| L0 warm-tier copy P50 | **~3 µs** | `results/bench_phase28_radix_prefix_v2.json` |
| CE Recall@1 @ 20% fire rate | **0.90** | `results/recall_miss_analysis_v2.json` |

**Do not cite:** the 153× ratio against a 12,500-token bloat strawman, or B1 as a fair comparison baseline.

---

## The journey: 9.48 s → 171 ms

### Phase 0 — The prefill wall

Standard agentic systems append full MCP tool schemas on every turn. The B1 strawman (all schemas as text, ~12.5k tokens) measured **9.48 s** TTFT P50. This anchor is useful for motivation only.

### Phase 1 — Honest baseline (B3)

Retrieve + single-schema text prefill: **1.33 s** TTFT, **0.91** tool-hit. This is the fair comparison point.

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

`NexusAgent.route_and_splice` with calibrated CE gate: **171 ms** TTFT P50, **0.91** accuracy (n=100).

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
| **LegoLink partial recompute** | KL up to 5.72 at P=1024 |
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

- Git: `1384a33b`
- Model: Qwen2.5-14B-Instruct Q4_K_M
- Platform: Darwin arm64 (Apple Silicon UMA)
