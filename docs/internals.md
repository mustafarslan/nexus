# Project Nexus v1.0: Low-Level Engine Internals

C++23 KV-Cache MMU mechanics, memory layouts, and FFI boundaries. JSON argument decoding, embedding inference, CE reranking, and MCP protocol handling live in Python (`nexus_agent.py`, `nexus_gateway.py`, `test/nexus_retrieval.py`).

Validation anchor: git `1384a33b`, Qwen2.5-14B-Instruct Q4_K_M, Darwin arm64.

---

## 1. Aeon Tool Block (.atb) Binary ABI

### 1.1 Header (`aeon_tool_block.hpp`)

128-byte `AeonToolBlockHeader` with 2 MiB page alignment (`NEXUS_PAGE_ALIGNMENT = 2097152`):

| Field | Purpose |
|-------|---------|
| `magic` | `"ATB1"` |
| `model_hash` | FNV-1a hash of model topology (compile-time fingerprint) |
| `n_layer`, `n_head_kv`, `d_head` | GQA-aware dimensions |
| `seq_len`, `base_pos` | Token count $S$, RoPE anchor $m_0$ |
| `k_tensor_offset`, `v_tensor_offset` | Contiguous F16 K/V payload offsets |

FNV-1a here hashes **model topology only** — it is not used for L0 prefix matching.

### 1.2 K/V Layout

Per-layer contiguous F16 tensors. `NexusPageMounter` mmap's with HugeTLB alignment; `NexusKvSplicer` blits stride-aware slices into `llama_kv_cache` cells.

---

## 2. SIMD Semantic Load Buffer (SLB)

`NexusSemanticSLB` ([`src/nexus_slb.hpp`](../src/nexus_slb.hpp)):

- INT8 quantized tool embeddings, ARM NEON `vdotq_s32` scan
- Hybrid dense + lexical RRF ($k=60$) when BM25 hashes provided
- Latency: **< 5 µs** P50 at $N \le 100$ tools

ColBERT MaxSim was removed from the SLB — see §3 Graveyard.

---

## 3. Graveyard

### 3.1 ColBERT MaxSim

Late-interaction ColBERT scoring disqualified:

- Latency P50 **709 µs** (production budget < 5 µs)
- Recall@1 **0.72** (target ≥ 0.90 with CE)

Evidence: [`results/bench_phase22_maxsim.json`](../results/bench_phase22_maxsim.json)

### 3.2 LegoLink Partial Recompute

Scattered token-level KV repair at long prefix ($P = 1024$):

- `legolink_k4`: KL **5.72**, top-1 **0.20**
- `legolink_full_chunk`: KL **0.00** but ~**1,430 ms** — research-only

G4 gate: `viable: false` — [`results/g4_gate_verdict.json`](../results/g4_gate_verdict.json)

### 3.3 Blockmask N1m Orchestrator

Block-diagonal attention masks isolate per-tool KV chunks:

- Tensor fidelity: KL **0.0**, top-1 **1.0**
- E2E tool accuracy: **0.0** — model cannot cross-attend between tools during generation

Evidence: [`results/bench_n1m_fidelity_blockmask.json`](../results/bench_n1m_fidelity_blockmask.json)

### 3.4 FNV-1a 32-Token Chunking

**Purged.** An earlier L0 design keyed radix edges on 32-token FNV-1a hash chunks. This caused:

- **0% cache hit rate** (terminal-leaf mismatch)
- L0 fragmentation under concurrent warm-slot eviction

Replaced by the Phase 3 DOD refactor: **exact-token LCP radix** (§6). FNV-1a chunking does not exist on the production hit path.

### 3.5 P>256 ATB Splice

`NexusOrchestrator::route_and_splice` blocks when `n_past > max_splice_pos_` (default 256). RoPE phase drift corrupts spliced KV at depth. Path B fallback is mandatory, not optional tuning.

---

## 4. Radix Trie FSM

`NexusTrieFSM` constrains tool-name decode via logit masking over a precompiled trie. States: `FORCE_RESOLVED` (high margin), `LEAF_REACHED`, abort on miss.

---

## 5. KV Splice and 5% Suffix Recompute

`NexusKvSplicer` ([`src/nexus_kv_splicer.cpp`](../src/nexus_kv_splicer.cpp)):

1. Inject `.atb` pages at `n_past` via mmap + Metal blit
2. Fused multi-chunk suffix recompute (`recompute_fused_multi`) on highest layer-1 deviation tokens
3. Lazy VRAM defragmentation after splice

Viable only at $P \le 256$ with ≤5% suffix fraction.

---

## 6. L0 Exact-Token LCP Radix Cache

`NexusRadixPrefixCache` ([`src/nexus_seq_warm_cache.hpp`](../src/nexus_seq_warm_cache.hpp), [`src/nexus_seq_warm_cache.cpp`](../src/nexus_seq_warm_cache.cpp)):

### 6.1 Data Structures

```cpp
struct RadixNode {
    uint32_t token_start_idx;   // index into token_arena_
    uint16_t token_length;      // edge label length (exact tokens)
    uint32_t first_child_idx;   // LCRS left-child
    uint32_t next_sibling_idx;  // LCRS right-sibling
    uint32_t pool_index;        // warm slot index
    uint32_t prefix_end_pos;    // KV position after this prefix
};
```

- `node_arena_` — flat vector of `RadixNode` (reserved `NODE_ARENA_RESERVE = 4096`)
- `token_arena_` — contiguous `int32_t` token labels (`TOKEN_ARENA_RESERVE = 65536`)
- `pool_meta_[32]` — warm `llama_seq_id` slots with LRU eviction and ref counting

### 6.2 Lookup Algorithm

`try_copy_prefix(ctx, prefix_tokens, dst_seq)`:

1. Acquire shared lock on `trie_rw_lock_`
2. Walk from root: `find_child(parent, prefix_tokens[off])` via LCRS sibling chain
3. Match full edge label byte-for-byte against `token_arena_`
4. Track best occupied pool slot along the walk (terminal-leaf hotfix: also check terminal node)
5. On hit: `llama_kv_cache_seq_cp` from warm slot → `dst_seq`, return `prefix_end_pos`

**Zero heap allocation on the hot path.** No hash chunking — pure exact-token LCP.

### 6.3 Warm-Path Update

`update_warm_from_seq` inserts exact token paths after successful text prefill, binding pool slots to tool IDs for subsequent hits.

### 6.4 Validated Performance

| Metric | Value |
|--------|-------|
| Hit rate P50 | **0.66** |
| Copy P50 | **~3 µs** |

Source: [`results/bench_phase28_radix_prefix_v2.json`](../results/bench_phase28_radix_prefix_v2.json) (regenerate via `./build/bench_phase28_radix_prefix`).

---

## 7. Dual-Path Gate (Orchestrator)

[`src/nexus_orchestrator.cpp`](../src/nexus_orchestrator.cpp):

```cpp
if (n_past > max_splice_pos_) {
    splice_guard_fallback_count_.fetch_add(1, std::memory_order_relaxed);
    return 0;  // Python Path B fallback
}
```

Python [`src/nexus_agent.py`](../src/nexus_agent.py) `_prefix_cache_text_fallback`: L0 `try_copy_to` then `decode_tokens` on schema text.

---

## 8. ML Internals (M3)

### 8.1 Intent Signatures

[`test/nexus_retrieval.py`](../test/nexus_retrieval.py) — `_intent_signature_prefix(name, verb_prefix)` prepended in `enriched_tool_document_text()`. Transient: not persisted in tool registry, not sent to generative prompts.

### 8.2 Margin Gate

`margin_gated_retrieve` → `effective_rerank_threshold()` returns **only** `CALIBRATED_MARGIN_THRESHOLD` from [`src/nexus_calibration.py`](../src/nexus_calibration.py):

```python
CALIBRATED_MARGIN_THRESHOLD = 0.013646852970123292  # P20 adversarial
CALIBRATED_AMBIGUOUS_MARGIN = 0.15                 # audit only
```

The 0.15 static clamp is **not** applied in `margin_gated_retrieve`. Routing uses P20 adversarial threshold exclusively.

### 8.3 CE v3 Load Path

`nexus_agent.py` defaults to `results/tool_cross_encoder_finetuned_v3`. Train locally:

```sh
python3 scripts/train_cross_encoder.py --output results/tool_cross_encoder_finetuned_v3
```

Checkpoint directory is gitignored.

---

## 9. Block Cache and Eviction

`NexusBlockCache` — 2Q eviction over `.atb` pages with hazard-pointer readers. FNV-1a used for **filepath** dedup keys only.

25% arena over-provision absorbs page-alignment fragmentation.

---

## 10. Python FFI

`nexus_fsm_ext` (pybind11 via [`src/bindings.cpp`](../src/bindings.cpp)): KV inject, decode, FSM mask, block cache, radix warm cache. Context pointer passed as `uintptr_t` from `llama_cpp`.

---

## 11. Canonical Benchmark Artifacts

| Artifact | Key evidence |
|----------|--------------|
| `bench_gateway_e2e_v2.json` | 171 ms / 0.91 |
| `bench_phase28_radix_prefix_v2.json` | ~3 µs L0 copy |
| `recall_miss_analysis_v2.json` | CE 0.90 @ 20% fire |
| `margin_calibration_p20.json` | P20 audit trail |
| `g4_gate_verdict.json` | P=1024 splice FAIL |
| `bench_phase22_maxsim.json` | ColBERT disqualification |
| `bench_n1m_fidelity_blockmask.json` | Blockmask E2E FAIL |

Full whitelist: [`results/README.md`](../results/README.md).
