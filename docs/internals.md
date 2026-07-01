# Nexus Internals: Implementation Invariants & ABI Specifications

This document outlines the low-level C++ invariants, the `.atb` file format layout, attention-layout mappings, and the empirical provenance of the v2.0 benchmark files.

---

## 1. System Invariants & Bounds

For splicing to execute without memory corruption or output degradation, the orchestrator and splicer enforce six invariants:

1.  **Context Capacity Guard**: Before any splice, the orchestrator checks that:
    $$n_{\mathrm{past}} + \text{schema\_len} + Q \le \text{llama\_n\_ctx}$$
    If this is violated, it throws a `std::length_error` immediately to prevent cache overflow.
2.  **Position Bound & Suffix Recompute**:
    *   If `n_past <= 256`, the splice runs in the anchored-exact regime using a baseline recompute fraction $R_{\mathrm{base}} = 5\%$.
    *   If `n_past > 256`, the orchestrator checks if `deep_splice_enabled_` is active. If `false`, it declines the splice and returns `0` (falling back to standard text prefill). If `true`, it calculates the depth-adaptive fraction $R(n_{\mathrm{past}})$ to repair RoPE drift, and splices the block.
3.  **Layout Writability**: The splicer determines the physical V-cache layout. Row-major V-cache layouts (`v_trans = false`, FlashAttention enabled) are spliced via simple copies. Transposed V-cache layouts (`v_trans = true`, soft-capped architectures like Gemma-2 with FlashAttention disabled) are spliced using `splice_v_layer` (UMA-only). Mismatched or unsupported GPU layouts throw an exception.
4.  **RoPE Parameter Validation**: The `.atb` block's `rope_freq_scale` and `rope_scaling_type` must match the live model configuration within $10^{-5}$. A mismatch causes the orchestrator to reject the block.
5.  **Re-entrant Context Lock**: Serializes routing and argument generation to guarantee bit-stable, deterministic outputs under multi-threaded execution.
6.  **Lifecycle Cleanup**: Splicing is wrapped in a Python RAII context manager (`NexusSpliceContext`). On turn exit, the spliced KV cache cells are cleared (`unsplice_tool`) to restore the cache to its pre-splice state.

---

## 2. `.atb` File Format ABI

Each precompiled tool schema is written as an Aeon Tool Block (`.atb`). The file starts with a 128-byte packed little-endian header, followed by page-aligned (2 MiB) contiguous key and value tensors.

```cpp
#pragma pack(push, 1)
struct alignas(128) AeonToolBlockHeader {
    char magic[4];               // Magic signature ("ATB1")
    uint32_t version;            // ABI version (currently 1)
    uint64_t model_hash;         // FNV-1a hash of model topology & architecture
    uint32_t n_layer;            // Transformer layer count
    uint32_t n_head_kv;          // Number of KV heads (GQA-compatible)
    uint32_t d_head;             // Dimension per attention head
    uint32_t seq_len;            // Length of the precompiled schema (tokens)
    uint32_t base_pos;           // Absolute RoPE position at compile time
    float rope_freq_base;        // RoPE base frequency
    float rope_freq_scale;       // RoPE frequency scaling factor
    uint32_t rope_scaling_type;  // RoPE scaling type (0: None, 1: Linear, 2: YaRN)
    uint32_t ggml_type_k;        // GGML numerical type for K (F16 = 1)
    uint32_t ggml_type_v;        // GGML numerical type for V (F16 = 1)
    uint64_t k_tensor_offset;    // Byte offset to contiguous K data
    uint64_t v_tensor_offset;    // Byte offset to contiguous V data
    uint64_t k_total_bytes;      // Size of K tensor in bytes
    uint64_t v_total_bytes;      // Size of V tensor in bytes
    char padding[40];            // Pad to exactly 128 bytes
};
#pragma pack(pop)

static_assert(sizeof(AeonToolBlockHeader) == 128);
static_assert(offsetof(AeonToolBlockHeader, k_tensor_offset) == 56);
```

*   `model_hash` is computed as an FNV-1a hash over the model name string concatenated with layer count, head dimension, base frequency, and KV head counts. A mismatched hash is rejected at runtime.
*   Contiguous FP16 tensors are aligned to `NEXUS_PAGE_ALIGNMENT = 2097152` bytes to allow zero-copy memory mapping (`mmap`).

---

## 3. Physical KV Layout Mappings & `v_trans`

Splicing directly writes raw F16 tensors into the physical memory pages of the LLM context. The physical stride pattern depends on whether FlashAttention is enabled:

```
v_trans = false (FlashAttention enabled)
   K Cache: [n_layer, n_head_kv, seq_len, d_head]
   V Cache: [n_layer, n_head_kv, seq_len, d_head]  <-- Row-major

v_trans = true (FlashAttention disabled / Gemma-2 soft-capping)
   K Cache: [n_layer, n_head_kv, seq_len, d_head]
   V Cache: [n_layer, seq_len, n_head_kv, d_head]  <-- Transposed (token-innermost)
```

For `v_trans = true`, a standard row-major copy would corrupt attention heads. Nexus solves this via a custom strided transposed blit in `nexus_kv_splicer.cpp`:
```cpp
void splice_v_layer(const ggml_tensor* src, float* dest, int layer, int head_kv, int seq_len, int d_head) {
    // Layout-aware copy transposing the V matrix during write
    for (int t = 0; t < seq_len; ++t) {
        float* dest_ptr = dest + t * n_head_kv * d_head + head_kv * d_head;
        const float* src_ptr = src + head_kv * seq_len * d_head + t * d_head;
        std::memcpy(dest_ptr, src_ptr, d_head * sizeof(float));
    }
}
```
*   On UMA platforms, this strided copy runs with negligible overhead.
*   On discrete-GPU backends, strided host-to-device transfers are pathological. The splicer rejects `v_trans = true` and falls back to Path B (text prefill).

---

## 4. Empirical Provenance & Verification

The primary quantitative metrics in the paper are backed by verified JSON artifacts under [results/v2.0_canonical/raw/](file:///Volumes/AI_SSD/Projects/nexus/results/v2.0_canonical/raw/).

### 4.1 Next-Token Divergence Sweep
*   **Harness**: `test/bench_dkl_sweep.py`
*   **Artifact**: `dkl_sweep.json`
*   **Findings**: An anchored splice ($\Delta\mathrm{pos} = 0$) achieves $D_{\mathrm{KL}} = 0.0$ and top-1 agreement of $1.0$. Off-anchor splices plateau at $D_{\mathrm{KL}} \sim 10^{-2}$ nats across the $0$--$2048$ range, while LegoLink scattered-recompute spikes to $D_{\mathrm{KL}} \approx 5.7$ nats at depth $1024$.

### 4.2 Gating and Drift Goggles
*   **Harness**: `test/profile_head_drift.py`
*   **Artifact**: `gating_nogo.json`
*   **Findings**: The reference-free preceding-context K-variance proxy fails to correlate with true drift (Mean Spearman $\rho = 0.193$). Max per-head drift ranges between $175$ and $207$ across context depths.

### 4.3 Deep-Splice TTFT
*   **Harness**: `test/bench_v2_capstone.py`
*   **Artifact**: `deep_splice_ttft.json`
*   **Findings**:
    *   **K=4 curve**: TTFT speedup of $1.63\times$ at depth $256$ ($R=5\%$), narrowing to $1.24\times$ at $512$ ($R=36.7\%$), and $0.98\times$ (parity/minor regression) at $1024$ ($R=100\%$).
    *   **K=16 curve**: TTFT speedup of $1.73\times$ at $256$ ($R=5\%$), $1.42\times$ at $512$ ($R=11.3\%$), $1.17\times$ at $1024$ ($R=24\%$), and $1.07\times$ at $2048$ ($R=49.3\%$).

### 4.4 Routing Scalability & Accuracy
*   **Harness**: `test/bench_routing_accuracy.py`
*   **Artifact**: `routing_accuracy_n250.json`
*   **Findings**: End-to-end routing accuracy is $92\%, 90\%, 89\%, 89\%$ at $N=10, 50, 100, 250$ tools. Top-1 recall is $74\%$, and top-3 recall is $95\%$ at $N=250$. Pure C++ SIMD vector dot product scan timing is $8.25\ \mu s$ (median), while the in-situ routing including Python FFI boundary crossings is $17.6\ \mu s$ (median).

### 4.5 Hybrid Sidecar Argument Generation
*   **Harness**: `test/bench_sidecar_accuracy.py`
*   **Artifact**: `sidecar/accuracy.json`
*   **Findings**: Sidecar routing accuracy is $86.7\%$. Argument accuracy on routed cases is $100\%$ ($95\%$ Wilson lower bound $\ge 91.2\%$), with $100\%$ JSON validity. End-to-end argument accuracy is $80\%$. First-argument token latency is $443.8$ ms vs $737.3$ ms (baseline), representing a $1.66\times$ speedup at an $\approx 80\%$ token savings.

---

## 5. Developer Precautions & Common Traps

*   **V-trans Layout**: Do not assume all GGUF models are row-major. Gemma-2 models force FlashAttention off on the current `llama.cpp` stack due to soft-capping, which changes the layout of the V-cache to token-innermost.
*   **Rerank Margin Margin-Threshold Default**: If the directory `src/` is not in the Python search path, `nexus_retrieval.DEFAULT_RERANK_MARGIN` defaults to `0.028`. The correct calibrated margin is `0.013646852970123292` (defined in `src/nexus_calibration.py`). Always set `PYTHONPATH=build:src:test`.
*   **Radix Cache vs Radix FSM**:
    *   `NexusRadixPrefixCache` (`nexus_seq_warm_cache.cpp`) is the **L0 prefix cache** (caches KV history to avoid prefill).
    *   `NexusRadixFSM` (`nexus_fsm.cpp`) is the **logits-masking trie** (constrains generated tokens during tool resolution).
*   **L0 Prefix Cache Copy Overhead**: The radix cache prefix copy is an actual physical copy in memory (`llama_kv_cache_seq_cp`). It runs in $\approx 3.04\ \mu s$ (median) but is NOT zero-copy. Do not describe it as zero-copy.
*   **Benchmark Serial Executions**: Never run TTFT latency benchmarks in parallel. Parallel execution causes thread contention, inflating B3 baseline values and corrupting latency measurements.

---

## 6. Code Extension Map

| Target | Source File | Key Entry Point |
|--------|-------------|-----------------|
| **Routing / Gating** | `src/nexus_orchestrator.cpp` | `route_and_splice()` (margin gate, RoPE validation, deep-splice branch) |
| **KV Cache Transplantation** | `src/nexus_kv_splicer.cpp` | `inject_tool_page_raw()` (layout validations, blit calls, relative shifts) |
| **Transposed-V Blit** | `src/nexus_kv_splicer.cpp` | `splice_v_layer()` (strided transposition loop) |
| **RoPE Reanchoring Math** | `src/nexus_rope_math.cpp` | `apply_absolute_rope_reanchor()`, `apply_relative_rope_shift()` |
| **INT8 SIMD Dense Search** | `src/nexus_slb.cpp` | `search()`, `search_hybrid()` |
| **Radix prefix cache** | `src/nexus_seq_warm_cache.cpp` | `try_copy_prefix()`, `probe_prefix()`, `update_from_seq()` |
| **Logits Masking Radix FSM**| `src/nexus_fsm.cpp` | `advance()`, `get_logits_mask()` |
| **Python FFI / RAII lifecycle**| `src/nexus_agent.py` | `NexusSpliceContext`, `NexusRoutingProcessor` |
| **C++ / Python FFI Bindings** | `src/bindings.cpp` | nanobind registrations (`nexus_fsm_ext`) |
