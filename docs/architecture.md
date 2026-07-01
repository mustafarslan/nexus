# Nexus Architecture: Depth-Adaptive Splicing and Decoupled Routing

This document defines the system topology, execution paths, and performance characteristics of Nexus. It reconciles the system design with the physical limits of Rotary Position Embedding (RoPE) drift and the hardware constraints of Unified Memory Architectures (UMA).

---

## 1. System Overview & The Dual Levers

Nexus is a local LLM serving prototype optimized for tool-augmented agentic turns. In standard systems standardizing on the Model Context Protocol (MCP), the model is presented with verbose JSON schemas that must be parsed and re-encoded every turn. Nexus eliminates this overhead through two complementary system-level levers:

1.  **Retrieval-Decoupled Routing (Primary Lever)**: Decouples tool selection from schema prefill. A Semantic Lookaside Buffer (SLB) selects the active tool via fast vector search. Arguments are generated in the main context over a type-hinted, compressed textual signature (median 19 tokens), avoiding prompt bloat and keeping registry scale independent of prefill latency.
2.  **Depth-Adaptive KV-Cache Splicing (Secondary Lever)**: Transplants precompiled tool schema key-value blocks (stored as `.atb` files) directly into the active KV cache cells. To prevent rotary position embedding (RoPE) phase drift, a trailing suffix fraction $R(n_{\mathrm{past}})$ of the schema is dynamically re-decoded at runtime to stitch the seam, achieving a never-regress fidelity guarantee ($D_{\mathrm{KL}} \approx 0$).

```
                     ┌──────────────────────────────────────────────┐
                     │                 User Query                   │
                     └──────────────────────┬───────────────────────┘
                                            ▼
                             ┌──────────────────────────────┐
                             │    INT8 Embedding Search     │
                             │       (SLB Dense Scan)       │
                             └──────────────┬───────────────┘
                                            ▼
                             ┌──────────────────────────────┐
                             │      Margin Gate (P20)       │
                             │       margin = s0 - s1       │
                             └──────────────┬───────────────┘
                                            │
                        ┌───────────────────┴───────────────────┐
                        ▼ (margin >= τ)                         ▼ (margin < τ)
             ┌─────────────────────┐                 ┌─────────────────────┐
             │     Auto-Route      │                 │     CE Rerank       │
             │  (Speculative top-1)│                 │    (20.8% queries)  │
             └──────────┬──────────┘                 └──────────┬──────────┘
                        └───────────────────┬───────────────────┘
                                            ▼
                                 ┌─────────────────────┐
                                 │  Resolved Tool ID   │
                                 └──────────┬──────────┘
                                            ▼
                                ┌──────────────────────┐
                                │   n_past > 256 ?     │
                                └───────┬──────┬───────┘
                                        │      │
                              yes ┌─────┘      └─────┐ no (Anchored regime)
                                  ▼                  ▼
                    ┌─────────────────────────┐   ┌──────────────────────────┐
                    │  deep_splice_enabled?   │   │  Direct .atb KV Splice   │
                    └────┬───────────────┬────┘   │  + 5% suffix recompute   │
                         │               │        └──────────┬───────────────┘
                  yes ┌──┘               └──┐ no             │
                      ▼                     ▼                ▼
         ┌─────────────────────────┐  ┌───────────┐  ┌───────────────────────┐
         │ Calculate R(n_past) Eq.2│  │ Text      │  │ Constrained Arguments │
         │   Blit .atb KV Splice   │  │ Prefill   │  │     via GBNF/FSM      │
         │   + Suffix Re-decode    │  │ (Path B)  │  └──────────┬────────────┘
         └────────────┬────────────┘  └─────┬─────┘             │
                      └─────────┬───────────┘                   ▼
                                ▼                       ┌──────────────┐
                          First Token                   │  Tool Call   │
                            Decoded                     └──────────────┘
```

---

## 2. Core Concepts & Terminology

*   **`.atb` (Aeon Tool Block)**: A frozen, page-aligned FP16 KV cache snapshot of a precompiled schema, containing a 128-byte header storing topology and RoPE anchors.
*   **Semantic Lookaside Buffer (SLB)**: An in-situ INT8 vector lookup register that caches tool names and descriptions, executing branchless SIMD scans on NEON/AVX.
*   **Anchored Splice ($\Delta\mathrm{pos} = 0$)**: Splicing a tool page at the exact base position it was compiled for (typically position 0 in a sidecar context). This path is output-exact ($D_{\mathrm{KL}} = 0$, top-1 agreement 1.0).
*   **Off-Anchor Splice ($\Delta\mathrm{pos} \ne 0$)**: Splicing a tool page at an arbitrary context depth. Requires RoPE re-rotation of keys and values from their compile anchor to $n_{\mathrm{past}}$.
*   **Depth-Adaptive Recompute**: The schedule $R(n_{\mathrm{past}})$ that determines what fraction of the spliced block's trailing tokens must be re-decoded to stich the seam and drive position drift error to zero.
*   **Never-Regress Guarantee**: A system invariant ensuring that the output next-token probability distribution of a spliced turn is mathematically identical to a full text prefill reference.
*   **transposed-V Layout**: An attention cache layout where values are stored in a token-innermost stride (common in soft-capped models like Gemma-2 when FlashAttention is disabled).

---

## 3. Retrieval and Routing Cascade

To minimize latency overhead, tool selection is structured as a hierarchical routing cascade:

1.  **SIMD Dense Scan**: The user query is embedded and quantized to INT8, then compared against all registered tool signatures in the SLB. On Apple UMA, the pure C++ SIMD scan runs in **$8.25\ \mu s$** for 250 tools, rising to **$17.6\ \mu s$** when including Python FFI boundary crossings.
2.  **Margin Gating**: The margin between the top two candidates ($m = s_0 - s_1$) is compared to a calibrated margin threshold $\tau = 0.0136$. If $m \ge \tau$, the router has high confidence and auto-routes the query using speculative top-1 cache injection.
3.  **Cross-Encoder Gating**: If $m < \tau$, the decision is flagged as low-confidence and escalated to a fine-tuned MiniLM-class cross-encoder. This reranking step fires on **$20.8\%$** of queries (P20 calibration), preserving accuracy while shielding the hot path from the costly cross-encoder forward pass.
4.  **Radix FSM Resolution**: If auto-routing is skipped, the model decodes the tool name through a radix trie logits mask (`RadixFSM`), ensuring the model cannot generate invalid tool names.

Overall routing accuracy remains nearly flat as the registry scales: **$92\%$** at $N=10$ tools, **$90\%$** at $N=50$, and **$89\%$** at $N=250$. Under the standard concatenate-all baseline, the prompt overflows the context window entirely at $N \ge 50$.

---

## 4. KV-Cache Splicing Mechanics

When a tool is resolved, its `.atb` file is mapped into memory. Splicing proceeds as follows:

```
.atb File (Disk)          Host Virtual Memory (mmap)          Live KV Cache
┌──────────────┐          ┌────────────────────────┐          ┌─────────────────┐
│ Header (128B)│ ──mmap─► │ Header (128B) [ignored]│          │ History (n_past)│
├──────────────┤          ├────────────────────────┤          ├─────────────────┤
│ Contig K F16 │          │ Contig K F16 [zero-copy] ──blit──►│ Spliced K Block │
├──────────────┤          ├────────────────────────┤          │ (Row-Major)     │
│ Contig V F16 │          │ Contig V F16 [zero-copy] ──blit──►├─────────────────┤
└──────────────┘          └────────────────────────┘          │ Spliced V Block │
                                                              │ (Transposed/Str)│
                                                              ├─────────────────┤
                                                              │ Seam re-decode  │
                                                              │ S * R(n_past)   │
                                                              └─────────────────┘
```

1.  **Header Verification**: The runtime top-1 block's parameters (layer count, KV heads, head dimension, and RoPE frequencies) are validated against the live model to prevent memory corruption.
2.  **Layout-Aware Blitting**:
    *   For standard architectures (e.g. Qwen2.5) where FlashAttention is active, the V-cache is row-major. Contiguous blocks are blitted directly.
    *   For soft-capped architectures (e.g. Gemma-2), FlashAttention is disabled, leaving the V-cache in a transposed layout (`v_trans = true`). Splicing executes a layout-aware strided blit (`splice_v_layer`) to transpose values during physical cache writing.
3.  **RoPE Reanchoring**: For off-anchor relocations ($\Delta\mathrm{pos} \ne 0$), keys are re-rotated by the phase difference:
    $$\Delta\theta_i = (n_{\mathrm{past}} - m_0)\theta_i$$
4.  **Seam Stitching**: A trailing fraction $R(n_{\mathrm{past}})$ of the schema block is invalidated and re-decoded to smooth the relative attention transition between the history and the newly spliced block.

---

## 5. Depth-Adaptive Splicing and the Never-Regress Curve

Relocating compiled blocks causes attention disruption due to positional RoPE context drift. While an anchored splice at $\Delta\mathrm{pos} = 0$ is output-exact ($D_{\mathrm{KL}} = 0$), off-anchor splices accrue a residual next-token divergence of $D_{\mathrm{KL}} \sim 10^{-2}$ nats.

Rather than accepting this drift, Nexus enforces a **never-regress guarantee** by scaling the recompute suffix fraction $R(n_{\mathrm{past}})$ linearly with context depth beyond a threshold $M = 256$:

$$R(n_{\mathrm{past}}) = R_{\mathrm{base}} + \frac{n_{\mathrm{past}} - M}{M(K - 1)}(100 - R_{\mathrm{base}})$$

where $R_{\mathrm{base}} = 5\%$. At deep context depths, the recompute fraction converges to $100\%$ (a full text prefill), driving next-token KL divergence to exactly $0$. 

### TTFT Performance Trade-offs
Because the repaired suffix adds re-decode work, the speedup decreases as depth increases:

*   **Moderate depth ($n_{\mathrm{past}} = 256$)**: Saves $1.63\times$ (default $K=4$) to $1.73\times$ (tuned $K=16$) TTFT.
*   **Intermediate depth ($n_{\mathrm{past}} = 1024$)**: Speedup narrows to $0.98\times$ (default $K=4$) or $1.17\times$ (tuned $K=16$).
*   **Deep context ($n_{\mathrm{past}} = 2048$)**: Converges to prefill parity ($1.00\times$ to $1.07\times$).

The tuning constant $K$ serves as a dial trading TTFT flatness for how early full recompute engages to preserve the never-regress contract.

---

## 6. Bounding Negative Results

The architecture is bounded by two distinct physical limits discovered during development:

### 6.1 The Failure of Reference-Free Drift Gating
We profiled a cheap, reference-free proxy (per-head preceding-context K-variance) to predict when a deep splice would drift, hoping to skip recompute when variance was low. However, this proxy failed to rank-correlate with true per-head drift (Mean Spearman $\rho = 0.193$, well below the target $\ge 0.40$). Consequently, Nexus relies on the deterministic depth-adaptive recompute curve rather than scalar-threshold gating.

### 6.2 Scattered Recompute Degradation (LegoLink)
An earlier design iteration attempted to repair drift by recomputing scattered, sparse tokens across the spliced block (LegoLink). This scattered recompute actively corrupted attention, leaving the next-token divergence at $D_{\mathrm{KL}} \approx 5.7$ nats at depth 1024 (two orders of magnitude worse than a bare contiguous splice). The production splicer has retired LegoLink, enforcing contiguous suffix re-decoding instead.
