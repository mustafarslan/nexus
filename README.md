# Nexus: Depth-Adaptive KV-Cache Splicing & Retrieval-Decoupled Tool Routing

[![arXiv](https://img.shields.io/badge/arXiv-2608.20397-b31b1b.svg)](https://arxiv.org/abs/2608.20397)

Nexus is a high-performance serving prototype designed to accelerate tool-augmented turns in agentic Large Language Models (LLMs) running on unified memory architectures (UMA). 

In standard Model Context Protocol (MCP) implementations, LLMs must re-encode verbose JSON schemas on every turn. The quadratic cost of prefill dominates the Time-to-First-Token (TTFT) as the tool registry grows. Nexus addresses this bottleneck by:
1. **Decoupling routing from prefill**: An INT8 Semantic Lookaside Buffer (SLB) with a calibrated margin gate resolves tool calls via embedding retrieval. Arguments are generated over a compressed textual signature (median 19 tokens) rather than the full schema.
2. **Depth-adaptive KV-cache splicing**: pre-compiled schemas are transplanted directly into the active KV cache cells. Position drift caused by Rotary Position Embeddings (RoPE) is corrected via an anchored suffix re-decode that dynamically scales with depth, guaranteeing exact output fidelity ($D_{\mathrm{KL}} \approx 0$).

---

## Key Results (v2.0 Canonical)

All quantitative measurements are recorded on the primary evaluation platform:
*   **Hardware**: Apple M4 Max SoC (16-core CPU, 40-core GPU, 16-core Neural Engine), 64 GB Unified Memory.
*   **Software**: Qwen2.5-14B-Instruct Q4_K_M + nomic-embed-text-v1.5. GGUF backend powered by `llama.cpp` at commit `cb2463bb`.
*   **Logs & Artifacts**: [results/v2.0_canonical/](file:///Volumes/AI_SSD/Projects/nexus/results/v2.0_canonical/)

| Dimension | Metric / Finding | Status / Value |
|-----------|------------------|----------------|
| **Routing Accuracy** | Flat accuracy up to $N=250$ tools (baseline overflows) | **89%** |
| **TTFT Latency** | Median TTFT speedup at moderate depth | **$1.1\times$ to $1.7\times$** |
| **Fidelity Guarantee** | Output correctness compared to full prefill | **$D_{\mathrm{KL}} \approx 0$, top-1 identical** |
| **SLB Routing Overhead**| INT8 SIMD scan latency over 250 tools | **17.6 $\mu s$** (incl. FFI) / **8.25 $\mu s$** (C++) |
| **L0 Radix warm-hit** | Prefix-reuse cache copy speed | **3.04 $\mu s$** (69.5% hit rate) |
| **Drift Prediction** | Reference-free K-variance proxy gate | **Failed** (Spearman $\rho = 0.193$) |

---

## Repository Map

*   [`src/`](file:///Volumes/AI_SSD/Projects/nexus/src): Core implementation
    *   [`nexus_orchestrator.cpp`](file:///Volumes/AI_SSD/Projects/nexus/src/nexus_orchestrator.cpp): Core scheduling, margin gating, and dual-path execution logic.
    *   [`nexus_kv_splicer.cpp`](file:///Volumes/AI_SSD/Projects/nexus/src/nexus_kv_splicer.cpp): Strided KV transplantation, including layout-aware transposed-V copies for soft-capped attention layouts.
    *   [`nexus_slb.cpp`](file:///Volumes/AI_SSD/Projects/nexus/src/nexus_slb.cpp): INT8 SIMD vector retrieval engine.
    *   [`nexus_rope_math.cpp`](file:///Volumes/AI_SSD/Projects/nexus/src/nexus_rope_math.cpp): RoPE phase alignment and reanchoring calculations.
    *   [`nexus_seq_warm_cache.cpp`](file:///Volumes/AI_SSD/Projects/nexus/src/nexus_seq_warm_cache.cpp): Radix cache for prefix-sharing.
    *   [`nexus_agent.py`](file:///Volumes/AI_SSD/Projects/nexus/src/nexus_agent.py): Python orchestrator interface and grammar-constrained argument parser.
*   [`test/`](file:///Volumes/AI_SSD/Projects/nexus/test): Benchmark drivers and correctness checks.
*   [`docs/`](file:///Volumes/AI_SSD/Projects/nexus/docs): Architectural specification and implementation details.
*   [`results/v2.0_canonical/`](file:///Volumes/AI_SSD/Projects/nexus/results/v2.0_canonical/): Bit-exact replication artifacts and evidence ledger.

---

## Replication

To compile the C++ bindings and reproduce each benchmark locally, see the step-by-step instructions in [REPRODUCE.md](file:///Volumes/AI_SSD/Projects/nexus/REPRODUCE.md).

---

## Scope & Architectural Preconditions

*   **Attention Layouts**: Splicing directly modifies the physical KV cache. Flash-attention compatible models (e.g. Qwen2.5) use row-major cache layouts. Soft-capped models (e.g. Gemma-2) disable Flash-Attention on this stack, storing the V-cache in a transposed state. Splicing Gemma-2 relies on a layout-aware transposed-V strided copy (UMA-only).
*   **Hardware limitations**: Because splicing writes directly into Metal/GPU cache lines, execution is restricted to local unified memory (UMA). Cloud/Remote APIs or discrete GPUs without zero-copy address mapping fall back automatically to standard text prefill.

---

## Citation

If you use this work, please cite the preprint:

```bibtex
@article{arslan2026nexus,
  title={Nexus: Depth-Adaptive KV-Cache Splicing and Retrieval-Decoupled Tool Routing for Agentic LLMs on Unified Memory},
  author={Arslan, Mustafa},
  journal={arXiv preprint arXiv:2608.20397},
  year={2026}
}
```
