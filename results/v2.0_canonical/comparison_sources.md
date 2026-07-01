# Comparison Sources (Phase 2 / P0-1)

Native numbers only — **no cross-hardware normalization**. Every non-Nexus row is *contextual*
(different HW/workload) and marked † in the paper table; none is a controlled head-to-head.
Nexus's row is the only one measured on the M4 host in this work.

| System | Mechanism | KV strategy | Routing | Reported metric (native) | Workload / HW | Source |
|---|---|---|---|---|---|---|
| **RAG-MCP** | Retrieve relevant MCP schemas before prefill | in-place (prompt-level; no KV relocation) | dense retrieval, top-K then prefill | tool-selection acc **43.13%** vs 13.62% baseline; prompt tokens **>50%** reduction | MCP stress test / benchmark tasks (LLM eval) | arXiv:2505.03275 |
| **vLLM / PagedAttention** | Paged KV, eliminates fragmentation | in-place (never relocates blocks) | — | **2–4×** throughput vs FasterTransformer/Orca | ShareGPT/Alpaca, NVIDIA A100/A10G | Kwon et al., SOSP'23 |
| **SGLang / RadixAttention** | Prefix KV reuse via radix tree | in-place (prefix sharing) | — | up to **6.4×** throughput | LLM programs, A10G/A100 | Zheng et al., NeurIPS'24 |
| **vAttention** | Dynamic KV memory, contiguous virtual + fragmented physical | in-place (no relocation) | — | up to **1.23×** throughput vs PagedAttention kernels | NVIDIA GPU serving | arXiv:2405.04437, ASPLOS'25 |
| **RedKnot** | Head-aware KV: position-independent reuse, prefix compression, hot/cold split | in-place (per-head policy, no relocation) | — | qualitative resource-efficiency gains (no headline number in abstract) | GPU serving (unspecified in abstract) | arXiv:2606.06256 |
| **Nexus (this work)** | Relocate per-tool compiled KV blocks (splice) + retrieval-decoupled routing | **relocates** coarse per-tool blocks | INT8 SLB + calibrated cross-encoder margin gate | routing **89%** @ 250 tools; first-arg TTFT **1.66×**; deep-splice **1.1–1.7×** | GitHub-MCP, Apple M4 Max UMA, Qwen2.5-14B-Q4_K_M | this paper (`results/v2.0_canonical/`) |

## Positioning notes
- **RAG-MCP is the named routing baseline.** Nexus adopts the same *retrieval principle* but adds a
  system-level lever (KV transplantation) and a zero-shot calibrated margin gate over frozen
  embeddings. The RAG-MCP metrics above are on a different model/workload — contextual, not a
  controlled head-to-head (left to future work).
- **The KV-serving systems (vLLM, SGLang, vAttention, RedKnot) keep KV in place**; none relocates a
  block to a new position, which is exactly the regime Nexus characterizes. Their throughput numbers
  are on discrete-GPU hardware and are not comparable to Nexus's UMA TTFT figures.
- All borrowed numbers trace to the arXiv/venue cited; do not normalize across hardware.
