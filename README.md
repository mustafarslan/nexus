# Project Nexus v1.0: Hybrid C++23 Agentic Routing Engine and KV-Cache MMU

Project Nexus is a **Hybrid C++23 Agentic Routing Engine and KV-Cache MMU** on **llama.cpp**. It accelerates MCP tool routing by pre-compiling tool schemas into offline `.atb` KV blocks, routing via quantized retrieval and a radix trie FSM, and splicing KV into the active context within strict physical boundaries.

**v1.0 production headline:** Gateway TTFT P50 **171 ms**, E2E tool-hit **0.91** (n=100, Qwen2.5-14B-Instruct Q4_K_M). Source: [`results/bench_gateway_e2e_v2.json`](results/bench_gateway_e2e_v2.json).

**Honest speedup:** **2.8×** vs B3 retrieve-and-prefill (N1 **466 ms** vs B3 **1.32 s**). The **153×** bloat strawman is retired.

Release notes: [`RELEASE_V1.md`](RELEASE_V1.md) · Architecture: [`docs/architecture.md`](docs/architecture.md) · Internals: [`docs/internals.md`](docs/internals.md)

> **Models are not in Git.** Weights (`.gguf`), KV binaries (`.atb`), and finetuned checkpoints are gitignored. Train CE v3 locally before running gateway benchmarks:
>
> ```sh
> python3 scripts/train_cross_encoder.py --output results/tool_cross_encoder_finetuned_v3
> ```

---

## 1. The Prefill Compute Wall

| Baseline | TTFT P50 | Role |
|----------|----------|------|
| B1 (full bloat, 12.5k tokens) | **9.48 s** | Strawman — do not cite |
| B3 (retrieve + single schema) | **1.33 s** | **Honest baseline** |
| GW_route (gateway production) | **171 ms** | **v1.0 headline** |
| N1 (splice + 5% suffix) | **466 ms** | Fast path at P≤256 |

---

## 2. Dual-Path Engine

### Path A ($P \le 256$)

POSIX-mapped `.atb` splice + **5% suffix recompute** + FSM decode. KL mean **0.0076** @ P=256.

### Path B ($P > 256$)

Splice blocked (RoPE physics). Fallback:

1. **Exact-token LCP radix cache** — zero-allocation LCRS tree (`node_arena_` + `token_arena_`), `llama_kv_cache_seq_cp` at **~3 µs** P50
2. Text prefill via `decode_tokens`

**No FNV-1a chunking on the L0 hit path** — purged in Phase 3 DOD refactor.

```python
MAX_SPLICE_POS = 256  # nexus_agent.py
```

---

## 3. Production Pipeline

```
User Query
  → Hybrid Dense Retrieval (< 5 µs)
  → Transient Intent Signatures (dense/CE only)
  → P20 Margin Gate → CE v3 Rerank (~20% fire rate)
  → P ≤ 256? → KV Splice + 5% Suffix | L0 LCP Radix / Text Prefill
  → FSM Logit-Masked Decode → GBNF Args
```

| Method | Recall@1 |
|--------|----------|
| Dense hybrid | **0.87** |
| CE v3 (P20 gated) | **0.90** |

Source: [`results/recall_miss_analysis_v2.json`](results/recall_miss_analysis_v2.json)

P20 threshold **0.01365** in [`src/nexus_calibration.py`](src/nexus_calibration.py). The 0.15 static clamp is audit-only — dropped from routing.

---

## 4. What We Do Not Ship

| Feature | Evidence |
|---------|----------|
| ColBERT MaxSim | P50 **709 µs**, Recall@1 **0.72** — [`bench_phase22_maxsim.json`](results/bench_phase22_maxsim.json) |
| LegoLink @ P=1024 | KL **5.72** — [`g4_gate_verdict.json`](results/g4_gate_verdict.json) |
| Blockmask N1m orchestrator | E2E accuracy **0.0** — [`bench_n1m_fidelity_blockmask.json`](results/bench_n1m_fidelity_blockmask.json) |
| FNV-1a 32-token chunking | 0% L0 hit rate — replaced by exact-token LCP radix |
| P>256 ATB splice | RoPE boundary — orchestrator enforced |

---

## 5. Canonical Benchmark Artifacts

Only these files live in `results/` (see [`results/README.md`](results/README.md)):

| File | Proves |
|------|--------|
| `bench_gateway_e2e_v2.json` | **171 ms** / **0.91** |
| `bench_e2e_v2.json` | B1 **9.48 s**, B3 **1.33 s** |
| `bench_e2e.json` | N1 **466 ms**, 2.8× vs B3 |
| `bench_phase28_radix_prefix_v2.json` | L0 **~3 µs** copy |
| `recall_miss_analysis_v2.json` | M3 CE gate |
| `margin_calibration_p20.json` | P20 audit trail |
| `g4_gate_verdict.json` | G4 physics FAIL |
| `bench_phase22_maxsim.json` | ColBERT graveyard |
| `bench_n1m_fidelity_blockmask.json` | Blockmask graveyard |
| `bench_phase21_ttft_real.json` | TTFT decomposition |

---

## 6. Build

```bash
git submodule update --init --recursive
pip install -r requirements.txt && pip install nanobind
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
export PYTHONPATH="$(pwd)/build:$(pwd)/src"

# Calibrate P20 margin gate
python3 scripts/calibrate_margin_threshold.py

# Train CE v3 (required for gateway/recall benches)
python3 scripts/train_cross_encoder.py --output results/tool_cross_encoder_finetuned_v3
```

| Target | Description |
|--------|-------------|
| `nexus_fsm_ext` | Python extension |
| `nexus_kv_compiler` | Offline `.atb` compiler |
| `bench_phase28_radix_prefix` | L0 radix microbench |
| `bench_phase22_concurrent_hazard` | Block cache hazard test |

---

## 7. Quick Start

```python
from nexus_agent import NexusAgent
from nexus_retrieval import CrossEncoderReranker

agent = NexusAgent(
    llm=llm, embedding_llm=embed_llm, dim=768,
    max_splice_pos=256,
    # rerank_margin=None → loads CALIBRATED_MARGIN_THRESHOLD from nexus_calibration.py
)
agent.configure_reranker(cross_encoder=CrossEncoderReranker(
    "results/tool_cross_encoder_finetuned_v3"
))

with agent.splice_context(query, seq_id=0) as ctx:
    resolved_id, tool_name, args_json = agent.generate_with_tool(query, seq_id=0)
```

---

## 8. Citation

> Gateway TTFT P50 **171 ms**, tool-hit **0.91**, vs B3 **1.33 s** (~**7.8×**), n=100, Qwen2.5-14B-Instruct Q4_K_M, git `1384a33b`.

Do not cite the 153× strawman or deep-context splice at $P \ge 1024$.
