# Phase H — H1 Second-Model Validation Review (Gemma2)

**Date:** 2026-06-20 · **Scope:** H1 only. No H2/H3/retraining/router/Path B/Tool IR/paper/README/architecture/llama.cpp-upgrade work.
**Outcome:** **NARROWING (hard, architecture-level).** The Path A splice mechanism cannot run on Gemma2 on the pinned stack — anchored/isolated fidelity is unmeasurable because KV injection is structurally blocked.

## 1. Executive Status
`gemma2:9b` was acquired, passed the load/ABI preflight (it loads under the pinned `cb2463bb`), and the H1 anchored-fidelity run was attempted. The splice **fails at `inject_tool_page`** with `NexusKVSplicer Error: Flash Attention must be enabled (v_trans == false)`. Root cause: Gemma2 uses attention logit soft-capping; the pinned llama.cpp forces flash-attention OFF when soft-capping is present, leaving the V cache transposed (`v_trans=true`), which the splicer rejects. No anchored fidelity, isolated fidelity, route-only, or TTFT number could be produced — injection is the first step and it is blocked. No fake comparison was constructed; no llama.cpp upgrade or splicer change was attempted (both out of scope).

## 2. Model Provenance
- Ollama id: `gemma2:9b` → `general.name = gemma-2-9b-it`.
- Local GGUF blob: `~/.ollama/models/blobs/sha256-ff1d1fc78170d787ee1201778e2dd65ea211654ca5fb7d69b5a2e7b123a50373`, 5,443,143,296 bytes.
- Quantization: `general.file_type = 2` (Q4_0). GGUF V3.
- Naming note: Ollama `gemma2:9b` resolves to the official `gemma-2-9b-it`; the model blob is a standalone GGUF loadable directly by `llama_cpp`.

## 3. ABI / Compatibility Preflight
Loads under pinned `cb2463bb`: **yes** (unlike `gemma4`). Field comparison:

| Field | Qwen2.5-14B (canonical) | gemma2:9b | Material for splice? |
|---|---|---|---|
| general.architecture | qwen2 | gemma2 | — |
| attention.head_count | 40 | 16 | GQA shape |
| attention.head_count_kv | 8 | 8 | KV heads (same) |
| **d_head (key/value_length)** | **128** (5120/40) | **256** (explicit) | **yes — `.atb` KV tensor layout `[n_head_kv, seq_len, d_head]`** |
| block_count | 48 | 42 | `.atb` n_layer |
| rope.freq_base | 1,000,000 | Gemma default (~10,000) | RoPE phase rate |
| attn_logit_softcapping | none | **50.0** | **yes — forces flash-attn OFF** |
| final_logit_softcapping | none | 30.0 | output only |
| sliding_window | none | 4096 (alternating) | attention mask semantics |

- **What had to be regenerated:** the harness recompiles model-bound `.atb` blocks for Gemma2 via `nexus_kv_compiler --model`; that step ran. The failure is downstream, at injection.
- **What broke the comparison:** **attention logit soft-capping.** Pinned llama.cpp log: `flash_attn is not compatible with attn_soft_cap - forcing off` → `flash_attn = 0` → `v_trans = true`. The splicer asserts `v_trans == false`. So the splice path is unreachable for Gemma2 regardless of `.atb` correctness.
- **Methodological cleanliness:** the comparison stays clean precisely by stopping here — forcing it would require disabling soft-capping (changes the model) or rewriting the splicer to support `v_trans==true` (architecture change, out of scope).

## 4. Results

| Dimension | Qwen2.5 (canonical) | gemma2:9b |
|---|---|---|
| Anchored fidelity (KL / top-1) | 0.0 / 1.00 | **BLOCKED** — `inject_tool_page` raises (v_trans) |
| Isolated fidelity | KL ~0.7 / top-1 ~0.48 | **NOT ATTEMPTED** — same prerequisite fails |
| Route-only latency | 160.2 ms (n=500) | **NOT VALID** — Path A splice unreachable; gateway would hit the same inject error |
| True TTFT | 518 ms (N1x, n=30) | **NOT VALID** — same reason |

No Gemma2 fidelity/latency numbers exist; all are blocked at the splice prerequisite.

## 5. Comparison to Canonical Qwen
- **Held:** model loads and `.atb` compilation runs under the pinned stack for Gemma2.
- **Changed/broke:** Qwen2.5 has no attention soft-capping → flash attention enables → `v_trans=false` → splicer works (anchored exact). Gemma2 has attention soft-capping → flash attention forced off → `v_trans=true` → splicer refuses. The splice mechanism is therefore **not portable to Gemma2** on this stack.
- **Narrowed:** the mechanism's applicability is bounded by a previously-unstated runtime prerequisite (flash-attention KV layout), which a whole architecture family (Gemma2/Gemma3, any model with attention logit soft-capping) violates.

## 6. Claim Impact
**Narrows the claim — concretely and at the architecture level.** Path A splice is not "transformer-general"; it requires a model that runs with flash attention enabled (`v_trans==false`), which excludes architectures using attention logit soft-capping (Gemma2 here, and by extension Gemma3). The validated mechanism is now explicitly scoped to **flash-attention-compatible, non-soft-capped architectures** (Qwen2.5 qualifies). H1 did not measure whether anchored-exact/isolated-degraded *numbers* generalize — it found that the splice cannot even execute on this second model — which is a stronger, structural narrowing than a numeric drift. The claim remains single-model-validated (Qwen2.5), now with a named architectural precondition.

## 7. Recommended Next Single Step
**H3 — n≥100 e2e + intervals.** H1 showed model-generality is architecturally constrained, so the entire value proposition now rests on the single validated configuration (Qwen2.5, this host). That configuration's headline (2.34× true TTFT, ~6-pt accuracy gap) is currently n=30 with wide CIs — doubly fragile (one config AND small n). H3 removes the small-n fragility and turns the one working result into a statistically defensible one. Not H2: a second host tests machine-drift but still on one model; given H1 just established the mechanism is architecture-bound, solidifying the magnitude/CIs of the only configuration that works is higher leverage than adding host coverage to it.

## 8. Stop Line
H1 is complete (negative/narrowing). No splicer change, no llama.cpp upgrade, no soft-cap disabling, and no H2/H3/retraining/router/Path B/Tool IR/paper/README/architecture work was performed or is authorized. Awaiting approval for the single recommended next step (H3).
