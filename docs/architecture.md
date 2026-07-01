# Nexus Architecture

This document is the mental model for Nexus: what the system is, how a request flows through
it, where the latency goes, and — just as importantly — where the design's preconditions bind.
It pairs with [internals.md](internals.md) (implementation truth) and the
[README](../README.md) (the headline and scope).

**Validation anchor.** Everything benchmarked here is the canonical tuple: Qwen2.5-14B-Instruct
Q4_K_M, `llama_cpp`/`llama.cpp` build `cb2463bb`, one Apple-Silicon host, git `1ce4aa4`,
frozen under `results/v2.0_canonical/`.

---

## 1. System overview

Nexus accelerates the *tool-augmented turn* in an LLM agent. The unit of work is: a user query
arrives, the agent must pick a tool, load that tool's schema into the model's context, and
produce a first output token. The cost of that turn is dominated by **prefill** — running the
tool's schema tokens through the transformer to populate the KV cache.

Nexus's thesis is that prefill of a *fixed* tool schema is redundant work: the KV tensors it
produces are deterministic given the model and the schema's position, so they can be computed
once, frozen to disk, and **spliced** back into the cache at runtime instead of recomputed.
Around that splice sits a routing pipeline that decides which tool to splice, and a gate that
decides how much to trust the cheap router before committing.

The system is two cooperating layers:

- A **C++ hot path** (orchestrator, splicer, router, caches) that does the latency-critical
  work and is bound tightly to the model's KV layout.
- A **Python control layer** (`nexus_agent.py`) that drives llama.cpp, owns the RAII lifecycle
  of a splice, and exposes the routing decision to a logits processor.

## 2. Design goals

1. **Eliminate redundant schema prefill** on the common (short-context) path.
2. **Never silently corrupt output.** A splice that cannot be done correctly must be declined,
   not approximated. This is why the dual-path gate and the architectural rejection checks
   exist.
3. **Make routing cheap and calibrated.** The router must cost microseconds, and the decision
   to escalate to a more expensive reranker must be statistically grounded, not hand-tuned.
4. **Be measurable and honest.** Every claim maps to a frozen artifact; route-only and true
   TTFT are kept distinct; scope is stated, not implied.

## 3. Core concepts and terminology

These names are used identically here, in the README, and in internals.

| Term | Meaning |
| --- | --- |
| **`.atb` (Aeon Tool Block)** | A precompiled, model-bound KV snapshot of a tool schema. A "tool page." |
| **Path A / N1x** | The accelerated path: retrieve → gate → splice `.atb` → 5% suffix recompute. |
| **Path B / B3 (B_RP)** | Fallback / baseline: retrieve, then full **text prefill** of the schema. |
| **Anchored splice (Δpos=0)** | `.atb` injected at the base position it was compiled for. Output-exact. |
| **Isolated / off-anchor splice** | `.atb` injected at a different position (Δpos≠0). Requires RoPE reanchor; degrades. |
| **SLB** | Semantic Load Balancer — an INT8 SIMD **dense vector router** over tool embeddings. Not a server load balancer. |
| **CE** | Cross-encoder reranker, fired only on low-margin decisions. |
| **P20 margin calibration** | The margin threshold `τ ≈ 0.01365` that gates CE firing at ~20% of queries. |
| **L0 warm cache** | `NexusRadixPrefixCache`: a runtime KV prefix-reuse tier. Distinct from the decode-time FSM trie. |
| **route-only latency** | Time for retrieve + gate + splice with `decode_us = 0`. Not TTFT. |
| **true TTFT** | Wall-clock to the first generated token, first-token decode included. The headline metric. |

## 4. End-to-end flow (query → answer)

```
                ┌─────────────────────────── Python control layer ───────────────────────────┐
 user query ─►  │ embed query ─► route_and_splice(FFI) ─► LogitsProcessor (FSM mask) ─► decode │
                └───────────────────────────────────┬──────────────────────────────────────────┘
                                                     ▼  (C++ hot path)
   1. GUARD        n_past > 256 ?  ── yes ─► Path B: decline, probe L0 (read-only), return 0
                       │ no
   2. RETRIEVE     SLB dense (or hybrid dense+lexical) search → top-3 candidates + scores
   3. MARGIN       margin = score[0] − score[1]
   4. GATE         passes_threshold = score[0] ≥ θ  OR  margin ≥ auto_route_margin
                   passes_margin    = margin ≥ speculative_margin
   5. SPECULATE    if (passes_threshold ∧ passes_margin): speculatively inject top-1 .atb early
   6. RESOLVE      else: decode tool name through Radix FSM (logit-masked) → resolved tool_id
   7. SPLICE       inject resolved tool's .atb KV  (anchored: direct blit; off-anchor: RoPE reanchor)
   8. RECOMPUTE    recompute last ~5% of schema tokens to stitch splice to live context
                                                     ▼
                                              first token decoded
```

The Python layer (`NexusSpliceContext`) wraps steps 5–8 in a context manager so that the
spliced KV range is always cleaned up (`unsplice_tool`) on turn exit, even on exception or
cancellation. The decision data from step 6 flows into `NexusRoutingProcessor`, a llama.cpp
`LogitsProcessor` that applies the FSM mask **only** while the model is decoding the tool name.

## 5. Retrieval / routing pipeline

Routing is a cheap-first, escalate-on-doubt cascade.

### 5.1 Dense router (SLB)

The SLB holds each tool's embedding quantized to INT8 with a per-vector scale, plus optional
lexical hashes and a tokenized "digest." A query is quantized the same way and scored by a
branchless SIMD dot product (AVX2 / AVX-512-VNNI / NEON). `search_hybrid` blends the dense
score with lexical overlap when query hashes are present; `search` is dense-only. Both return
the top-3. This is the entire retrieval cost on the hot path, and it is **microseconds** (§12).

### 5.2 Margin and the gate

The orchestrator computes `margin = score[0] − score[1]`. Two predicates decide whether the
top-1 can be trusted without a reranker:

- `passes_threshold`: top score clears an absolute bar **or** the margin clears an auto-route bar.
- `passes_margin`: the margin clears a speculative bar.

If both hold, Nexus **speculatively injects** the top-1 tool page before the model has even
decoded a tool name — buying back the splice latency on the (calibrated-likely) correct guess.
If not, it defers to the FSM.

### 5.3 CE reranking

When the dense margin is below the calibrated threshold, a cross-encoder reranks the candidates
before the tool is committed. The CE is the expensive, accurate arm; the whole point of
calibration is to fire it rarely.

### 5.4 FSM tool resolution

When the gate does not auto-route, the tool name is decoded through `NexusRadixFSM`, a radix
trie over tool-name tokens. During the `NAVIGATING` state it masks logits to only the valid
continuation tokens, so the model cannot hallucinate a non-existent tool. This is a separate
structure from the L0 warm cache — see internals; do not conflate the two.

## 6. CE / calibration / gate behavior

The gate's threshold is not hand-picked. A P20 margin calibration sets
`τ = 0.013646852970123292` so that the cross-encoder fires on roughly the **least-confident
20%** of decisions. H4 measured the consequences:

- Dense Recall@1: **0.87**.
- CE-gated Recall@1 at ~20% fire: **0.88** — the 0.90 target was **not met**.
- A fire-rate sweep showed *raising* the fire rate **lowers** accuracy (0.88 → 0.85): broad CE
  application hurts, and the P20 gate is near-optimal for this stack.

Artifacts: `phaseH/margin_calibration_H4.json`, `phaseH/recall_miss_analysis_H4.json`,
`phaseH/recall_sweep_thr_*.json`.

> **Footgun.** If `src/` is not on the import path, the margin threshold silently falls back to
> `0.028` instead of the calibrated `0.01365`. See internals.

## 7. ATB / tool-page / splice design

An `.atb` file is a 128-byte header followed by contiguous FP16 K and V tensors — exactly the
KV state a normal prefill would have produced for that schema, frozen to disk. The header binds
the block to a specific model topology and RoPE configuration (full ABI in internals).

At runtime the splicer (`inject_tool_page_raw`):

1. Validates the block against the live model (layer count, KV heads, head dim, RoPE params)
   and rejects any block whose KV layout it cannot legally write (the `v_trans` check, §10).
2. Computes the physical KV-cache row for the logical position, handling circular-buffer wrap.
3. Decides anchored vs off-anchor from `delta_pos = n_past − base_pos`.
4. Transfers the K/V tensors into the cache — a parallel per-layer blit on UMA, or async H2D on
   discrete GPUs.
5. For off-anchor injections, applies a RoPE reanchor (CPU kernel or GPU path).

A short suffix (~5% of the schema, computed by the orchestrator) is then recomputed so the
spliced block is coherent with whatever preceded it in the live context.

## 8. Anchored vs isolated splice — what the regimes mean

This is the conceptual heart of the fidelity story.

- **Anchored (Δpos = 0).** The block is injected at exactly the position it was compiled for.
  No RoPE adjustment is needed; the KV is bit-for-bit what prefill would have produced. H5
  measured this directly: **logit-KL 0.0, top-1 agreement 1.0 across n=50.** The output
  distribution is indistinguishable from a real prefill.
- **Isolated / off-anchor (Δpos ≠ 0).** The block must be RoPE-reanchored to its new position.
  H5 found this **degrades**: logit-KL ~0.9, top-1 agreement ~0.33–0.67. The reanchor does not
  recover prefill-equivalent state.

The practical consequence: Path A is built to keep splices in the anchored regime, and the
dual-path gate exists to refuse positions where the splice would have to leave it.

> **The old graceful-boundary story is retired.** Earlier docs described a smooth tensor-KL
> curve (`0.0076 @P256` rising to `5.72 @P1024`). That curve is **not a live claim**: no
> tensor-KL harness exists in the repo, and the numbers do not reproduce in any current metric.
> What survives is the binary regime story above — anchored exact, off-anchor degraded —
> measured in **logit/output-distribution KL** and corroborated by position-independent
> **tensor-L2** (~12, no boundary effect). See internals for the full retirement.

## 9. Runtime constraints and architectural preconditions

Path A is correct only inside a box defined by three preconditions:

1. **Position.** `n_past ≤ MAX_SPLICE_POS = 256`. Beyond this, the RoPE drift required to place
   the block is too large to stay in the anchored-exact regime, so Nexus declines (Path B).
2. **RoPE configuration match.** The block's `rope_freq_scale` and `rope_scaling_type` must
   match the live model within tolerance, or the orchestrator throws.
3. **Attention architecture.** The KV layout must be one the splicer can legally write — which
   on the pinned stack means flash-attention-compatible and non-soft-capped (§11).

## 10. Why Path A works on the validated tuple

Qwen2.5 on the pinned stack runs with flash attention enabled, which keeps the V cache in the
non-transposed layout (`v_trans = false`) that the splicer writes into. Its RoPE configuration
is stable and matchable, and tool schemas compiled at `base_pos = 0` are injected on the short
path where `Δpos = 0`. All three preconditions hold, so the splice lands in the anchored-exact
regime — which is exactly what H5's fidelity measurement and H3's end-to-end speedup confirm.

## 11. Why it narrows / fails on Gemma2

H1 tried a second model and found a **structural** block, not a numeric regression:

- Gemma2 uses **attention logit soft-capping**.
- The pinned `llama.cpp` forces **flash attention off** when soft-capping is present.
- With flash attention off, the V cache is left **transposed** (`v_trans = true`).
- The splicer **rejects** `v_trans = true` — it cannot legally write that layout.

So the splice cannot even *execute* on Gemma2. This is why the scope is stated as
**flash-attention-compatible, non-soft-capped architectures**, and why **Qwen2.5 remains the
only validated working configuration**. H1 did not measure whether the anchored/isolated
fidelity numbers generalize; it found the mechanism is structurally inapplicable to this second
family. That is a stronger narrowing than a drift would be. Artifact:
`phaseH/PHASE_H_H1_SECONDMODEL_REVIEW.md`.

## 12. Performance model and trade-offs

Where the latency goes, and which metric measures which part:

| Stage | Cost (canonical tuple) | Metric family |
| --- | --- | --- |
| SLB dense scan | ~3.71 µs @ N=100 (<5 µs at N≤100; ~21 µs at N=1000) | router cost (H5) |
| L0 warm-cache copy | ~3.06 µs P50 (69.5% hit) | cache |
| Retrieve + gate + splice | 160.2 ms P50 (`decode_us=0`) | **route-only** |
| Full turn to first token, Path A (N1x) | **457.5 ms** | **true TTFT** (H3) |
| Full turn to first token, Path B (B3) | **1131.4 ms** | **true TTFT** (H3) |

The trade Path A makes: it spends a few hundred microseconds of routing + a KV blit +
a 5%-suffix recompute, to avoid a full schema prefill. On the canonical tuple that nets a
**2.47×** true-TTFT speedup (CI [2.41, 2.72]) with **no detectable accuracy gap** (Δ −0.010,
CI [−0.080, +0.050]).

Critically, **route-only (160.2 ms) and true TTFT (457.5 ms) are different quantities.**
Dividing one baseline's full TTFT by the other arm's route-only latency is the error that
produced the retired "6.3×" figure. The only honest speedup is true-TTFT vs true-TTFT: 2.47×.

## 13. Evidence-backed claims vs design hypotheses

| Statement | Status |
| --- | --- |
| Anchored splice is output-exact | **Evidence** (H5 fidelity, logit-KL 0.0) |
| Path A is 2.47× faster TTFT than B3, no detectable accuracy gap | **Evidence** (H3, n=100) |
| Router is <5 µs at N≤100 | **Evidence** (H5 scalability) |
| Splice requires flash-attn-compatible, non-soft-capped arch | **Evidence** (H1, structural) |
| Off-anchor / deep-context splice could be made viable with a better RoPE reanchor | **Hypothesis** (unproven; off-anchor currently degrades) |
| L0 warm tier would save real work on the deep path | **Hypothesis** (only probed read-only; never consumed) |
| Mechanism generalizes to other models/hosts | **Unvalidated** (one tuple only) |

## 14. Benchmark map (metric → artifact)

| Claim | Phase | Artifact |
| --- | --- | --- |
| True-TTFT speedup 2.47× [2.41,2.72], n=100; accuracy Δ −0.010 | H3 | `phaseH/h3_e2e_n100.json` (+ `PHASE_H_H3_E2E_INTERVAL_REVIEW.md`) |
| Gemma2 structural block; Qwen2.5-only scope | H1 | `phaseH/PHASE_H_H1_SECONDMODEL_REVIEW.md` |
| Dense 0.87 / CE-gated 0.88; 0.90 not met | H4 | `phaseH/recall_miss_analysis_H4.json`, `recall_sweep_thr_*.json` |
| P20 calibration τ = 0.01365 | H4 | `phaseH/margin_calibration_H4.json` |
| SLB scan 1.375→21.1 µs (N=10→1000) | H5 | `phaseH/slb_scalability_H5.csv` |
| Anchored fidelity logit-KL 0.0, top-1 1.0 | H5 | `phaseH/splice_fidelity_anchored_H5.json` |
| Tensor-L2 / logit-KL boundary (replaces tensor-KL) | H5 | `phaseH/tensor_boundary_canary_5pct.json`, `logit_kl_boundary_phaseA.json` |
| Gateway route-only 160.2 ms; L0 69.5% / 3.06 µs | canonical | `CANONICAL_RESULTS_MANIFEST.json` |

## 15. Failure modes and risk boundaries

- **Wrong tool routed.** Bounded by CE-gated Recall@1 ≈ 0.88; routing is not perfect and the
  ceiling was confirmed in H4.
- **Position out of range.** `n_past > 256` → splice declined → text-prefill fallback (correct
  but unaccelerated).
- **Architecture mismatch.** Soft-capped / non-flash-attn model → splicer rejects → mechanism
  inapplicable (Gemma2).
- **RoPE config mismatch.** Block vs live RoPE params disagree → orchestrator throws rather than
  splice incorrectly.
- **Off-anchor use.** Injecting a block away from its compiled position degrades the output
  distribution; the design avoids this rather than relying on the reanchor to fix it.
- **Single-tuple generalization risk.** All evidence is one model/host/stack; treating it as
  general is unsupported.
