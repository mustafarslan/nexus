# Nexus

Nexus is a research prototype for **low-latency tool routing in LLM agents**. It cuts the
time-to-first-token (TTFT) of a tool-augmented turn by skipping the most expensive part of
that turn — re-prefilling the chosen tool's schema into the model's KV cache — and instead
**splicing a precompiled KV block** (the tool's "page") directly into the cache. A small
suffix is recomputed to stitch the splice to the live context, and a calibrated retrieval +
cross-encoder gate decides *which* tool to splice.

On the one configuration where the mechanism is validated end-to-end, this delivers a
**2.47× median true-TTFT speedup with no detectable accuracy gap** at n=100. The rest of this
document is about exactly what that claim covers and what it does not.

---

## The core idea

A normal tool-using turn pays for the tool's schema twice over: once to retrieve it, and again
to *prefill* it — run every schema token through the transformer to populate the KV cache —
before the model can emit its first output token. Prefill is compute-bound and grows with
schema length, so for realistic tool schemas it dominates TTFT.

Nexus precompiles each tool's schema into an **`.atb` file** (Aeon Tool Block): a frozen,
model-bound snapshot of the K and V tensors that a normal prefill *would* have produced. At
runtime, instead of prefilling, Nexus:

1. **Retrieves** the likely tool with an INT8 SIMD dense router (the SLB) over tool embeddings.
2. **Gates** the decision: if the top-1 margin is confident it auto-routes; otherwise a
   calibrated cross-encoder reranks before committing.
3. **Splices** the chosen tool's `.atb` KV directly into the live cache.
4. **Recomputes a short suffix** (~5%) so the spliced block is coherent with the surrounding
   context, then decodes the first token.

The expensive schema prefill never runs. That is the entire source of the speedup — and also
the entire source of the mechanism's constraints, because a precompiled KV block is only valid
under tight architectural and positional preconditions.

## Why it matters

For agentic workloads where every turn invokes a tool, prefill latency is a recurring tax on
interactivity. Caching the *result* of prefill rather than recomputing it is the obvious move;
the hard part is doing it correctly — KV state is position-dependent (RoPE) and
architecture-dependent (attention layout). Nexus is an honest study of where that trade is
real and where it breaks.

---

## Evidence status

All validated numbers come from a **single canonical tuple**, frozen 2026-06-20 under
`results/v1.1_canonical/`:

| Dimension | Value |
| --- | --- |
| Model | Qwen2.5-14B-Instruct, Q4_K_M (`model_hash a09ea5e7`) |
| Stack | `llama_cpp` / `llama.cpp` build `cb2463bb` |
| Host | one Apple-Silicon machine |
| Code | git `3ea7441`, current harness definitions |

**Validated** (canonical artifacts under `results/v1.1_canonical/phaseH/`):

- **2.47× median true-TTFT speedup**, N1x vs B3, n=100 (H3).
- **No detectable accuracy gap** at n=100 (H3).
- **Anchored splice is output-exact** (H5 fidelity).
- **Router scales to N≤100 tools under 5 µs** (H5 scalability).

**Not validated:**

- Any second model or second host (all data is one tuple).
- Portability beyond flash-attention-compatible, non-soft-capped architectures (see Scope).
- Path B (deep-context, P>256) as an end-to-end accelerated path — it is wired but only ever
  falls back to text prefill.
- The 0.90 cross-encoder Recall@1 target — measured 0.88, not met.

**Retired** (do **not** cite as live evidence):

- The tensor-KL boundary curve `0.0076 @P256` / `5.72 @P1024` — no tensor-KL harness exists;
  see [internals](docs/internals.md).
- Prior route-only headlines `171 ms` and `237.4 ms` — neither reproduces on the host.
- The earlier n=30 "6-point accuracy gap" — small-n noise; gone at n=100.

---

## Headline result

The headline is **true TTFT** — wall-clock to the first generated token, including first-token
decode — measured serially, n=100 per arm:

| Arm | What it is | Median true-TTFT |
| --- | --- | --- |
| **N1x** (Path A) | retrieve → gate → ATB splice → 5% suffix recompute | **457.5 ms** |
| **B3** (B_RP) | retrieve → full text prefill of the schema | **1131.4 ms** |

- **Speedup: 2.47×**, 95% CI **[2.41, 2.72]**.
- **Accuracy delta (N1x − B3): −0.010**, 95% CI **[−0.080, +0.050]**; McNemar not significant.
  This is "no detectable gap at n=100" — *not* a claim of equal accuracy.

Artifact: [`h3_e2e_n100.json`](results/v1.1_canonical/phaseH/h3_e2e_n100.json).

> **Route-only is not TTFT.** A separate gateway benchmark times retrieval + gate + splice with
> `decode_us = 0` and reports **160.2 ms** (n=500). That is *route-only latency*, useful for
> profiling the routing path, and it is never the headline. TTFT is always the H3 number above.

### Compact benchmark summary

| Metric | Value | Kind | Artifact |
| --- | --- | --- | --- |
| True-TTFT speedup (N1x vs B3) | 2.47× [2.41, 2.72], n=100 | true TTFT | `phaseH/h3_e2e_n100.json` |
| Accuracy delta (N1x − B3) | −0.010 [−0.080, +0.050] | accuracy | `phaseH/h3_e2e_n100.json` |
| Gateway routing+splice P50 | 160.2 ms, n=500 | **route-only** | canonical manifest |
| Dense Recall@1 | 0.87 | retrieval | `phaseH/recall_miss_analysis_H4.json` |
| CE-gated Recall@1 (~20% fire) | 0.88 (target 0.90 not met) | retrieval | `phaseH/recall_miss_analysis_H4.json` |
| Anchored splice fidelity | logit-KL 0.0, top-1 1.0, n=50 | fidelity | `phaseH/splice_fidelity_anchored_H5.json` |
| SLB scan P50 @ N=100 | 3.71 µs (<5 µs at N≤100) | router cost | `phaseH/slb_scalability_H5.csv` |
| L0 warm-cache copy P50 | ~3.06 µs, 69.5% hit | cache | canonical manifest |

---

## Architecture in one screen

```
query ──► SLB dense router (INT8 SIMD) ──► margin gate ──► [confident?] ──► auto-route
                                              │                              │
                                              └─ uncertain ─► CE rerank ─────┤
                                                                             ▼
                                            n_past ≤ 256 ?  ──yes──► Path A: splice .atb KV
                                                  │                          + 5% suffix recompute
                                                  └──no───► Path B: decline → text prefill (fallback)
                                                                             ▼
                                                                       first token
```

- **Dual-path gate.** Path A (the accelerated path) is taken only when the live context is
  short enough (`n_past ≤ MAX_SPLICE_POS = 256`) that a precompiled KV block can be spliced
  without unacceptable RoPE position drift. Beyond that, Nexus declines and falls back to
  ordinary text prefill (Path B).
- **Anchored splice.** A `.atb` compiled at the same base position it is injected at
  (`Δpos = 0`) is **output-exact**. Off-anchor (isolated) splices degrade. Path A relies on
  the anchored regime.
- **Calibrated routing.** A P20 margin threshold (`τ ≈ 0.01365`) fires the cross-encoder on
  roughly the least-confident 20% of decisions; the rest auto-route from the dense router alone.

Full design: [docs/architecture.md](docs/architecture.md). Implementation truth:
[docs/internals.md](docs/internals.md).

---

## Scope, limitations, non-goals

**Architectural precondition (hard).** The splice path requires a
**flash-attention-compatible, non-soft-capped** attention architecture. Concretely: Gemma2
uses attention logit soft-capping, which forces flash attention *off* on the pinned stack,
which leaves the V cache transposed (`v_trans = true`), which the splicer rejects. The splice
cannot even execute on Gemma2 — this is a structural block, not a tuning gap. **Qwen2.5 is the
only validated working configuration.** Nexus Path A is *not* transformer-general.

**One tuple.** Every validated number is one model, one host, one stack. No portability across
models or hardware is claimed.

**Path B is a fallback, not an accelerator.** Deep-context turns (P>256) decline the splice and
text-prefill. The L0 warm tier is *probed* there for measurement only; it is not consumed.

**Routing accuracy is ceiling-bound.** CE-gated Recall@1 is 0.88; the 0.90 target was not
reachable by gate tuning alone (H4).

**Non-goals.** Nexus is not a production gateway, not a general KV-cache library, and not a
serving framework. It is a prototype that isolates and measures one specific latency trade.

---

## Repo map

| Path | What lives there |
| --- | --- |
| `src/nexus_orchestrator.{cpp,hpp}` | `route_and_splice`: routing, gate, dual-path decision |
| `src/nexus_kv_splicer.{cpp,hpp}` | `inject_tool_page_raw`: low-level KV splice, `v_trans` check |
| `src/nexus_slb.{cpp,hpp}` | Semantic Load Balancer — INT8 SIMD dense router |
| `src/nexus_seq_warm_cache.{cpp,hpp}` | L0 warm prefix cache (`NexusRadixPrefixCache`) |
| `src/nexus_rope_math.{cpp,hpp}` | RoPE reanchor kernels for off-anchor splices |
| `src/aeon_tool_block.hpp` | `.atb` 128-byte header ABI |
| `src/nexus_agent.py` | Python FFI, RAII splice context, routing processor |
| `src/bindings.cpp` | nanobind bindings (`nexus_fsm_ext`) |
| `docs/architecture.md` | system design and dataflow |
| `docs/internals.md` | implementation invariants, ABI, artifact provenance |
| `results/v1.1_canonical/` | frozen canonical evidence (manifest is source of truth) |
| `results/v1.1_canonical/phaseH/` | H1/H3/H4/H5 reviews + primary JSON/CSV artifacts |

## Reproduction orientation

Benchmarks run only against the pinned stack and model on the canonical host:

- Use the project venv: `.venv/bin/python` (system `python3` cannot import `llama_cpp`).
- Set `PYTHONPATH=build:src:test`.
- Run end-to-end (e2e) benchmarks **serially** — concurrent runs load the 14B model twice and
  inflate absolutes.
- The canonical manifest (`results/v1.1_canonical/CANONICAL_RESULTS_MANIFEST.json`) is the
  authority on which artifacts are live; superseded files carry `_deprecation` stamps and
  pristine copies live in `results/_quarantine_2026-06-20/`.

See [docs/internals.md](docs/internals.md) for full provenance and the artifact taxonomy.

---

## What we claim — and what we do not

**We claim:**

- On the canonical tuple, Path A reaches first token **2.47×** faster than full-prefill B3
  (n=100, CI-backed).
- At n=100 there is **no detectable accuracy gap** between the two.
- Anchored splices are **output-exact**; the router is **cheap** (<5 µs at N≤100).

**We do not claim:**

- That this generalizes to other models, hosts, or attention architectures.
- That accuracy is *equal* (we claim no *detectable* gap at this sample size).
- That route-only latency (160.2 ms) is TTFT.
- The retired boundary curve, the old route-only headlines, or the n=30 accuracy gap.
