# Phase H — Documentation Rewrite Review

Review of the from-scratch rewrite of `README.md`, `docs/architecture.md`, and
`docs/internals.md` against the Phase-H canonical evidence base.

## Executive Status

The three primary docs were fully replaced (not patched) and now form one coherent suite built
on a single ground truth: the canonical tuple (Qwen2.5-14B Q4_K_M, `llama.cpp cb2463bb`, one
Apple-Silicon host, git `3ea7441`). The headline is now the H3 n=100 **true-TTFT 2.47×**
result; scope is explicitly narrowed to flash-attention-compatible, non-soft-capped
architectures (H1); the legacy tensor-KL boundary curve is retired everywhere; and route-only
latency is consistently separated from true TTFT. Shared vocabulary, benchmark story, scope
story, and architecture story are aligned across all three files. **Status: complete and
internally consistent.**

## Files Rewritten

| File | Role / entry point | Result |
| --- | --- | --- |
| `README.md` | Front door — evaluator | Rewritten: core idea, evidence status (validated/not/retired), H3 headline, scope, repo map, claim/non-claim section |
| `docs/architecture.md` | Mental model — contributor | Rewritten: end-to-end flow, routing/gate, ATB/splice design, anchored vs isolated, Path A success + Gemma2 failure, performance model, metric→artifact map, failure modes |
| `docs/internals.md` | Implementation truth — extender | Rewritten: invariants, `.atb` ABI table, `v_trans`/flash-attn dependence, H1/H3/H5 meaning, artifact taxonomy, developer footguns, extension map |
| `results/v1.1_canonical/phaseH/PHASE_H_DOC_REWRITE_REVIEW.md` | This review | New |

## Major Content Changes

- **New headline.** True-TTFT 457.5 ms (N1x) vs 1131.4 ms (B3), 2.47× [2.41, 2.72], n=100,
  replacing all prior route-only-as-TTFT framing.
- **Route-only vs TTFT split made explicit.** Gateway 160.2 ms is labeled route-only everywhere
  and explicitly never headlined; the "6.3×" unlike-quantities error is called out and retired.
- **Scope narrowed structurally.** H1's flash-attn/`v_trans`/soft-cap chain is documented as a
  hard precondition in all three files; "transformer-general" framing removed; Qwen2.5 stated as
  the only validated config.
- **Boundary curve retired.** `0.0076 @P256` / `5.72 @P1024` removed as live evidence; replaced
  by the anchored-exact / isolated-degraded regime story in logit-KL, with tensor-L2 (~12,
  position-independent) noted; "no tensor-KL harness exists" stated plainly.
- **Accuracy framing corrected.** "No detectable gap at n=100" (Δ −0.010, CI straddles 0,
  McNemar n.s.) replaces the n=30 "6-point gap"; explicitly *not* claimed as equal accuracy.
- **Terminology disambiguated.** `NexusRadixPrefixCache` (L0 warm KV cache) vs `NexusRadixFSM`
  (decode-time masking trie) separated; "zero-copy" dropped in favor of "warm copy via
  `llama_kv_cache_seq_cp`"; SLB defined as Semantic Load Balancer (dense router), not a server LB.

## Benchmark / Evidence Alignment

| Live claim | Value | Artifact |
| --- | --- | --- |
| True-TTFT speedup | 2.47× [2.41, 2.72], n=100 | `phaseH/h3_e2e_n100.json` |
| Accuracy delta | −0.010 [−0.080, +0.050], McNemar n.s. | `phaseH/h3_e2e_n100.json` |
| Scope precondition | flash-attn-compatible, non-soft-capped; Qwen2.5-only | `phaseH/PHASE_H_H1_SECONDMODEL_REVIEW.md` |
| SLB scalability | 1.375→21.1 µs (N=10→1000), 3.71 µs @ N=100 | `phaseH/slb_scalability_H5.csv` |
| Anchored fidelity | logit-KL 0.0, top-1 1.0, n=50 | `phaseH/splice_fidelity_anchored_H5.json` |
| Retrieval | dense 0.87 / CE-gated 0.88 (0.90 not met) | `phaseH/recall_miss_analysis_H4.json` |
| Calibration | τ = 0.013646852970123292 | `phaseH/margin_calibration_H4.json` |
| Route-only (not TTFT) | 160.2 ms, n=500 | `CANONICAL_RESULTS_MANIFEST.json` |
| L0 warm cache | 69.5% hit, ~3.06 µs P50 | `CANONICAL_RESULTS_MANIFEST.json` |

Retired and confirmed absent as live claims: `171 ms`, `237.4 ms`, `2.34×`/`518`/`1214` (n=30),
`6.3×`, `0.0076`/`5.72` tensor-KL, `153×` strawman, "transformer-general", "zero-copy",
"production-ready", "eliminates prefill entirely", "equivalent accuracy".

## Scope / Limit Changes

- Mechanism scope reduced from implied transformer-generality to **one validated tuple** with a
  named architectural precondition.
- Path B reclassified explicitly as a **fallback, not an accelerator** (deep-path L0 is
  read-only/probed, never consumed or written back).
- Routing accuracy documented as **ceiling-bound** (0.88; 0.90 unreachable by gate tuning, H4).
- Off-anchor splice viability and cross-model generalization labeled **hypothesis /
  unvalidated**, not claims.

## Remaining Follow-ups

- **`RELEASE_V1.md`** still carries v1.0 framing and some legacy headline language; out of scope
  for this rewrite, should be aligned to the H3/H1 story or marked historical.
- **`docs/paper/`** (`main.tex`, `sections/`, `tables/`, `figures/`) — paper headline and any
  figures derived from retired numbers (boundary curve, old TTFT) still need alignment; figures
  remain legacy.
- **`results/README.md`** — verify it points at the canonical manifest and does not restate
  retired numbers.
- **Second-model / second-host evidence** — none exists; any future portability claim requires a
  new tuple and a re-run of H1-style structural checks first.
- **Path B acceleration** — currently unproven; if pursued, the deep-path L0 read-only probe is
  the place to start measuring whether a real save exists.

## Stop Line

Rewrite of the three target docs plus this review is complete and self-consistent. No source
code, benchmarks, or canonical artifacts were modified. Out-of-scope files (RELEASE notes,
paper, results README) were intentionally left untouched and are listed above as follow-ups.
Stopping here.
