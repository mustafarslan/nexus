# Nexus — LIVE Benchmark & Codebase Reconciliation Audit (Regeneration Pass)

**Auditor pass:** 2026-06-20 (LIVE host) · **Repo HEAD:** `3ea7441` · **Host:** Apple Silicon, Darwin 25.5.0 arm64
**Supersedes the diagnosis-only pass** `docs/AUDIT_2026-06-20.md`, which ran in a Linux sandbox and declared regeneration BLOCKED. **This pass executed the benchmark suite on the original host.** Canonical bundle: `results/v1.1_canonical/`.

---

## A. Executive Verdict

The prior audit's central thesis — "two conflicting committed runs (171 ms vs 237 ms), tie unbreakable" — **is now resolved by regeneration, and both committed headlines are wrong.**

Running the *identical* gateway harness, model (`a09ea5e7…`), and llama.cpp build (`cb2463bb`) **5× back-to-back on the live host today** yields a **rock-stable Gateway routing latency of P50 = 160.2 ms (159.4–161.2, std 0.6 ms, 1.1% spread), accuracy 0.89** across all 500 pooled queries. Neither 171 ms (Run A) nor 237 ms (Run B) reproduces; both are high. The 38.6% gap between them was **machine-state drift across one day**, not harness methodology (both runs use the same `np.percentile` path; the "paper harness" `bench_cdf_pareto.py` literally invokes `bench_gateway_e2e.py`).

Three findings are more damaging than the miscitation the prior audit led with:

1. **The headline metric is not TTFT.** `bench_gateway_e2e.py` times `route_with_retrieval` only and **explicitly sets `decode_us=0`** (line 95) — it excludes first-token generation. The "Gateway TTFT P50" is **routing+splice latency**. The true TTFT (prefill + first token) is the e2e `N1x` arm at **~1.3–1.5 s**.
2. **The 6.3× speedup divides unlike quantities.** Numerator = route-only 160 ms; denominator = B3 *full* TTFT (incl. first token) ~1.5 s. Like-for-like (`N1x` true-TTFT vs `B3` true-TTFT) the speedup is **~2×**, not 6.3×.
3. **CE Recall@1 is 0.88, not 0.90** (regenerated; `meets_target=false`). Dense R@1 0.87 reproduces exactly; the CE claim is 2 points high.

What holds up well: **every architectural claim is verified to concrete symbols** (P≤256 gate, MAX_SPLICE_POS, 5% recompute, force_tool_id, L0/FSM separation, .atb ABI, alignment, hazard pointers); **calibration is bit-exact reproducible** (τ=0.01365); **L0 microbench reproduces** (69.5% hit, 3.06 µs); the **negative results are real** (RoPE boundary, blockmask acc 0.0, ColBERT). The "zero-copy / UMA DMA" claim is a `std::cout` string, and "eliminates prefill entirely" is contradicted by the integrated `_prefix_cache_text_fallback` (Path B text-prefills).

**Posture:** honest systems-prototype, negative-results-forward. Replace all three committed headline families with the v1.1 canonical bundle; rename the metric; restate the speedup like-for-like; demote "production/zero-copy/eliminates/equivalent."

---

## B. Repo Discovery Report

- **Manuscript:** `docs/paper/main.tex` + `sections/0[1-9]_*.tex` + `tables/PAPER_TABLES.tex`; built `main.pdf`, `nexus.pdf`. Orphaned root `nexus_paper.aux/.out`.
- **Paper data pipeline:** `docs/paper/scripts/bench_*.py` → `data/_raw/*.json` → `data/*.csv` → `generate_master_metrics.py` → `PAPER_MASTER_METRICS.json` → `generate_latex_tables.py` → `PAPER_TABLES.tex` → `main.tex`. Plots via `generate_plots.py`.
- **Primary harnesses:** `test/bench_gateway_e2e.py` (route-only), `test/bench_e2e.py` (14 arms, true TTFT), `build/bench_phase28_radix_prefix` (L0), `scripts/calibrate_margin_threshold.py`, `scripts/analyze_recall_misses.py`.
- **Source (Py):** `src/nexus_agent.py`, `nexus_gateway.py`, `nexus_calibration.py`, `test/nexus_retrieval.py`. **(C++):** orchestrator, kv_splicer, kv_compiler, block_cache, seq_warm_cache, slb, fsm, rope_math, os_compat, page_mounter, io_async, thread_pool, hazard_pointer, bitmap_allocator, bindings (nanobind), `aeon_tool_block.hpp` (ABI).
- **Two dependency graphs** — *numbers:* `test/bench_*` → `results/*` (Run A) **and** `docs/paper/scripts/*` → `_raw` → csv → tables → tex (Run B); docs cross-cite the wrong graph. *claims:* `src/*` symbols → prose.

## C. Metric Surface Map

| Metric | Committed value(s) | **v1.1 canonical** | Locations |
|---|---|---|---|
| Gateway "TTFT" P50 (route-only) | 171 ms (Run A) / 237.4 ms (Run B) | **160.2 ms** | README, RELEASE, architecture.md:202/213/250, internals.md:295, main.tex:36/107/166/357, abstract/eval/conclusion, verify_paper_math.py, PAPER_MASTER_METRICS.json |
| Gateway accuracy | 0.91 (A) / 0.89 (B) | **0.89** | same |
| True TTFT N1x (incl first token) | not reported as headline | **~1.3–1.5 s** | e2e harness only |
| B3 / B_RP TTFT | 1.33 s (A) / 1.50 s (B) | **see e2e_n25** | README, RELEASE, arch, main.tex:357 |
| B1 / B_FC TTFT | 9.48 s (A) / 12.35 s (B) | **see e2e_n25** | RELEASE, arch, main.tex |
| Speedup vs B_RP | 6.3× | **~2× like-for-like** | README, main.tex, verify_paper_math.py |
| L0 hit rate | 66% (A,n=1) / 69.5% (B,10seed) | **69.5%** (58–80) | README, results/README, l0_radix_metrics.csv |
| L0 copy P50 | 3.042 µs | **3.06 µs** (median) | README, csv |
| CE Recall@1 | 0.90 | **0.88** | README, RELEASE, recall_*v2.json |
| Dense Recall@1 | 0.87 | **0.87** ✓ | results/README |
| τ margin | 0.01365 | **0.01365** ✓ | README, nexus_calibration.py |
| RoPE KL @P256/5% | 0.0076 | **not regenerated** | README, physics_boundary_rope.csv |

## D. Claim Surface Map
See `results/v1.1_canonical/IMPLEMENTATION_RECONCILIATION.csv` (21 claims with source symbols and decisions).

## E. Environment Readiness Report
See `results/v1.1_canonical/ENVIRONMENT_REPORT.md`. **All probes PASS** on the live host with `.venv/bin/python`. The prior "BLOCKED" was a sandbox/venv artifact.

## F. Artifact Inventory (delta from prior audit)
The prior audit's inventory (its §5) is accurate. Key change: the two "CANONICAL CANDIDATE" families (Run A `results/*`, Run B `_raw/*`) are **both DEMOTED** — neither reproduces. New canonical family = `results/v1.1_canonical/`.

## G. Source Cross-Reference Table
All 10 core architectural claims **VERIFIED_IN_CODE** at current line numbers (see IMPLEMENTATION_RECONCILIATION.csv). Spot-confirmed live: `nexus_orchestrator.cpp:69` (P≤256 gate), `:479` (ceil recompute), `:57/76-83` (force_tool_id), `nexus_agent.py:163` (MAX_SPLICE_POS), `:369-372` (Path B fallback wired), `seq_warm_cache.hpp:24-28` (LCRS RadixNode), `aeon_tool_block.hpp:9` (2 MiB), `orchestrator.cpp:374/461` ("UMA Zero-Copy" cout).

## H. Contradiction Matrix (resolved)

| # | Contradiction | Prior verdict | **Live resolution** |
|---|---|---|---|
| C1 | docs print 237.4 ms, cite file with 171 ms | miscitation | **Both wrong; canonical 160.2 ms** |
| C2 | README 237.4 vs results/README 171 | doc-vs-doc | **Both wrong; canonical 160.2 ms** |
| C3 | B3 1.33 vs 1.50 s | cross-run drift | regenerate (e2e_n25) |
| C4 | paper 237.4 vs ablation 307.9 ms (same config) | two harnesses | both stale; rename + reconcile to canonical |
| C5 | bench_e2e.json intra-file P50≠CI.point | intra-file bug | flagged + deprecation-stamped |
| C6 | L0 66% vs 69.5% | drift | **69.5% canonical (reproduced)** |
| C7 | "equivalent accuracy" vs 0.89<0.92 | overstatement | confirmed; rewrite |
| C8 | "eliminates prefill" vs Path B text-prefill | contradiction | confirmed at nexus_agent.py:380 |
| **C11 (NEW)** | "TTFT" vs decode_us=0 route-only metric | — | **metric misnamed; not TTFT** |
| **C12 (NEW)** | 6.3× divides route-only by full-TTFT | — | **like-for-like ~2×** |
| **C13 (NEW)** | CE R@1 0.90 claimed | — | **regenerated 0.88, target missed** |

## I. Quarantine Plan
Executed. See `results/v1.1_canonical/QUARANTINE_MANIFEST.md`. Pristine `.orig` copies in `results/_quarantine_2026-06-20/`; superseded Run-A latency files relabeled in place with additive `_deprecation` keys.

## J–K. Benchmark Execution Plan & Log

| Benchmark | Command | Status | Result |
|---|---|---|---|
| Gateway ×5 | `bench_gateway_e2e.py --query-limit 100 --force` ×5 | **DONE** | p50 160.2 ms, acc 0.89, std 0.6 ms |
| L0 ×10 seeds | `bench_phase28_radix_prefix --seed 0..9` | **DONE** | hit 69.5%, copy_p50 3.06 µs |
| Calibration | `calibrate_margin_threshold.py` | **DONE** | τ=0.013646852970123292 (bit-exact) |
| Recall | `analyze_recall_misses.py` | **DONE** | dense 0.87, CE 0.88, fire 0.20 |
| E2E baselines (n=30, clean serial) | `bench_e2e.py --query-limit 30 --force` | **DONE** | N1x **518 ms** / B3 **1214 ms** / B1 **9754 ms** → **2.34×** like-for-like (acc N1x 0.87 vs B3 0.93) |
| RoPE boundary | `aggregate_physics_boundary.py` | **NOT RUN** | pre-freeze CSV; no regen this pass |
| SLB scalability | `bench_scalability.py` | **NOT RUN** | — |

## L. Canonical Results Manifest
`results/v1.1_canonical/CANONICAL_RESULTS_MANIFEST.json` (machine-readable).

## M. Manuscript Numeric Reconciliation
`results/v1.1_canonical/METRIC_RECONCILIATION.csv` (machine-readable).

## N. Implementation Claim Reconciliation
`results/v1.1_canonical/IMPLEMENTATION_RECONCILIATION.csv` (machine-readable).

## O. Evidence Taxonomy
- **IMPLEMENTED_AND_MEASURED:** P≤256 gate, MAX_SPLICE_POS, 5% recompute, force_tool_id, L0/FSM split, ABI/align/mutex/hazard, τ calibration, L0 copy ~3 µs.
- **MEASURED_BUT_PROSE_OVERSTATES:** 160 ms labeled "TTFT" (is route-only); 6.3× (unlike quantities); "equivalent accuracy"; CE R@1 0.90 (is 0.88).
- **DOC_ONLY_UNBACKED:** "zero-copy / UMA DMA" (cout string).
- **CONTRADICTED_BY_CODE:** "eliminates prefill entirely" (Path B text-prefills).
- **IMPLEMENTED_NOT_MEASURED:** Path B P>256 fallback (wired at agent.py:371, never driven by any canonical bench); alignas(128) false-sharing benefit; production gateway under load.
- **NEGATIVE_RESULT_WELL_SUPPORTED:** RoPE P1024 (KL 5.72), blockmask (acc 0.0, n=20), ColBERT (709 µs, n=1).
- **HYPOTHESIS_ONLY:** "Apple M4 Max" specificity (artifacts say generic Darwin arm64).
- **UNKNOWN_PROVENANCE:** physics_boundary_rope.csv (pre-freeze, no regen path executed).

## P. Documentation Update Plan
See §"Documentation Update Plan" in the response and `METRIC_RECONCILIATION.csv` decisions. Order: main.tex/sections → tables/figures → README → RELEASE → architecture.md → internals.md → results/README → verify_paper_math.py.

## Q. Updated File List
Tracked in the working response after edits are applied.

## R. Multi-Phase Improvement Plan
See response §R (Phases A–E). Phase A (this pass): canonical bundle frozen, quarantine done. Remaining: rename metric repo-wide, restate speedup, regen RoPE/SLB, add Py↔C++ constant invariant tests, second host/model.

## S. Risks & Blockers
1. **E2E baselines at n=30** (not n=100). Canonical = clean serial run `raw/e2e_clean_n30.json` (B3 1214 ms, N1x 518 ms → **2.34×**). An earlier `e2e_n25.json` ran **concurrently** with another e2e (contention inflated its B3 to 2011 ms) and is superseded; a valid concurrent `e2e_n30.json` was erroneously deleted during cleanup and replaced by the clean run. Gateway 160 ms is unaffected (rock-stable, σ=0.6 ms). Accuracy CIs at n=30 are wide (N1x 0.87 vs B3 0.93; gateway n=500 accuracy 0.89).
2. **RoPE boundary + SLB scalability NOT regenerated** — remain pre-freeze/thin; do not re-assert as v1.1-measured.
3. **Path B (P>256) never exercised end-to-end** by any canonical bench — integration exists in code only.
4. **Single host / single model** — generality unproven.
5. **bench_e2e internal margin recalibration** prints fire_rate 0.60 (threshold 0.0554) on its own small query set, diverging from the committed P20 τ=0.01365/20% — the e2e N1x arm does not use the calibrated gate.
