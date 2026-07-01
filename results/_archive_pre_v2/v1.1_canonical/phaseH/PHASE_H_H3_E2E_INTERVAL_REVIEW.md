# Phase H — H3 E2E Interval Review

**Date:** 2026-06-21 · **Scope:** H3 only. No H1 retry, H2, llama.cpp upgrade, splicer/soft-cap changes, retraining, router, Path B, Tool IR, paper/README/architecture work.
**Outcome:** **PRESERVED (strengthened).** At n=100 with intervals, the true-TTFT advantage is clearly separated and the n=30 accuracy gap does not survive.

## 1. Executive Status
Ran the canonical e2e harness serially at query-limit 100 (single process, no concurrency). All required arms produced n=100 samples (B3pc naturally splits hit=69 / miss=31). Computed median, IQR, and bootstrap CIs per arm, plus paired N1x-vs-B3 deltas. Did not run any other workstream. H3 strengthens the only validated configuration: the 2.34× n=30 headline becomes **2.47× with a tight CI**, and the n=30 6-point accuracy gap collapses to a non-significant ~1 point.

## 2. Provenance
- Canonical tuple, like-for-like: git `3ea7441`, llama_cpp `cb2463bb`, Qwen2.5-14B Q4_K_M (`a09ea5e7…`), Darwin arm64, embed = Ollama nomic v1.5 (`970aa74c…`, τ-parity confirmed in tranche 1).
- Artifact timestamp 2026-06-21T09:38:25Z. Serial single-process run (PID 56384), no concurrent benches — contamination lesson from the prior n=25 concurrent run respected.
- Dataset ceiling: the query pool is exactly 100 (all match the 10-tool corpus); n=100 is the max achievable without new data. No definitions altered.
- Artifacts: `results/bench_e2e_h3_n100.json` (canonical copy: `results/v1.1_canonical/phaseH/h3_e2e_n100.json`).

## 3. Experimental Design
- Arms (canonical, unchanged): N1x, N1, B3, B3pc(_hit/_miss), B1 reported; full harm set (14 arms) ran but only the required arms are analyzed here.
- Sample size: n=100 per arm (1400 records total).
- Metric: **true TTFT** = `ttft_us` (prefill + first token). Route-only is **not** measured by this harness and is **not** reported here (route-only 160.2 ms remains a separate gateway metric).
- Statistics: median + IQR (p25–p75); bootstrap 95% CI (10,000 resamples, seed 1337) for per-arm median and for the paired N1x/B3 median-ratio and accuracy delta. Accuracy compared paired by `case_id`; McNemar from harness `paired_stats`.

## 4. Results

**True TTFT (n=100):**

| arm | n | median (ms) | IQR (ms) | CI95 median (ms) | accuracy |
|---|---|---|---|---|---|
| N1x | 100 | 457.5 | [427, 477] | [445, 469] | 0.910 |
| N1 | 100 | 431.1 | [423, 473] | [428, 438] | 0.920 |
| B3 (B_RP) | 100 | 1131.4 | [1008, 1307] | [1121, 1260] | 0.920 |
| B3pc_hit | 69 | 289.4 | [283, 310] | [287, 299] | 0.986 |
| B1 (bloat) | 100 | 8487.7 | [8173, 9435] | [8322, 8998] | 0.980 |

**Pairwise N1x vs B3 (true TTFT, paired):**

| metric | value | CI95 |
|---|---|---|
| speedup (median B3 / median N1x) | **2.47×** | **[2.41, 2.72]** |
| accuracy delta (N1x − B3) | **−0.010** | **[−0.080, +0.050]** |
| McNemar (B3 vs N1, harness) | p = 1.0 | not significant |

Route-only: not measured in this harness (see §3) — not reported, not headlined.

## 5. Interpretation
- **Speedup is real once uncertainty is shown.** N1x median 457.5 ms (CI [445,469]) and B3 median 1131.4 ms (CI [1121,1260]) have **non-overlapping CIs by a wide margin**; the bootstrap speedup CI [2.41, 2.72] sits far above 1. The advantage does not degrade to marginal under intervals — it holds at ~2.5×.
- **The accuracy trade is now negligible.** At n=100 the N1x−B3 delta is −0.010 with CI straddling zero, and McNemar is non-significant. The n=30 "6-point gap" (0.87 vs 0.93) was small-n noise and does not reproduce. This removes the prior trade-off caveat.
- **Net positive in the canonical tuple:** ~2.5× true-TTFT reduction at statistically indistinguishable accuracy. (B3pc_hit at 289 ms / 0.986 remains the prefix-cache reference; B1 confirms the bloat strawman scale.)
- Skeptical caveats: still **one model, one host**; n=100 is the dataset ceiling (not extensible without new queries); accuracy CIs at 100 samples are still ±~5 pts, so "negligible gap" means "no detectable gap at this n," not "proven equal."

## 6. Claim Impact
**PRESERVED (strengthened).** Within the validated tuple (Qwen2.5, this host, pinned stack), the true-TTFT advantage is now interval-backed at 2.47× [2.41, 2.72] and the accuracy concern is reduced to a non-significant ~1 point. The headline is not dropped or narrowed by intervals — it is firmer. This does not extend generality (H1 already scoped the mechanism to flash-attention-compatible, non-soft-capped architectures; host generality is untested), but the magnitude of the one working configuration is now defensible.

## 7. Recommended Next Single Step
**H6 — repo/paper evidence hygiene.** Four corrections have now accumulated without being reconciled into the repo's standing claims: H3 (n=100 → 2.47× and negligible accuracy gap, superseding the n=30 518/1214/2.34×/6-pt figures), H1 (architecture-scope precondition: FA-compatible, non-soft-capped), H5 (legacy tensor-KL 0.0076/5.72 invalidated), H4 (accuracy gap is CE-ceiling-bound). The repo's recurring failure mode is mixed old/new state (the original audit). Consolidating these into one coherent, correctly-cited evidence base is asset-free and higher leverage now than H2, which would add a host axis on top of an un-reconciled base (and may hit a second-host asset blocker like H1 did). H2 should follow once the evidence base is locked.

## 8. Stop Line
H3 is complete. No H1 retry, H2, llama.cpp upgrade, splicer/soft-cap change, retraining, router, Path B, Tool IR, or paper/README/architecture work was performed or is authorized. Awaiting approval for the single recommended next step (H6).
