# Phase H — Tranche 1 (H4/H5) Decision-Grade Review

**Date:** 2026-06-20 · **Source:** `PHASE_H_TRANCHE1_RESULTS.md` (wording-corrected) · **Status:** consolidation only, no new runs.
**Governing plan:** `results/phaseB_depth/PHASE_H_PLAN.md`.

## 1. Executive Status
The H4 (cheap accuracy-gap diagnosis) and H5 (RoPE/SLB regeneration) tranche was executed serially on the primary host. H1, H2, H3, Path B, Tool IR, and any paper/README/architecture work were not executed. This artifact exists to classify the tranche's evidence by class and to gate exactly one next execution step.

## 2. What This Tranche Upgraded
- **SLB scalability — REGENERATED (measured this tranche).** `bench_scalability.py` produced scan P50 1.375→21.1 µs (N=10→1000), 3.71 µs at N=100; the <5 µs L1 budget at N≤100 holds. Reproduces legacy 1.67→21.5 µs. This moves SLB from thin/pre-freeze to regenerated.
- **Anchored splice fidelity — MEASURED, independently corroborated.** Output-distribution next-token KL = 0.0, top-1 agreement = 1.00 in the anchored (Δpos=0) regime. This is an independent output-level corroboration of the Path A fidelity claim. It does not numerically reproduce the legacy tensor-KL because the metric differs.
- **RoPE-drift rationale — MEASURED corroboration.** Off-anchor (Δpos≠0) splice drifts badly (top-1 agreement 0.48), independently supporting the P≤256 gate rationale.
- **Calibration τ — REGENERATED.** Bit-exact (0.013646852970123292), which also confirms embedding parity of the Ollama nomic GGUF used.

## 3. What Remains Legacy or Unresolved
- **Legacy tensor-KL boundary values (0.0076 @P256, 5.72 @P1024)** remain LEGACY/pre-freeze. They are a different metric from the output-distribution KL measured here and were not regenerated. `aggregate_physics_boundary.py` is not a valid regen path (it transcribes hardcoded pre-freeze constants).
- **n=30 e2e accuracy-gap uncertainty.** The N1x 0.867 vs B3 0.933 gap is 2 net cases at n=30, inside the wide n=30 CI. Magnitude not robustly established.
- **No H1/H2/H3.** Second-model, second-host, and n≥100-with-intervals generality are all still unproven.
- **Residual pre-freeze dependence.** Any boundary claim still rests on pre-freeze tensor-KL artifacts until a real position-sweep regeneration is run.

## 4. Cleaned H4 Takeaway
Calibration reproduced bit-exact. The cheap gate/fire-rate changes tested did not improve the global metric (dense R@1 0.87, CE-gated 0.88; raising fire to 53%/100% lowered it to 0.85). Misses cluster in confusable GitHub `create_*` families at both the routing and e2e levels. Stronger gains now likely require heavier retrieval/CE work with explicit latency trade-offs; that is out of the cheap-diagnosis scope and was not started.

## 5. Cleaned H5 Takeaway
SLB scalability is genuinely regenerated and reproduces the legacy curve, with the <5 µs N≤100 budget intact. Anchored splice is exact at the output-distribution level in the tested regime (KL 0.0, top-1 1.0); off-anchor splice drifts badly (top-1 0.48). This does not numerically reproduce the legacy tensor-KL boundary values because the metric differs. `aggregate_physics_boundary.py` is not a valid regeneration path.

## 6. Recommended Next Single Step
**Choose B: run a tensor-KL position sweep (P=256/512/1024) to replace the remaining legacy RoPE boundary numbers.**

Defense: the largest claim-hardening leverage right now is retiring pre-freeze evidence the paper/system still leans on. The boundary numbers (0.0076 / 5.72) are currently LEGACY with no valid regen path, and this tranche showed the named regen script is a transcriber — so those numbers are effectively unbacked. A position sweep through the real splice-fidelity harness directly converts an unbacked legacy claim into measured evidence, in the same metric family already running, at low marginal cost.

Why not A (H3 n≥100) first: H3 is valuable but lower leverage now. The accuracy gap it would tighten is a 2-case, n=30 effect whose direction is already understood (confusable families, CE-ceiling-bound); H3 would size it but not change the strategic picture, and it costs more compute. The boundary numbers, by contrast, are a standing provenance hole that B closes outright. B before A.

## 7. Stop Line
No new runs were performed in this consolidation step. Nothing beyond the single recommended next step (B: tensor-KL position sweep) is authorized. Awaiting approval.
