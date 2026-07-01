# Phase H — Tensor-KL Boundary Sweep Review

**Date:** 2026-06-20 · **Scope:** approved step B only (RoPE/splice boundary regeneration at P=256/512/1024). No H1/H2/H3/Path B/Tool IR/paper/README work.
**Provenance:** git `3ea7441`, llama_cpp `cb2463bb`, Qwen2.5-14B Q4_K_M (`a09ea5e7…`), Darwin arm64, n_gpu_layers=999, serial.

## 1. Executive Status
Ran two boundary sweeps at P=256/512/1024: (a) per-layer **KV tensor L2 divergence** (`bench_p1024_canary_bisect`, 5% suffix recompute) and (b) **logit/output-distribution KL** (`bench_phaseA_fidelity`, isolated RoPE-shifted splice). Did **not** run, and was not authorized to run, any H1/H2/H3, retraining, or paper work. Outcome is the **acceptable-failure** case: the legacy tensor-KL boundary numbers cannot be faithfully regenerated, and this is proven rather than papered over.

## 2. Harness Validity
- **No tensor-level KL harness exists in the repo.** The only tensor-level metric is **L2** (`token_l2` in `bench_p1024_canary_bisect`), not KL. The only KL is **logit/output-distribution** KL (`kl_ref_to_splice` in `bench_phaseA_fidelity` and `bench_splice_fidelity`).
- **Legacy values are prose-sourced.** `0.0076 @P256` / `5.72 @P1024` are cited from `results/g4_gate_verdict.json`, which stores only a note string ("LegoLink partial recompute KL up to 5.72 at P=1024") — no reproducible per-position KL computation.
- **`aggregate_physics_boundary.py` was not used** (it re-emits hardcoded pre-freeze constants; established last tranche).
- Minimal deviations: ran canary at `--suffix-pct 5` to match the legacy "5% recompute" wording; ran phaseA at `--positions 256,512,1024`, tool-limit 3, query-limit 3. No code was modified. Small n (3 tools; 9 logit-KL cases) — diagnostic, not headline.

## 3. Results

**Tensor-level KV divergence — L2 (NOT KL), 5% suffix recompute:**

| P | tensor L2 mean | tensor L2 max | n |
|---|---|---|---|
| 256 | 12.36 | 56.9 | 3 |
| 512 | 11.78 | 57.6 | 3 |
| 1024 | 12.14 | 58.0 | 3 |

→ Position-independent (~12 mean across 256–1024). Tensor divergence does **not** grow with depth.

**Logit / output-distribution KL — isolated (RoPE-shifted, delta_pos=P), no recompute:**

| P | delta_pos | KL mean | KL max | top-1 agreement |
|---|---|---|---|---|
| 256 | 256 | 1.083 | 1.572 | 0.444 |
| 512 | 512 | 0.881 | 1.692 | 0.667 |
| 1024 | 1024 | 0.896 | 1.632 | 0.333 |

→ ~Uniformly poor for P≥256; **no P=1024 blowup**, no 0.0076 floor.

(Companion: tranche-1 anchored/delta_pos=0 logit-KL = 0.0, top-1 = 1.00.)

## 4. Comparability to Legacy
- **Directly comparable to 0.0076 @P256:** none of these. The 0.0076-with-top1=1.0 description matches the **anchored** (delta_pos=0) regime, where the regenerated value is logit-KL ≈ 0.0 (tranche-1) — i.e. the legacy "P256" number was the anchored path, not an isolated boundary point.
- **Directly comparable to 5.72 @P1024:** none. 5.72 was a **LegoLink scattered-partial-recompute** configuration; no current harness implements it. Isolated logit-KL at P1024 is ~0.90; tensor-L2 is ~12 (flat). Neither reproduces 5.72.
- The legacy numbers and these measurements are **not the same metric**: legacy is labeled "tensor-KL" but no tensor-KL exists; the regenerable metrics are tensor-L2 and logit-KL.

## 5. Evidence Upgrade
- **LEGACY → REGENERATED (corroboration, different metric):** the dual-path rationale. Anchored splice exact (logit-KL 0.0, top-1 1.0); isolated splice poor for P≥256 (logit-KL ~0.9–1.1, top-1 0.33–0.67); tensor-L2 confirms perturbation is position-independent.
- **MEASURED, different metric family:** tensor-L2 boundary (flat ~12) and isolated logit-KL boundary (~0.9–1.1). Both new, correctly labeled.
- **LEGACY still unresolved / INVALID to cite:** `0.0076 @P256` and `5.72 @P1024` as "tensor-KL." Not reproduced in any metric; provenance is a prose note. Do **not** cite as regenerated. The 5.72 belongs to a rejected LegoLink config with no runnable path.

**Are the legacy boundary numbers now retired? Yes — as citable evidence, but by invalidation, not replacement.** They are shown to be mislabeled (no tensor-KL exists), unreproducible (5.72), or actually the anchored ≈0 case (0.0076). They are replaced not by a faked tensor-KL but by two correctly-labeled regenerated boundaries (tensor-L2; isolated logit-KL) plus the anchored-exactness result. No equivalence is claimed.

## 6. Interpretation
The Path A boundary story holds **mechanistically** but the **legacy quantitative curve does not**. What is actually true under measurement: anchored (delta_pos=0) splice is exact; off-anchor splice is **uniformly** degraded for all P≥256 (top-1 ~0.33–0.67), **not** a smooth depth-graded decay culminating in a 5.72 spike. Tensor magnitude (L2) is position-independent, so the deep-position failure is an output-distribution/RoPE-phase amplification effect, not a tensor-blowup. Practical consequence: the gate's justification ("splice only where delta_pos=0, i.e. anchored P≤256") is supported, but any narrative of a graceful, position-monotone fidelity boundary — and the specific 0.0076/5.72 figures — must be dropped.

## 7. Recommended Next Single Step
**H1 — second-model validation.** The boundary story is now mechanistically anchored-vs-isolated, not a fragile number; the largest remaining claim risk is single-model generality (the `.atb` ABI encodes model-specific RoPE/GQA, and anchored exactness was shown on one model). H1 tests whether anchored splice stays exact and isolated stays degraded on a second RoPE/GQA config — directly hardening the now-corrected boundary claim. Not H3 (it sizes an already-understood n=30 accuracy effect — lower leverage). Not H2 (host generality matters but is downstream of confirming the mechanism holds on another model).

## 8. Stop Line
No broader work is authorized. H1/H2/H3, retraining, Path B, Tool IR, paper/README/architecture edits were not performed and remain unapproved. This step is complete; awaiting approval for the single recommended next step.
