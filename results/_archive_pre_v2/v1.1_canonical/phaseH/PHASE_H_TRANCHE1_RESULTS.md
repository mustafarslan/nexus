# Phase H — Tranche 1 (H4 + H5) Results

**Date:** 2026-06-20 · **Scope:** H4 (cheap accuracy-gap diagnosis) + H5 (RoPE/SLB regeneration) only. H1/H2/H3 not started.
**Execution:** serial, primary host. Governed by `results/phaseB_depth/PHASE_H_PLAN.md`.

## Provenance (all runs this tranche)
- git_sha: `3ea7441` · llama_cpp_sha: `cb2463bb` · host: Darwin 25.5.0 arm64 (Apple Silicon)
- generative model: Qwen2.5-14B-Instruct Q4_K_M (model_hash `a09ea5e7…`)
- embed model: **Ollama-distributed `nomic-embed-text:v1.5` GGUF**, blob digest `sha256:970aa74c…` (274 MB, f16-by-size). Not the canonical `nomic-embed-text-v1.5.f16.gguf` *filename*, but **τ reproduces bit-exact (see H4.1)**, confirming embedding parity.
- CE checkpoint: `results/tool_cross_encoder_finetuned_v3`.
- Metric discipline: route-only ≠ TTFT, kept separate throughout.

Evidence tags: **MEASURED** = run this tranche; **REGENERATED** = re-run of a prior pipeline; **STRUCTURAL** = derived from code; **LEGACY** = prior/pre-freeze, not re-run.

---

## H4 — Accuracy-gap diagnosis (cheapest experiments only)

### H4.1 Calibration re-derivation — REGENERATED
`scripts/calibrate_margin_threshold.py` → τ = **0.013646852970123292** — **bit-exact** to canonical. Synthetic calibration set is already 133 queries (53 adversarial + 83 hard-pair). No wider *real-query* calibration set exists without violating the zero-leakage policy. Artifact: `nexus_calibration_H4.py`, `margin_calibration_H4.json`.

### H4.2 Recall / routing-level error analysis — REGENERATED + MEASURED
`scripts/analyze_recall_misses.py` (n=100 hold-out): dense R@1 **0.87**, CE-gated R@1 **0.88** (~20% fire), target 0.90 **not met** — reproduces canonical. 13 dense misses. Artifact: `recall_miss_analysis_H4.json`.

Miss decomposition (the 13 dense misses vs τ):
- CE fired (margin<τ), 8 misses: 5 corrected to gold, 3 still wrong (1 is a retrieval-ceiling miss — gold not in top-5).
- CE not fired (margin≥τ), 5 misses: all 5 stay wrong (high-margin confident-but-wrong; gate never engaged).
- Net dense 0.87 → gated 0.88 (+1): CE corrects ~5 fired cases but also flips ~4 dense-correct cases when it fires → near-wash.

### H4.3 Fire-rate sweep (latency/accuracy trade) — MEASURED
Threshold-only sweep, same CE (no retraining):

| gate | fire fraction | CE-gated R@1 |
|---|---|---|
| τ=0.01365 | 0.20 | **0.88** |
| thr=0.05 | 0.53 | 0.85 |
| thr=999 (full rerank) | 1.00 | 0.85 |

**Raising fire rate lowers accuracy** (0.88→0.85). `rerank_recall_at_1=0.97` is recall on the *fired subset only* (CE is accurate on genuinely-ambiguous queries) — it is NOT global full-rerank accuracy (global = 0.85). The P20 gate at 20% is **near-optimal**; broad CE hurts. Artifacts: `recall_sweep_thr_0.05.json`, `recall_sweep_thr_999.json`.

### H4.4 N1x vs B_RP e2e disagreement — MEASURED (from canonical artifact, no new run)
From `results/v1.1_canonical/raw/e2e_clean_n30.json` (420 per-query records): N1x acc **0.867** vs B3 **0.933**, n=30. Disagreement: **3** cases N1x-wrong/B3-right, 1 B3-wrong/N1x-right, 1 both-wrong → net **2-case** N1x deficit. All 3 gap-source cases are confusable GitHub `create_*` families (create_or_update_file↔create_branch↔create_repository↔fork_repository) — the **same failure mode** as the routing misses.

### H4 conclusion
The accuracy gap (CE 0.88<0.90; N1x 0.87 vs B3 0.93) clusters in a small set of confusable tool families. The cheap H4 levers tested in this tranche did not improve the global metric; broader CE firing was counterproductive, so further gains now appear to require heavier retrieval/CE changes with explicit latency trade-offs. The within-tuple trade for raising fire is negative (more CE latency AND lower accuracy), so no free accuracy gain exists in the levers tested. Per the H4 decision rule, **STOP and report; do not proceed to retrain.** Caveat: the e2e gap is **2 net cases at n=30** — within the wide n=30 CI; its magnitude is not robustly established (this is what H3, not approved, would settle).

---

## H5 — RoPE / SLB regeneration

### H5.1 SLB scalability — REGENERATED (real microbench)
`docs/paper/scripts/bench_scalability.py` times `slb.search()` (INT8 scan) across N. Artifact: `slb_scalability_H5.csv`.

| N tools | scan P50 (µs) | scan P99 (µs) | mem (MB) |
|---|---|---|---|
| 10 | 1.375 | 4.606 | 0.04 |
| 50 | 2.625 | 5.178 | 0.19 |
| 100 | **3.708** | 10.061 | 0.38 |
| 500 | 10.791 | 17.587 | 1.89 |
| 1000 | 21.125 | 33.619 | 3.78 |

Measured **1.375→21.1 µs (N=10→1000)** vs legacy **1.67→21.5 µs** — reproduces closely; the **<5 µs L1 budget at N≤100 holds (3.71 µs)**. (The CSV's `e2e_ttft` column is route-only-class ~170 ms, **NOT true TTFT** — excluded from headline.)

### H5.2 Splice fidelity (RoPE boundary, output-distribution level) — MEASURED
`test/bench_splice_fidelity.py` compares next-token distributions: full-prefill reference vs Nexus KV splice (50 cases). Artifacts: `splice_fidelity_anchored_H5.json`, `splice_fidelity_H5.json`.

| compile mode | Δpos | next-tok KL p50 (nats) | top-1 agreement | NLL-delta p50 |
|---|---|---|---|---|
| **anchored** (base_pos=n_past) | 0 | **0.0000** | **1.00** | 0.000 |
| isolated (base_pos≠n_past) | ≠0 | 0.577 (mean 0.717) | 0.48 | 3.48 |

**Anchored splice is distributionally exact** (KL 0.0, top-1 1.0) — an independent output-level corroboration of the Path A fidelity claim; it does not numerically reproduce the legacy tensor-KL result because the metric is different. **Isolated (off-anchor) splice drifts badly** (top-1 agreement 0.48) — independently corroborates the RoPE-drift rationale for the P≤256 gate.

### H5 evidence boundary
This tranche upgrades SLB to regenerated evidence and upgrades the RoPE story mechanistically at the output-distribution level, but it does not yet retire the legacy tensor-KL boundary numbers.

### H5 honesty flags
- The legacy **0.0076 @P256 / 5.72 @P1024** are **tensor-level KL** (g4 artifact) — a **different metric** from the output-distribution KL measured here. They are **neither reproduced nor refuted** by H5.2.
- `docs/paper/scripts/aggregate_physics_boundary.py` is **NOT a regeneration path** — it re-emits hardcoded pre-freeze constants (its own `SOURCE_NOTE`). It must not be cited as regenerated evidence.
- **Remaining gap:** a position-sweep tensor-KL regeneration (P=256/512/1024) via the real fidelity/g4 harness was not run this tranche — the specific legacy boundary numbers remain LEGACY/pre-freeze until that sweep is done.

## Canonical Phase-H artifacts (this tranche)
- `results/v1.1_canonical/phaseH/nexus_calibration_H4.py`, `margin_calibration_H4.json`
- `results/v1.1_canonical/phaseH/recall_miss_analysis_H4.json`, `recall_sweep_thr_0.05.json`, `recall_sweep_thr_999.json`
- `results/v1.1_canonical/phaseH/slb_scalability_H5.csv`
- `results/v1.1_canonical/phaseH/splice_fidelity_anchored_H5.json`, `splice_fidelity_H5.json`
- `results/v1.1_canonical/phaseH/PHASE_H_TRANCHE1_RESULTS.md` (this file)
