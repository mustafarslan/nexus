# Phase H — H6 Evidence Hygiene Review

**Date:** 2026-06-21 · **Scope:** H6 only — claim inventory + reconciliation map. **No claim-file edits, no benchmarks, no code.** This artifact is the audit; applying it is a separate (not-yet-approved) step.

## 1. Executive Status
Audited the repo/paper-facing claim surfaces (README, RELEASE_V1, architecture.md, internals.md, results/README, verify_paper_math.py, docs/paper/*). No files were changed. Result: the surviving claims are recoverable, but the repo currently carries **three classes of stale state** that must be reconciled to Phase-H artifacts: (a) the e2e headline is n=30 everywhere and is superseded by H3 n=100; (b) the legacy tensor-KL boundary figures (0.0076/5.72) are cited as live in ~10 locations but were invalidated by H5; (c) two contradictory accuracy-gap framings ("~6-point 0.87/0.93" and "~3-point 0.89/0.92") coexist, both superseded by H3's non-significant ~1-point result. H1's architecture-scope precondition is not yet recorded anywhere.

## 2. Governing Evidence Base (Phase-H artifacts that now back surviving claims)
- **e2e / TTFT / accuracy:** `results/v1.1_canonical/phaseH/h3_e2e_n100.json` (+ `PHASE_H_H3_E2E_INTERVAL_REVIEW.md`) — n=100, intervals.
- **Model scope:** `PHASE_H_H1_SECONDMODEL_REVIEW.md` — FA-compatible / non-soft-capped precondition; Gemma2 blocked.
- **Boundary/fidelity:** `splice_fidelity_anchored_H5.json`, `splice_fidelity_H5.json`, `tensor_boundary_canary_5pct.json`, `logit_kl_boundary_phaseA.json`, `slb_scalability_H5.csv` (+ `PHASE_H_TENSORKL_BOUNDARY_REVIEW.md`).
- **Routing/accuracy diagnosis + calibration:** `recall_miss_analysis_H4.json`, `recall_sweep_thr_*.json`, `nexus_calibration_H4.py`, `margin_calibration_H4.json` (+ `PHASE_H_TRANCHE1_RESULTS.md` / `_REVIEW.md`).
- Unchanged-and-still-valid: gateway route-only 160.2 ms / 0.89 (n=500), L0 3.06 µs / 69.5%, τ bit-exact, dense R@1 0.87, CE 0.88<0.90.

## 3. Claim Reconciliation

| Claim area | Previous state | Corrected state (Phase-H) | Disposition | Canonical artifact |
|---|---|---|---|---|
| True-TTFT headline | N1x 518 / B3 1214 ms, 2.34×, n=30 | N1x **457.5** / B3 **1131.4** ms median, **2.47× CI[2.41,2.72]**, n=100 | RESTATE | h3_e2e_n100.json |
| Accuracy trade-off | "~6-point (0.87 vs 0.93)" AND "~3-point (0.89 vs 0.92)" | Δ=**−0.010, CI[−0.08,+0.05], McNemar p=1.0** (not significant, n=100) | RESTATE + NARROW (drop "trade-off"/"6-point"; do NOT claim "equivalent" — say "no detectable difference, CI ±~5pt") | h3_e2e_n100.json |
| B1 strawman | 9754 ms / 0.97, n=30 | 8487.7 ms median, n=100 (still "retired strawman") | RESTATE | h3_e2e_n100.json |
| B3pc reference | 334 ms (n=30) | 289.4 ms median (n=69 hit) | RESTATE | h3_e2e_n100.json |
| Legacy tensor-KL @P256 | "KL = 0.0076, top-1 1.0" (cited g4) | No tensor-KL harness exists; figure not reproduced. Anchored **output-dist** KL≈0.0 / top-1 1.0 is the real regenerated fidelity | DROP 0.0076 as "tensor-KL"; RESTATE to output-dist anchored exactness | splice_fidelity_anchored_H5.json |
| Legacy KL @P1024 = 5.72 | "LegoLink KL 5.72" (cited g4) | LegoLink config not reproducible; isolated logit-KL ~0.9, tensor-L2 flat; 5.72 unbacked | LEGACY / DO NOT CITE | PHASE_H_TENSORKL_BOUNDARY_REVIEW.md |
| Boundary narrative | graceful position-monotone decay 0.0076→5.72 | Anchored exact; off-anchor **uniformly** degraded P≥256 (no smooth curve, no 5.72 spike) | RESTATE | logit_kl_boundary_phaseA.json, tensor_boundary_canary_5pct.json |
| Model generality | "single host and model" (paper) | Add precondition: **FA-compatible, non-soft-capped** architectures (Gemma2 blocked: soft-cap → FA off → v_trans) | NARROW (add precondition) | PHASE_H_H1_SECONDMODEL_REVIEW.md |
| SLB scalability | 1.67 → 21.50 µs (N=10→1000) | **1.375 → 21.1 µs**; 3.71 µs @ N=100 (<5µs budget holds) | RESTATE | slb_scalability_H5.csv |
| Route-only vs TTFT | already distinguished (160 ms ≠ TTFT) | unchanged | KEEP | (gateway, v1.1) |
| CE Recall@1 | 0.88 < 0.90 target | unchanged; H4 adds: gap is CE-ceiling-bound, raising fire rate hurts (0.88→0.85) | KEEP (optionally annotate with H4) | recall_miss_analysis_H4.json |
| "memory-bandwidth tradeoff" (07_phys:13) | stated as causal | unmeasured causal claim | NARROW → future-work wording | (none; flagged in REWRITE_PLAN) |
| "robust OOD generalization in multi-domain registries" (05_ml_routing:38) | strong CE-training claim | unsupported generality wording | NARROW (soften) | — |

## 4. Mixed-State Findings (internal contradictions present now)
1. **Accuracy gap:** paper abstract/conclusion/06_eval say "~3-point (0.89 vs 0.92)"; architecture.md:222 and main.tex:316/Table say "~6-point (0.87 vs 0.93)". Two framings, both superseded by H3 (n.s. ~1pt).
2. **0.0076/5.72 cited as live** in README:68/147, RELEASE:78, internals:208, architecture:62/84/184, main.tex:160/402-403/412, 07_phys:9/32, PAPER_TABLES:64-65 — contradicts H5 (invalidated).
3. **Scalability CSV vs prose:** `scalability_curve.csv` already holds the H5-regenerated 1.375 µs, but 06_eval:68 prose still says 1.67/21.50 — direct CSV-vs-text disagreement.
4. **n=30 everywhere:** all e2e tables/CSVs (PAPER_TABLES, data/v1_1/*.csv, main.tex Table I, architecture §7.2) are n=30; H3 n=100 exists only in phaseH — divergence risk if not reconciled.
5. **Build artifacts** (`docs/paper/main.aux`, `main.bbl`) carry stale captions ("~6 accuracy points", "2.34×", "n=30") — regenerate on recompile; not source, but will reappear until the .tex is corrected and rebuilt.
6. **Stale raw artifacts** (`docs/paper/data/_raw/gateway_full.json` 237 ms, `e2e_full.json`) remain referenced by some CSVs — already known-superseded but still present.

## 5. Required Corrections (minimum to make the base coherent)
1. Replace n=30 e2e figures (518/1214/9754/334/2.34×) with H3 n=100 (457.5/1131.4/8487.7/289.4/**2.47× [2.41,2.72]**) in: README:21,168; RELEASE:20,24; verify_paper_math:90; architecture:216-222; main.tex:28-29,305-307,312-313,321,473; PAPER_TABLES:17-19; 06_eval:25; 09_concl:6; 01_abstract:5; data/v1_1/{pareto_frontier,macro_ttft}.csv.
2. Replace both accuracy-gap framings with "Δ=−0.010, n.s. (McNemar p=1.0, n=100); CI ±~5pt" — drop "~6-point", "~3-point", "trade-off"; do not assert "equivalent".
3. Remove/retire 0.0076 and 5.72 as cited fidelity numbers; restate anchored fidelity as output-distribution KL≈0.0 / top-1 1.0 (regenerated) and the boundary as "off-anchor uniformly degraded," with the tensor-KL legacy figures marked DO NOT CITE. Affects all §3-row-5/6/7 locations.
4. Add the H1 architectural precondition (FA-compatible, non-soft-capped) wherever model scope is stated (main.tex:482, abstract, conclusion).
5. Update SLB prose 1.67→**1.375**, 21.50→**21.1** at 06_eval:68 (or mark legacy per the rewrite plan).
6. Recompile the paper after .tex edits so `.aux`/`.bbl`/`.pdf` stop carrying stale captions.

These are corrections to apply in the *next* step, not done here.

## 6. Stray Artifact Classification
- **`results/v1.1_canonical/phaseH/h3_e2e_n100.json`** = canonical Phase-H copy (governing).
- **`results/bench_e2e_h3_n100.json`** = raw harness default-output of the same run (identical content). Classification: **retain as raw harness output**; the canonical copy under `phaseH/` is authoritative. Recommended relationship: treat root path as `raw`, phaseH path as `canonical`. **Do not delete now** — defer removal/relocation to an explicit, authorized cleanup step. No deletion performed.

## 7. Recommended Next Single Step
**Documentation patch pass — narrowly scoped to applying the §5 corrections only.** H6 has identified concrete, internally-contradictory state (0.0076/5.72 cited-but-invalid; n=30-vs-n=100; dual accuracy framings; CSV-vs-prose scalability). Resolving these is asset-free, restores coherence, and directly addresses the project's recurring mixed-state failure. It must be limited to the §3/§5 reconciliations — no rewriting beyond them. Not H2: a second host adds a generality axis on top of a base that is currently self-contradictory, and (like H1) may hit a second-host asset blocker; coherence first, host generality after.

## 8. Stop Line
H6 (audit) is complete. No claim files, code, benchmarks, or paper sources were modified; no llama.cpp/splicer/retraining/router/Path B/Tool IR/H2 work was performed or is authorized. Awaiting approval for the single recommended next step (a narrowly-scoped documentation patch pass applying §5).
