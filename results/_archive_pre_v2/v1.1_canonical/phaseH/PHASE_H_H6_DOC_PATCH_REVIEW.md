# Phase H — H6 Documentation Patch Review

**Date:** 2026-06-21 · **Scope:** narrow documentation patch applying the H6 §5 corrections only. No benchmarks, no code/perf changes, no scope broadening.

## 1. Executive Status
Applied the H6 corrections across the authoritative claim surfaces. The repo/paper-facing evidence now maps to current Phase-H artifacts: TTFT headline is n=100 interval-backed (2.47× CI[2.41,2.72]), the accuracy framing is "no detectable gap at n=100" (not "trade-off", not "equivalent"), the legacy tensor-KL figures (0.0076/5.72) are retired, model scope carries the H1 architectural precondition, and SLB scalability uses the regenerated H5 numbers. The paper (`main.tex`) **compiles cleanly** after the table edits. Residual stale-number matches in authoritative files are all corrective context (explicit retirements), not live claims. Superseded mirrors (`sections/`, `PAPER_TABLES.tex`) were left unbuilt but marked.

## 2. Files Updated
- **README.md** — metric-definition note (n=100 TTFT, 2.47× CI, accuracy n.s., scope precondition); Dual-Path table (anchored-exact fidelity; Path B = text-prefill, L0 not consulted); graveyard row (5.72 retired); citation line.
- **RELEASE_V1.md** — headline TTFT row (n=100); metric note (2.47× CI, accuracy n.s.); graveyard row (5.72 retired).
- **docs/architecture.md** — §3.1 viability (anchored output-dist exactness, 0.0076 retired); §3.2 boundary (off-anchor uniform degradation, 5.72 retired); §6 graveyard row; §7.1 caveat (2.47×); §7.2 e2e table + speedup/accuracy prose (n=100, CIs).
- **docs/internals.md** — §4.3 suffix-recompute fidelity (anchored exact; 0.0076 retired).
- **verify_paper_math.py** — `load_e2e()` now prefers `phaseH/h3_e2e_n100.json`; verdict strings updated (2.47× CI, accuracy n.s., scope, retired figures). Re-run confirmed: derives B3 1131 / N1x 458 / 2.5×.
- **docs/paper/main.tex** — abstract (TTFT/accuracy/scope incl. Gemma2 precondition); architecture §(0.0076 retired); Table I (n=100 medians+CIs) + speedup/accuracy prose; CDF & Pareto captions (marked n=30/legacy, point to n=100 table); routing microbench (458 ms); SLB scalability (regenerated 1.375→21.1 µs); physics-boundary table replaced with regenerated anchored-vs-isolated output-dist KL (0.0076/5.72 withdrawn); conclusion (n=100 figures, Gemma2 scope); intro/background TTFT figures (8.49 s / 1.13 s).
- **docs/paper/tables/PAPER_TABLES.tex** — added SUPERSEDED header (n=30 + retired KL; points to phaseH; not \input by main.tex).

## 3. Applied Corrections

| Claim area | Old state | New state | Governing artifact |
|---|---|---|---|
| True-TTFT headline | 518 / 1214 ms, 2.34×, n=30 | 457.5 / 1131.4 ms median, **2.47× CI[2.41,2.72]**, n=100 | phaseH/h3_e2e_n100.json |
| Accuracy framing | "~6-point" / "~3-point" trade-off | Δ=−0.010, CI[−0.08,+0.05], McNemar p=1.0 — no detectable gap (not "equivalent") | phaseH/h3_e2e_n100.json |
| B1 / B3pc | 9754 / 334 ms (n=30) | 8487.7 / 289.4 ms (n=100) | phaseH/h3_e2e_n100.json |
| Tensor-KL @P256/@P1024 | 0.0076 / 5.72 cited live | retired (no tensor-KL harness); anchored output-dist KL≈0/top-1 1.0 | splice_fidelity_anchored_H5.json; PHASE_H_TENSORKL_BOUNDARY_REVIEW.md |
| Boundary narrative | graceful monotone curve | off-anchor uniformly degraded P≥256; tensor-L2 position-independent | logit_kl_boundary_phaseA.json; tensor_boundary_canary_5pct.json |
| Model scope | "one host, one model" | + FA-compatible, non-soft-capped precondition (Gemma2 blocked) | PHASE_H_H1_SECONDMODEL_REVIEW.md |
| SLB scalability | 1.67→21.5 µs (pre-freeze) | 1.375→21.1 µs; 3.71 µs @N=100 | slb_scalability_H5.csv |
| Route-only vs TTFT | mostly distinguished | preserved; n=100 TTFT kept separate from 160 ms route-only | (gateway v1.1 / phaseH) |

## 4. Remaining Non-Goals (intentionally untouched)
- **`docs/paper/sections/*`** — superseded mirror (own `SUPERSEDED.md`); not `\input` by main.tex; not built. Left as-is.
- **Figure PDFs** (`fig_e2e_cdf_v1_1.pdf`, `fig_pareto_frontier_v1_1.pdf`) — show n=30 data; captions now mark them legacy and point to the n=100 table. Regenerating the plots from the n=100 artifact is a data-pipeline task, not a doc patch — deferred.
- **`docs/paper/data/*` CSVs** (incl. `data/v1_1/*` n=30, `_raw/*`) — derived/raw; `main.tex` uses inline tables, so these do not feed the authoritative build. Generator rewire is Phase B, out of scope.
- **No code/splicer/llama.cpp/benchmark changes.** No claims invented; no stale headline numbers preserved as current.

## 5. Artifact Policy
- `results/v1.1_canonical/phaseH/h3_e2e_n100.json` = **canonical** (governing; cited by all updated surfaces and by `verify_paper_math.py`).
- `results/bench_e2e_h3_n100.json` = identical raw harness default-output, **retained as raw provenance**; not deleted. The phaseH copy is authoritative. Removal/relocation deferred to an explicit authorized cleanup step.

## 6. Recommended Next Single Step
**Stop and consolidate.** The evidence base was just stabilized and now self-consistent; no new experiment is warranted right now. The only open loose ends are derivative (regenerating the two figures and the n=30 data CSVs from the n=100 artifact), not a new measurement. H2 (second host) would add a generality axis, but it requires a second-host asset (possible blocker as in H1) and host-generality is not the current gap — model-generality was already bounded by H1, and magnitude was just solidified by H3. Defer H2 until you decide it's worth the asset; consolidate the stabilized base first.

## 7. Stop Line
The documentation patch is complete and the paper compiles. No H2, no new benchmarks, no H1 retry, no llama.cpp/splicer/retraining/router/Path B/Tool IR/architecture work was performed or is authorized. Awaiting your direction.
