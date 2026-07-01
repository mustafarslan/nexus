# Stale Artifact Quarantine Manifest — 2026-06-20

**Principle:** no evidence destroyed. Pristine copies preserved in `results/_quarantine_2026-06-20/*.orig`. Files whose paths are hardcoded by docs/scripts were relabeled **in place** with an additive `_deprecation` key (non-breaking — consumers that read existing fields are unaffected).

## Why these are superseded
A fresh 5× re-run of the **identical harness, model (`a09ea5e7…`), and llama.cpp build (`cb2463bb`)** on the original host (2026-06-20) produced a stable Gateway TTFT P50 of **160.2 ms** (std 0.6 ms across 5 runs). The two committed headline values — **171.3 ms** (`results/*`, git `1384a33b`) and **237.4 ms** (`docs/paper/data/`, git `3ea7441`) — both fail to reproduce; both are high. The 171↔237 gap the prior audit could not break is run-time machine-state drift, not harness methodology (both use the same `summarize_arm_stats`/`np.percentile` path; `bench_cdf_pareto.py` literally *invokes* `bench_gateway_e2e.py`).

## Relabel-in-place (hardcoded paths kept working)

| File | Run | Stale metric | Canonical replacement | Action |
|---|---|---|---|---|
| `results/bench_gateway_e2e_v2.json` | A | GW 171.3 ms / 0.91 | `v1.1_canonical/raw/gateway_run[1-5].json` → **160.2 ms / 0.89** | `_deprecation` stamped; `.orig` preserved |
| `results/bench_e2e_v2.json` | A | B3 1.33 s, B1 9.48 s | `v1.1_canonical/raw/e2e_n25.json` | `_deprecation` stamped; `.orig` preserved |
| `results/bench_e2e.json` | A | B3 P50/CI intra-file mismatch | `v1.1_canonical/raw/e2e_n25.json` | `_deprecation` stamped (also flags C5) |
| `results/bench_phase28_radix_prefix_v2.json` | A | hit 66% (n=1), copy 3.042 µs | `v1.1_canonical/raw/l0_seed_[0-9].json` → **69.5%, 3.06 µs** | `_deprecation` stamped; `.orig` preserved |

## Quarantined (pristine copies only; not deleted)
| File | Reason |
|---|---|
| `results/_quarantine_2026-06-20/bench_phase21_ttft_real.json.orig` | superseded TTFT decomposition; "153× strawman retired" |
| `results/_quarantine_2026-06-20/bench_phase22_maxsim.json.orig` | ColBERT graveyard, n=1 (kept in place too as negative-result evidence) |
| `results/_quarantine_2026-06-20/nexus_paper.aux`, `.out` | orphaned root LaTeX intermediates — **DELETE_CANDIDATE** (also gitignored class) |

## Left untouched (verified / negative-result / regenerable)
- `results/margin_calibration_p20.json` — bit-exact regenerable (τ=0.01365).
- `results/recall_miss_analysis_v2.json` — regenerable; canonical R@1 0.87 / CE 0.88 (was 0.90).
- `results/g4_gate_verdict.json`, `bench_n1m_fidelity_blockmask.json`, `bench_phase22_maxsim.json` — well-supported negative results.
- `results/phaseA_tool_match_work/*.atb` — input fixtures (gateway depends on them).
- `results/tool_cross_encoder_finetuned_v3/` — CE v3 checkpoint (required input).
- `docs/paper/data/_raw/*` — kept for cross-run comparison; **not** promoted to canonical (see CANONICAL_RESULTS_MANIFEST.json).

## Not run by this pass (still need regeneration on this host)
- `docs/paper/data/physics_boundary_rope.csv` — header says *"pre-freeze validation records"*; `aggregate_physics_boundary.py` is the intended generator but was **not executed** this pass. Remains **UNKNOWN_PROVENANCE**.
- `docs/paper/data/scalability_curve.csv` (SLB) — not regenerated this pass.
