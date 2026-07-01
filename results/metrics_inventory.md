# v2.0 Canonical Metrics Inventory

All paper numbers trace to `results/v2.0_canonical/raw/`. Generated for the referee revision
(Phase 0). "Raw arrays?" = whether per-trial data is present for bootstrap, vs. only summaries.

| File | Backs (paper) | Key fields | Raw arrays? |
|---|---|---|---|
| `deep_splice_ttft.json` | Table II (`tab:perf`), Fig `fig:ttft`, abstract 1.1–1.7× | `curves.full_mult_{4,16}[]` each: `p_start, eff_pct, prefill_ms_median, prefill_ms_ci95, splice_ms_median, splice_ms_ci95, speedup, top1_agree, kl_nats` | No — medians + stored `*_ci95` only (n=15/cell in `config.trials`). Re-run needed for raw ratio-CI. |
| `routing_accuracy_n250.json` | §VII-B, Table III, Fig `fig:route` | `routing_summary.{10,50,100,250}`: `mean_acc_b` (routing), `mean_acc_a` (oracle), `mean_rec1/3/5`, `avg_lat_slb_ms`, `avg_tokens_a/b`, `token_savings_pct` | No — per-N summaries (n_queries=100, runs=1, std=0). |
| `slb_latency.json` | Table III pure-SIMD 8.25 µs | `median_us`, `ci95_us`, `n_tools=250`, `trials=200` | No — median + ci95 only. |
| `calibration.json` | §V-A τ=0.0136, gate fire 20.3% | `calibration.CALIBRATED_MARGIN_THRESHOLD`, `p20.fire_rate`, `per_query[]` | Partial — `per_query[]` margins present (n=133 synth / 53 adversarial). |
| `gating_nogo.json` | §IV-B, Table I (`tab:gating`), ρ=0.193 | `mean_spearman`, `per_condition[6]` (`max_drift, mean_drift, spearman`), `findings[]` | Per-condition rows (6), not per-head raw. |
| `rope_boundary_gate.json` | §IV-A retired LegoLink 5.72 nats | `p_start=1024`, `note` (5.72 @ P=1024), `viable=false` | No — single retired-config summary. |
| `sidecar/accuracy.json` | §VII-C, Table III, hybrid 443.8/737.3 ms | key `"10"`: `routing_accuracy, arg_accuracy_routed_only, arg_accuracy_e2e, json_valid_rate_hybrid, ir_tokens_p50/p99, ttft_hybrid_p50_ms, ttft_oracle_p50_ms, token_savings_pct` | No — n=30 consistency set summary (routed=26). |
| `l0_radix.json` | Table III radix 3.04 µs / 69.5% | `copy_p50_us_median`, `hit_rate_mean`, `per_seed_hit[10]`, `per_seed_copy_p50_us[10]` | **Yes** — 10-seed raw arrays. |

## Real denominators (for honest Wilson CIs)
- Routing acc N=250: **89/100**. SLB R@1: **74/100**. R@3: **95/100** (`n_queries=100`).
- Sidecar (n=30 consistency set, routed=26): routing **26/30**, e2e arg **24/30**, routed-only specified-arg **26/26**.
  Note: the referee draft inferred 80/100 and 30/30 — those denominators are wrong; use 30/26.

## Not present / requires generation
- **No D_KL-vs-offset sweep** in v2.0 (only Δpos=0 and the retired 1024 point). Phase 4 generates
  `dkl_sweep.json` on the on-disk Qwen model.
- **No raw per-trial TTFT arrays** — Phase 3 re-runs `bench_v2_capstone.py` (additive patch) to emit
  them for the speedup-ratio bootstrap CI.
