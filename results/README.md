# Nexus Benchmark Artifacts

> **⚠️ 2026-06-20 — Canonical evidence moved.** The files below were regenerated on the original
> Apple-Silicon host and the headline numbers **did not reproduce**. The single source of truth is now
> [`v1.1_canonical/`](v1.1_canonical/) (see `CANONICAL_RESULTS_MANIFEST.json`). Superseded files in this
> directory carry a `_deprecation` field; pristine copies are in `_quarantine_2026-06-20/`.
> Canonical gateway routing latency: **160.2 ms P50 / 0.89** (5×100 = 500 queries), **not** 171 ms or 237.4 ms.

This directory contains earlier Nexus v1.0 evidence files. All iterative Phase 22–29 experiment artifacts have been purged.

Regenerate locally (models and checkpoints are **not** committed to Git):

```sh
# P20 margin calibration audit trail
python3 scripts/calibrate_margin_threshold.py

# L0 radix microbench (requires built bench_phase28_radix_prefix)
./build/bench_phase28_radix_prefix --output results/bench_phase28_radix_prefix_v2.json

# Cross-encoder v3 (required before recall / gateway benches)
python3 scripts/train_cross_encoder.py --output results/tool_cross_encoder_finetuned_v3

# M3 recall analysis
python3 scripts/analyze_recall_misses.py --output results/recall_miss_analysis_v2.json

# E2E and gateway benches
python3 test/bench_e2e.py --output results/bench_e2e_v2.json
python3 test/bench_gateway_e2e.py --output results/bench_gateway_e2e_v2.json
```

## Canonical whitelist

| File | Proves |
|------|--------|
| [`bench_gateway_e2e_v2.json`](bench_gateway_e2e_v2.json) | ⚠️ SUPERSEDED. Gateway **routing+splice** P50 171 ms / 0.91 (n=1 run). Canonical: **160.2 ms / 0.89** (`v1.1_canonical/`). Note: this metric excludes first-token decode (`decode_us=0`) — it is routing latency, not TTFT. |
| [`bench_e2e_v2.json`](bench_e2e_v2.json) | ⚠️ SUPERSEDED. Baselines B1 9.48 s, B3 1.33 s (Run A). Regenerated in `v1.1_canonical/raw/e2e_n25.json`. |
| [`bench_e2e.json`](bench_e2e.json) | ⚠️ SUPERSEDED + internally inconsistent (B3 P50≠CI.point). |
| [`bench_phase28_radix_prefix_v2.json`](bench_phase28_radix_prefix_v2.json) | ⚠️ SUPERSEDED (n=1, 66%). Canonical 10-seed: **69.5% hit (58–80%), copy_p50 3.06 µs**. |
| [`recall_miss_analysis_v2.json`](recall_miss_analysis_v2.json) | ⚠️ M3 gate: dense **0.87** (reproduces), CE **0.88** (regenerated; was 0.90 — does **not** meet 0.90 target), **20%** CE invocation. |
| [`margin_calibration_p20.json`](margin_calibration_p20.json) | P20 adversarial calibration audit trail |
| [`g4_gate_verdict.json`](g4_gate_verdict.json) | G4 physics boundary: splice **FAIL** at P=1024 |
| [`bench_phase22_maxsim.json`](bench_phase22_maxsim.json) | ColBERT graveyard: **709 µs** P50, Recall@1 **0.72** |
| [`bench_n1m_fidelity_blockmask.json`](bench_n1m_fidelity_blockmask.json) | Blockmask graveyard: tensor PASS, E2E **0.0** accuracy |
| [`bench_phase21_ttft_real.json`](bench_phase21_ttft_real.json) | TTFT decomposition; 153× strawman retired |
