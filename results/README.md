# Nexus v1.0 Benchmark Artifacts

This directory contains **only** the canonical evidence files for Nexus v1.0. All iterative Phase 22–29 experiment artifacts have been purged.

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
| [`bench_gateway_e2e_v2.json`](bench_gateway_e2e_v2.json) | Production gateway: **171 ms** TTFT P50, **0.91** E2E accuracy |
| [`bench_e2e_v2.json`](bench_e2e_v2.json) | Baselines: B1 **9.48 s** strawman, B3 **1.33 s** honest |
| [`bench_e2e.json`](bench_e2e.json) | M2 reference run: N1 **466 ms**, **2.8×** vs B3 |
| [`bench_phase28_radix_prefix_v2.json`](bench_phase28_radix_prefix_v2.json) | L0 exact-token radix: **~3 µs** copy P50, **66%** hit rate |
| [`recall_miss_analysis_v2.json`](recall_miss_analysis_v2.json) | M3 gate: dense **0.87**, CE **0.90**, **20%** CE invocation |
| [`margin_calibration_p20.json`](margin_calibration_p20.json) | P20 adversarial calibration audit trail |
| [`g4_gate_verdict.json`](g4_gate_verdict.json) | G4 physics boundary: splice **FAIL** at P=1024 |
| [`bench_phase22_maxsim.json`](bench_phase22_maxsim.json) | ColBERT graveyard: **709 µs** P50, Recall@1 **0.72** |
| [`bench_n1m_fidelity_blockmask.json`](bench_n1m_fidelity_blockmask.json) | Blockmask graveyard: tensor PASS, E2E **0.0** accuracy |
| [`bench_phase21_ttft_real.json`](bench_phase21_ttft_real.json) | TTFT decomposition; 153× strawman retired |
