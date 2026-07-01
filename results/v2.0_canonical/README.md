# Nexus v2.0 Canonical Evidence Bundle

This directory contains the frozen canonical evidence logs backing the measurements, figures, and tables in the Nexus paper.

## Directory Structure

```
results/v2.0_canonical/
├── EVIDENCE_LEDGER.csv       # Unified registry mapping claims to code and artifacts
├── REFERENCE_AUDIT.csv        # Detailed audit records mapping paper mentions to findings
├── comparison_sources.md     # Citations and contextual info for baseline comparisons
├── raw/                      # JSON log outputs from benchmark runs
│   ├── calibration.json      # Margin-gate calibration results (τ = 0.0136)
│   ├── causal_isolation.json # Results of isolated attention-mask controls
│   ├── deep_splice_ttft.json # TTFT latency under deep splice curves (Table II, Fig 3)
│   ├── dkl_sweep.json        # Next-token D_KL vs splice offset sweep (Fig 1)
│   ├── gating_nogo.json      # Reference-free drift gate Spearman evaluations (Table I)
│   ├── l0_radix.json         # L0 radix cache copy latency & hit rates (Table III)
│   ├── rope_boundary_gate.json # Legacy scattered partial-recompute (LegoLink) boundary data
│   ├── routing_accuracy_n250.json # End-to-end routing accuracy vs tool scale (Table III, Fig 4)
│   ├── slb_latency.json      # Pure C++ SIMD vector dot product scan timings
│   └── sidecar/
│       └── accuracy.json     # Hybrid sidecar routing & argument accuracies (Table III)
```

## Mapping to Paper Results

### Table I (Reference-free Drift Gate)
- **Artifact**: [raw/gating_nogo.json](file:///Volumes/AI_SSD/Projects/nexus/results/v2.0_canonical/raw/gating_nogo.json)
- **Reproduction**: `python test/profile_head_drift.py`

### Table II (Deep Splice TTFT vs Depth) & Figure 3
- **Artifact**: [raw/deep_splice_ttft.json](file:///Volumes/AI_SSD/Projects/nexus/results/v2.0_canonical/raw/deep_splice_ttft.json)
- **Reproduction**: `python test/bench_v2_capstone.py --trials 15`

### Table III (Micro-costs & Routing/Argument Accuracies)
- **Artifacts**:
  - Routing accuracy and SLB top-1/top-3 recall: [raw/routing_accuracy_n250.json](file:///Volumes/AI_SSD/Projects/nexus/results/v2.0_canonical/raw/routing_accuracy_n250.json)
  - SLB C++ latency: [raw/slb_latency.json](file:///Volumes/AI_SSD/Projects/nexus/results/v2.0_canonical/raw/slb_latency.json)
  - Radix cache: [raw/l0_radix.json](file:///Volumes/AI_SSD/Projects/nexus/results/v2.0_canonical/raw/l0_radix.json)
  - Argument accuracies & hybrid sidecar accuracy: [raw/sidecar/accuracy.json](file:///Volumes/AI_SSD/Projects/nexus/results/v2.0_canonical/raw/sidecar/accuracy.json)
- **Reproduction**:
  - Routing scale: `python test/bench_routing_accuracy.py --tool-sizes "10 50 100 250" --runs 1`
  - Hybrid sidecar: `python test/bench_sidecar_accuracy.py`

### Figure 1 (Next-token Divergence vs Splice Offset)
- **Artifact**: [raw/dkl_sweep.json](file:///Volumes/AI_SSD/Projects/nexus/results/v2.0_canonical/raw/dkl_sweep.json)
- **Reproduction**: `python test/bench_dkl_sweep.py`
