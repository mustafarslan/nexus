# Verification Report — paper claims vs. v2.0 artifacts

Re-derived by `scripts/verify_paper_numbers.py` (read-only over `results/v2.0_canonical/raw/`).
**Result: 60/60 claims MATCH, 0 discrepancies.** No number in `main.tex` needed correction; the
Phase-6 edits are reframings mandated by the referee, not numeric fixes.

## Headline claims

| Claim | Paper | Recomputed | Verdict |
|---|---|---|---|
| Routing acc (N=10/50/100/250) | 92/90/89/89% | 92/90/89/89 | MATCH |
| Oracle acc N=10, overflow N≥50 | 98%, 0 | 98, 0/0/0 | MATCH |
| SLB R@1 / R@3 @ N=250 | 74 / 95% | 74 / 95 | MATCH |
| SLB in-situ (incl. FFI) @ N=250 | 17.6 µs | 17.64 µs | MATCH |
| SLB pure C++ SIMD | 8.25 µs | 8.250 µs | MATCH |
| τ (P20 adversarial) / gate fire | 0.0136 / 20.3% | 0.01365 / 20.30% | MATCH |
| Gating mean Spearman ρ | 0.193 | 0.193 | MATCH |
| Gating max-drift spread | 175–207 | 175–207 | MATCH |
| L0 radix copy P50 / warm-hit | 3.04 µs / 69.5% | 3.041 / 69.5 | MATCH |
| Sidecar routing / e2e arg / JSON | 86.7 / 80 / 100% | 86.67 / 80 / 100 | MATCH |
| IR tokens p50/p99 | 19/32 | 19/32 | MATCH |
| Hybrid vs oracle TTFT | 443.8 / 737.3 ms (1.66×) | 443.77 / 737.28 (1.661×) | MATCH |
| Sidecar main-context saving | ≈80% | 80.09% | MATCH |
| Table II TTFT (all 8 cells: R%, prefill, splice, speedup) | see `tab:perf` | all match | MATCH |
| Eq.(2) reproduces Table II R(%) | — | reproduces all 8 | MATCH |

## Eq.(2) ↔ Table II R(%) (re-confirmed)
`R(npast,K)` with M=256, R_base=5% reproduces every R(%): K=4 → 5.0/36.7/100/100; K=16 →
5.0/11.3/24.0/49.3. Add a one-line caption note to `tab:perf` stating this is verified so the
recompute schedule is auditable without the code.

## Wilson 95% CIs (real k/n — for Phase 3)
| Quantity | k/n | point | Wilson 95% CI |
|---|---|---|---|
| Routing acc N=250 | 89/100 | 89.0% | [81.4, 93.7] |
| SLB R@1 N=250 | 74/100 | 74.0% | [64.6, 81.6] |
| SLB R@3 N=250 | 95/100 | 95.0% | [88.8, 97.8] |
| Sidecar routing | 26/30 | 86.7% | [70.3, 94.7] |
| Sidecar e2e arg | 24/30 | 80.0% | **[62.7, 90.5]** |
| Sidecar routed-only specified-arg | 26/26 | 100% | **[87.1, 100]** → "100% (95% CI ≥ 87.1%)" |

**Denominator correction:** the referee draft used 80/100 and 30/30; the true sidecar set is n=30
(routed=26). The CIs above use the real counts and are correspondingly wider — the honest figures.
