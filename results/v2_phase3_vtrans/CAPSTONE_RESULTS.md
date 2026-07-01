# Nexus v2.0 Capstone Results (2026-06-30)

Model: Qwen2.5-14B-Instruct Q4_K_M, Apple M4 Max (UMA/Metal). schema_len=270, query_len=67.
Bench: `test/bench_v2_capstone.py`. Deep splice = inject + depth-adaptive recompute;
baseline = full text re-prefill (the decline path). top1/KL vs no-splice reference.

## TTFT vs depth — never-regress curve (full_mult=4, the orchestrator default)
| p_start | eff_pct | prefill_ms | splice_ms | speedup | top1 | KL |
|--------:|--------:|-----------:|----------:|--------:|:----:|----:|
| 256  | 5.0   | 1627 | 1041 | 1.56x | OK | 0.00018 |
| 512  | 36.7  | 2562 | 2166 | 1.18x | OK | 0.00018 |
| 1024 | 100.0 | 4323 | 4578 | 0.94x | OK | 0.00000 |
| 2048 | 100.0 | 7875 | 7944 | 0.99x | OK | 0.00000 |

## TTFT vs depth — tuned curve (full_mult=16)
| p_start | eff_pct | prefill_ms | splice_ms | speedup | top1 | KL |
|--------:|--------:|-----------:|----------:|--------:|:----:|----:|
| 256  | 5.0  | 1779 | 1089 | 1.63x | OK | 0.00018 |
| 512  | 11.3 | 2726 | 1986 | 1.37x | OK | 0.00017 |
| 1024 | 24.0 | 4281 | 3747 | 1.14x | OK | 0.00012 |
| 2048 | 49.3 | 7525 | 7649 | 0.98x | OK | 0.00116 |

N=250 SLB search latency: ~6-8 us (median) — tool count adds negligible TTFT.

## Honest conclusions
- **Accuracy: never-regress holds at every depth and every curve** (top1 OK, KL ~0).
- **TTFT is NOT "completely flat"** under the never-regress constraint: deep splice gives
  1.1-1.6x speedup through moderate/deep context, converging to prefill-parity at very
  deep context as the adaptive recompute escalates. Flat-TTFT and never-regress are in
  genuine tension (flagged in the plan's risk section).
- **`full_mult` is the accuracy<->flatness knob.** The default (4) is conservative;
  full_mult=16 sustains 1.1-1.6x through p=1024 with top-1 intact for this workload.
  Set per-workload via `agent.enable_deep_splice(full_mult=...)`.
- **N=250 routing accuracy = 83% (>=81% target met)** is established and depth-invariant
  (SLB search is position-independent); see results/_v2_phase5_routing_accuracy.json.
