# Phase 2 — Content-Driven Micro-Gating: GO/NO-GO Result (2026-06-30)

## Shipped: Component A — Memory alignment hardening
`src/nexus_bitmap_allocator.hpp`: `QuantizedBitmapAllocator` is now `alignas(128)`
(NEXUS_CACHE_LINE) with the contended `mutex_` pinned to its own cache line and
`static_assert`s on alignment + whole-line size. False-sharing / RFO-storm guard.
Build clean; 19 regression tests green; zero behavior change.

## NOT shipped: Components C/D — online micro-gate (failed the go/no-go)

The micro-gate needs a cheap, reference-free runtime signal that predicts whether a
deep splice will catastrophically drift on global heads. Two findings kill it:

1. **The proxy doesn't predict drift.** Per-head live preceding-context K-variance vs
   TRUE per-head splice drift (`test/profile_head_drift.py`, Qwen-14B, tool_0.atb):
   mean Spearman = **0.193** (threshold 0.4). Reference-free signals (live variance,
   RoPE positional leverage) cannot see the context-dependence that actually drives drift.

   | p_start | ctx | maxDrift | meanDrift | spearman |
   |--------:|----:|---------:|----------:|---------:|
   | 256  | on  | 207.1 | 105.2 | 0.147 |
   | 256  | off | 195.3 |  97.8 | 0.221 |
   | 1024 | on  | 193.2 |  98.4 | 0.182 |
   | 1024 | off | 189.1 |  96.4 | 0.250 |
   | 2048 | on  | 194.7 |  97.6 | 0.159 |
   | 2048 | off | 175.4 |  91.0 | 0.197 |

2. **There is no per-request drift variance to gate on.** Mean drift is ~91–105 across
   every depth and both on-/off-topic contexts — nearly constant. A scalar-threshold gate
   would either always-fire or never-fire; nothing to discriminate. And Phase 1's adaptive
   recompute already repairs this drift (capstone: top-1 held, KL→0 at all depths).

## Decision
Per the approved honesty-first plan, ship Component A only; keep Phase 1's depth-adaptive
recompute curve as the Path-A↔B mechanism (validated never-regress). A working content-driven
gate would require a true reference (the full-prefill cost we are avoiding) or a custom
attention kernel — out of scope. The go/no-go correctly prevented shipping a non-predictive
heuristic.
