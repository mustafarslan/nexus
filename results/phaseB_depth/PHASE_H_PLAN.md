# Phase H — Path A Hardening & Generalization (REVIEW-ONLY EXECUTION PLAN)

**Status:** planning only. No runs, no code, no paper edits. Approval-gated.
**Premise:** Path A (P≤256 ATB splice + 5% suffix recompute) is the only validated win — true-TTFT N1x 518 ms vs B3 1214 ms = 2.34× (n=30, single host, single model, −6 pt accuracy, CE 0.88<0.90). The load-bearing weakness is **generality**, not features.

## H1 — Second-model validation
- Objective: show the headline reproduces on a model with different RoPE base / GQA config.
- Tasks: pick a second GGUF (e.g. a different-family or different-size instruct model); re-run gateway (route-only) + e2e (true TTFT) under identical harness/provenance discipline; verify `.atb` ABI assumptions (`rope_freq_base/scale/scaling_type`, `n_head_kv`, `d_head`) still hold or document where they break; confirm the P=256 boundary still yields KL within tolerance.
- Risk: a different RoPE base may move the viable splice depth or break ABI → would narrow the claim.

## H2 — Second-host validation
- Objective: separate machine-state drift (the 171/237 ms artifact) from real latency.
- Tasks: run the canonical suite on a second Apple-Silicon host (and, if available, a CUDA host to exercise the NUMA/PCIe splice path); report route-only P50 and true TTFT with provenance blocks; compare against the 160.2 ms / 2.34× canonical.
- Risk: second host shifts absolutes → reinforces that single-number headlines are unsafe (report ranges).

## H3 — E2E sample-size increase + CIs
- Objective: replace n=30 point estimates with defensible intervals.
- Tasks: raise e2e to n≥100 per arm, serial-only (no concurrency — the n=25 contamination lesson); report median + IQR + bootstrap CI for TTFT and accuracy on N1x/N1/B3/B3pc/B1.
- Risk: at n≥100 the 2.34× may compress or the −6 pt accuracy gap may widen → honest narrowing.

## H4 — Accuracy-gap track
- Current gap: N1x acc 0.87 vs B3 0.93 (≈6 pt) at n=30; gateway routing 0.89 at n=500; CE R@1 0.88 vs 0.90 target.
- Likely causes (ranked): (1) thin calibration set (n=53 adversarial) → τ may be mis-set; (2) small CE training set (226 synthetic pairs); (3) MiniLM-class CE ceiling; (4) retrieval recall ceiling (dense R@1 0.87).
- Cheapest experiments first: (a) widen the calibration set and re-derive τ (cheap, no retrain); (b) error-analysis of the N1x vs B3 disagreement set to localize loss (cheap); (c) expand CE training pairs and retrain (medium); (d) only if needed, a stronger CE (expensive — see guardrail).
- Guardrail (tuple-relative) — no accuracy gain is treated as free. Any CE/calibration change must report its latency cost (route-only and, where relevant, true TTFT) **within the same (model, host, build, embed-model) tuple**, and the change is acceptable only if the within-tuple Path A value proposition still holds after the latency cost is accounted for. There is no fixed millisecond threshold; the bar is that the latency/accuracy trade stays net-positive for Path A in that tuple, and any change that does not is reported as a trade-off rather than a win.

## H5 — Regenerate pre-freeze evidence
- RoPE boundary: run `aggregate_physics_boundary.py` to replace the pre-freeze `physics_boundary_rope.csv`; confirm KL@P256 (~0.0076) and KL@P1024 (5.72) reproduce; remove "pre-freeze" labels only if they do.
- SLB scalability: run `bench_scalability.py` for the N=10→1000 curve with real n; replace the thin/pre-freeze `scalability_curve.csv`.
- Risk: regenerated numbers differ from pre-freeze → must restate, not re-assert.

## H6 — Repo/paper evidence hygiene
- Objective: every surviving claim traces to a Phase-H canonical artifact; route-only vs true-TTFT distinguished everywhere; single-number headlines replaced by ranges/CIs.
- **All future headlines must cite Phase-H canonical artifacts only** — the repo previously carried mixed old/new state and stale metric cross-citations (per `docs/AUDIT_2026-06-20_LIVE.md`), and no headline may cite pre-Phase-H or legacy artifacts.
- Output: reconciled metric map (documentation/claim-trace only, no code change).
- Success: every number traces to a Phase-H artifact; no route-only-as-TTFT, no single-number headline, no stale citation.

## Exit criteria (successful hardening)
- Headline (route-only P50, true-TTFT speedup) reproduces within CI on ≥1 second model AND ≥1 second host.
- e2e reported at n≥100 with CIs/IQR; no single-number headlines.
- Accuracy gap either closed to within a stated tolerance or honestly bounded with a Pareto statement.
- RoPE + SLB regenerated; no "pre-freeze" labels remain.

## Kill / narrow criteria (force a narrower claim)
- Second model breaks `.atb` ABI or moves the P=256 boundary → claim narrows to "single model family."
- Once intervals are reported (n≥100, CIs/IQR), if the hardened like-for-like true-TTFT advantage degrades to a *marginal* result — i.e. the speedup interval no longer cleanly separates Path A from B_RP, or the gain is small enough that the accuracy cost dominates the picture — then narrow or drop the TTFT framing (reposition as a throughput/secondary result). The trigger is "advantage is marginal once intervals and the accuracy trade are honestly reported," not a fixed multiplier.
- Accuracy gap proves CE-ceiling-bound and only closeable by adding latency → publish as an explicit latency/accuracy trade-off, drop any "equivalent accuracy" framing entirely.
- Second host shows the win is Metal-UMA-specific and absent on CUDA → scope to Apple-Silicon.
