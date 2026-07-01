# Nexus Paper Ground-Up Rewrite — Plan & Record (2026-06-20)

Source of truth ladder: (1) codebase → (2) `results/v1.1_canonical/` → (3) reconciliation CSVs → (4) prose → (5) legacy/quarantined. Prose never outranks code or canonical artifacts.

## Executive verdict
The pre-rewrite `main.tex` was a half-corrected state: an erratum block bolted on top of **two** `\begin{abstract}` environments (a LaTeX error), while Table I, the ablation table, all four figures, the §VI body, related work, and the conclusion still printed stale Run-B numbers (237.4 ms, 6.3×, 52×, "eliminates", "strictly dominates", "equivalent accuracy"). It has been rewritten from the ground up to the canonical bundle. The paper now compiles to **5 pages, 0 errors, 0 undefined refs/citations**.

## What the evidence supports (and only this)
- Nexus **mitigates** tool-schema prefill for a **bounded shallow-prefix regime** (P ≤ 256). It does not solve MCP bloat, eliminate prefill, generalize, or run in production.
- End-to-end TTFT (true, incl. first token, n=30 clean serial): **N1x 518 ms vs B_RP 1214 ms = 2.34×**, at ~6-point accuracy cost (0.87 vs 0.93; gateway routing 0.89 at n=500).
- Routing+splice latency (NOT TTFT): **160.2 ms** (n=500). L0 **3.06 µs / 69.5%**. Dense R@1 **0.87**, CE R@1 **0.88** (target missed). τ bit-exact.
- Negative results: RoPE P1024 divergence (pre-freeze + g4 corroboration); block-diagonal masking acc 0.0 (n=20, well supported).

## Manuscript rewrite diff (main.tex)
- **Title:** "Eliminating…" → "Bypassing Tool-Schema Prefill … in a Bounded Shallow-Prefix Regime".
- **Abstract:** removed double-abstract + erratum hack → one clean abstract; canonical numbers; route-only vs TTFT defined; explicit single-host/model scope; 0.90→0.88; trade-off not parity.
- **Intro:** "does/does-not" framing; contributions tagged by evidence class; removed "eliminates".
- **Background:** eq:attention and eq:rope-drift kept with explicit "definitional, not predictive" qualifiers; notation normalized (n_past, m0, base b=1e6).
- **Architecture/Systems:** Path A/B honest; Path B flagged "wired, not exercised end-to-end"; zero-copy restricted to block residency vs seq_cp splice; zero-alloc scoped to C++ hot path.
- **ML:** eq:intent-signature labeled conceptual heuristic; CE 0.90→0.88 (+ removed fabricated 91% combined); CE described as ~22M MiniLM-class (verified from config.json); leakage stated as provenance not measured.
- **Evaluation:** Table I regenerated to true-TTFT canonical (9754/1214/518); new routing-micro paragraph (160 ms, L0); CDF+Pareto → `*_v1_1.pdf` from canonical, captions de-overclaimed; ablation kept but labeled **legacy/not-regenerated**; scalability table/figure removed from main, SLB cited as legacy text; explicit rejection of 6.3× with the 7.6× arithmetic.
- **Negative results:** RoPE table labeled **pre-freeze**; "memory-bandwidth tradeoff" causal claim → "plausible but unmeasured, future work"; causal-isolation kept as strongest finding.
- **Related work:** removed "eliminates it entirely" and "true architectural novelty"; "targets a different regime", no outperformance claim.
- **Conclusion:** removed "eliminates"/futurism/"intelligent memory management" manifesto; ends on measured bounded takeaways + open items.

## Generated assets
- `docs/paper/scripts/regen_canonical_paper_data.py` (NEW): artifact-driven, reads `results/v1.1_canonical/` only → `docs/paper/data/v1_1/*.csv` + `fig_*_v1_1.pdf`.
- Stale figures → `docs/paper/figures/_legacy_prefreeze/`.
- `PAPER_TABLES.tex` rewritten to canonical (provenance header; generator rewiring flagged Phase B).
- `sections/SUPERSEDED.md` marks the vestigial modular mirror.
- `verify_paper_math.py` already reads canonical (2.34× derived).

## Documentation rewrite (done previous pass, re-verified consistent)
README, RELEASE_V1, architecture.md, internals.md, results/README all carry canonical numbers (160 ms route-only; 518/1214 ms TTFT; 2.34×; L0 69.5%/3.06 µs; CE 0.88) and the route-only-vs-TTFT distinction.

## Multi-phase improvement plan
**Phase A — Evidence Integrity Finalization.** Obj: zero residual stale numbers. Tasks: confirm no 237.4/6.3×/0.90 outside corrective context (done); lock canonical refs. Stop/go: grep clean. Risk: low. Deliverable: this rewrite + audit CSVs.

**Phase B — Pipeline Hardening.** Obj: artifact-driven tables/figures, fail-on-missing-canonical. Tasks: rewire `generate_latex_tables.py`/`generate_master_metrics.py` to read `results/v1.1_canonical/` (or `data/v1_1/`); make `main.tex` `\input` regenerated tables/sections instead of inline; CI check that every paper number traces to a canonical artifact. Deps: A. Stop/go: `make tables figures` reproduces main.tex numbers from canonical only. Risk: generator refactor churn.

**Phase C — Codebase-Proof Hardening.** Obj: invariant + metric-semantics tests. Tasks: assert `MAX_SPLICE_POS` Py==C++; test `recompute_pct` token count; test route-only metric sets decode_us=0 (naming guard); emit `IMPLEMENTATION_MANIFEST.json`. Deps: none. Stop/go: CI fails on constant divergence or metric mislabel.

**Phase D — Missing Experiments.** Obj: close gaps the paper flags. Tasks: regenerate RoPE boundary (`aggregate_physics_boundary.py`) and SLB scalability (`bench_scalability.py`); exercise Path B (P>256) end-to-end; add second host/model; raise e2e n and report CIs/IQR. Deps: host time. Stop/go: pre-freeze labels removed; Path B measured. Risk: Path B numbers may be unflattering — report honestly.

**Phase E — Venue Positioning.** Obj: claims = evidence. Decision: position as a **systems-prototype, negative-results-forward short/workshop paper** (one host, one model, n=30 e2e, two solid negative results). Lead with the RoPE boundary + causal-isolation findings and the honest 2.34× bounded result. Deps: A–D. Stop/go: title/abstract/claims all trace to canonical artifacts.

## Final risks / open gaps
1. e2e at n=30 only; accuracy CIs wide. 2. RoPE + SLB pre-freeze (not regenerated). 3. Path B never exercised end-to-end. 4. Single host / single model. 5. Generator still reads stale CSVs (Phase B). 6. `sections/` modular mirror is vestigial pending Phase B decision (delete or wire in).
