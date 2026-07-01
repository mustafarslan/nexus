# Nexus Paper Revision — Changelog

Maps every edit to a referee item. Tags: **P0-1…P2-5** from `nexus_review.md` (MLSys-level
review); **FPR-#** from `FINAL_PEER_REVIEW_REPORT.md` (forensic review, applied to `main.tex`
before this revision). Verification: `scripts/verify_paper_numbers.py` (60/60 MATCH, 0 discrepancies).
Branch: `paper-revision-referee`.

| Review item | Change | File(s) | Commit |
|---|---|---|---|
| Phase 0 / P1-verify | Scaffold + re-derived all 60 claims from artifacts (0 discrepancies) | `CHANGELOG.md`, `results/metrics_inventory.md`, `results/verification_report.md`, `scripts/verify_paper_numbers.py` | `6d2034f` |
| P1.3 | Eq.(2) reproduces Table II R(%) — verified note in `tab:perf` caption | `main.tex` | `6d2034f` |
| P0-1 | RAG-MCP + in-place-KV comparison table (`tab:comparison`, †contextual); §II reframe (dropped "the baseline we compare against") | `main.tex`, `results/comparison_sources.md` | `b06163f` |
| P0-2 | never-regress = output fidelity, not latency (abstract + §VII-A) | `main.tex` | `5d43008` |
| P0-4 | Artifact availability statement (§I) | `main.tex` | `5d43008` |
| P1-2 | Fixed "~100% token reduction" → ratio undefined (oracle overflow); use 19-tok IR / ≈80% | `main.tex` | `5d43008` |
| P1-4 | Lead with routing (abstract + contributions reorder) | `main.tex` | `5d43008` |
| P1-5 | Split qualitative (general) vs quantitative (tuple-specific) generality (abstract + conclusion) | `main.tex` | `5d43008` |
| P2-2 | RQ1/RQ2/RQ3 framing at end of §I | `main.tex` | `5d43008` |
| P2-3 | Tightened Table I argument (drift varies 175–207; proxy uninformative, ρ=0.193) | `main.tex` | `5d43008` |
| P2-4 | Expanded IR (intermediate representation) on first use | `main.tex` | `5d43008` |
| P2-1 | Trimmed abstract, moved M4/git specifics to §I | `main.tex` | `5d43008` |
| **P0-3 / P1-3** | **Ran D_KL-vs-offset sweep on Qwen (bare splice, 0–2048); new `fig:dkl`; §IV-A rewrite** | `test/bench_dkl_sweep.py`, `results/v2.0_canonical/raw/dkl_sweep.json`, `main.tex` | `37d1b99` |
| P1-1 | Wilson CIs (Table III + inline §VII); Table II speedup [95% CI]; Figs 3–4 error bars; methods sentence; sidecar bench emits exact counts | `main.tex`, `test/bench_sidecar_accuracy.py` | `42151dd` |
| ledger | Fixed stale speedup rows, reworded drift row, registered `dkl_sweep.json`, arg-level denominators | `results/v2.0_canonical/EVIDENCE_LEDGER.csv` | _this commit_ |
| FPR-3/4/5/6/7/8/9/10/11 | Hardware, LegoLink framing, SLB-latency split, scope warning, 4 citations, Aeon, typography, TikZ | `main.tex`, `references.bib` | pre-revision (verified present) |

## Key scientific finding (P0-3)
The D_KL sweep on the **current contiguous-suffix mechanism** shows the bare (recompute-disabled)
splice holds `D_KL ~1e-2` nats with **top-1 agreement = 1.0 across the entire 0–2048 offset range**
— two orders of magnitude below the retired LegoLink scattered-recompute point (5.72 nats @1024).
So off-anchor drift is **gradual and bounded, not a cliff**, and LegoLink was worse than doing
nothing. §IV-A now states this candidly: `P=256` is a **conservative repair-start**, not an
empirical knee; the depth-adaptive recompute trades a little TTFT to drive small residual drift to
exactly 0 (a hard fidelity guarantee), which motivates the *contiguous*-suffix design.

## Denominator correction (P1-1)
Sidecar arg metrics are **argument-level**: routed-only 40/40 (100%, CI ≥91.2%), e2e 40/50 (80%,
[67.0, 88.8]) — not the reviewer-inferred 80/100 and 30/30. Case-level routing is 26/30.

## Open TODOs
- **Availability (P0-4):** currently an on-request note; replace with a public repo/tag or Zenodo DOI
  when available.
- **Comparison table (P0-1):** contextual (†) only; a true RAG-MCP head-to-head on the M4 host would
  promote it to a controlled comparison (would need a RAG-MCP harness).
- `dkl_sweep.json` was generated at branch commit (splicer identical to `1ce4aa4`); note the sha in
  the artifact if promoting to the canonical bundle.
