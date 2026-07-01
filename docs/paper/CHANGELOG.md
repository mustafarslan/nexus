# Nexus Paper Revision — Changelog

Maps every edit to a referee item. Tags: **P0-1…P2-5** from `nexus_review.md` (MLSys-level
review); **FPR-#** from `FINAL_PEER_REVIEW_REPORT.md` (forensic review, mostly already applied
to `main.tex` before this revision). Verification: `scripts/verify_paper_numbers.py` (60/60 MATCH).

| Review item | Change | File(s) | Commit |
|---|---|---|---|
| Phase 0 | Scaffold: changelog, metrics inventory, verification script/report | `docs/paper/CHANGELOG.md`, `results/metrics_inventory.md`, `results/verification_report.md`, `scripts/verify_paper_numbers.py` | _pending_ |
| P1 verify | Re-derived all 60 claims from artifacts → 0 discrepancies | `results/verification_report.md` | _pending_ |
| P1.3 | Eq.(2) reproduces Table II R(%) — add verified note to `tab:perf` caption | `main.tex` | _pending_ |
| P0-1 | RAG-MCP comparison table + §II reframe (contextual not head-to-head) | `main.tex`, `references.bib`, `results/comparison_sources.md` | _pending_ |
| P0-2 | never-regress = fidelity (not latency) clarification | `main.tex` (abstract, §VII-A) | _pending_ |
| P0-3 | D_KL-vs-offset sweep + `fig:dkl` + §IV-A rewrite | `test/bench_dkl_sweep.py`, `results/v2.0_canonical/raw/dkl_sweep.json`, `main.tex` | _pending_ |
| P0-4 | Artifact availability statement | `main.tex` | _pending_ |
| P1-1 | Wilson CIs (Table III) + bootstrap TTFT/ratio CIs (Table II) + methods sentence | `main.tex`, `test/bench_v2_capstone.py` | _pending_ |
| P1-2 | Fix "~100% token reduction" → ratio undefined (oracle overflow) | `main.tex` (§VII-B) | _pending_ |
| P1-3 | Distinguish LegoLink failure from RoPE boundary (folded into P0-3 §IV-A) | `main.tex` | _pending_ |
| P1-4 | Lead with routing (abstract + contributions + §IX reorder) | `main.tex` | _pending_ |
| P1-5 | Split generality: qualitative (general) vs quantitative envelope | `main.tex` | _pending_ |
| P2-2 | RQ1/RQ2/RQ3 at end of §I | `main.tex` | _pending_ |
| P2-3 | Tighten Table I argument (proxy uninformative; drift has spread) | `main.tex` (§IV-B) | _pending_ |
| P2-4 | Define TTFT/SLB/IR/.atb/UMA on first use | `main.tex` | _pending_ |
| P2-1 | Trim abstract toward ~200–250 words | `main.tex` | _pending_ |
| P2-5 | Consistent sig-figs | `main.tex` | _pending_ |
| FPR-3 | M4 Max hardware profile | `main.tex` (§I, §VI) | already applied |
| FPR-4 | LegoLink KL retired framing | `main.tex` (abstract, §IV-A) | already applied |
| FPR-5 | SLB latency FFI vs pure-SIMD split | `main.tex` (§VII-B, Table III) | already applied |
| FPR-6/9 | Scope & generality (Gemma4/cloud) warning | `main.tex` (§I, §VI-B, §VII) | already applied |
| FPR-7 | Citations TSCG/NTILC/RedKnot/Leyline | `references.bib`, `main.tex` §II | already applied (verify) |
| FPR-8 | Aeon v3 relationship | `main.tex` §II | already applied |
| FPR-9/10 | LaTeX typography (`D_KL` math, `\code{}`) | `main.tex` | already applied (verify) |
| FPR-11 | TikZ: re-entrant lock (Fig 1), mmap block (Fig 3) | `main.tex` | already applied |

## Open TODOs
- (Phase 7) Availability: use on-request note unless the author supplies a public repo/DOI.
- (Phase 2) Comparison table is contextual (†) unless a RAG-MCP head-to-head is run on the M4 host.
