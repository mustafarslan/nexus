# Phase 2 Review — Deep-Path L0 Reuse

**Date:** 2026-06-20 · **Scope:** measurement design + read-only instrumentation only.
**Status:** **CLOSED — verdict reached from code + instrumentation; no engine experiments run.**
**Verdict (one line):** Path B deep-path L0 prefix reuse is **NON-VIABLE under the current population policy** — a structural property of the code, not a tuning gap.

> Honesty note (canonical standard): this review reports a **code-derived structural result**, not a measured one. No E0/E1/E2/E3 runs were executed (the nomic embedding model is absent on this host, and the deep-workload drivers were intentionally not implemented). Where a claim is structural vs empirical is marked explicitly. No numbers are presented as measured.

---

## 1. Instrumentation landed (read-only)

F0 (Phase 1) + Phase 2 added deep-path observability with **no behavior change** to Path A or the Path B text-prefill fallback. All additions are read-only; none consume L0, populate the pool on the deep path, or alter fallback semantics.

| Element | Location | Purpose |
|---|---|---|
| `probe_prefix()` (read-only twin of `try_copy_prefix`, shared `match_prefix_locked`) | `src/nexus_seq_warm_cache.{hpp,cpp}` | compute `best_end` for a deep prefix without `seq_cp`/LRU/ref mutation |
| `deep_path_{entered,l0_hit,l0_miss,text_fallback}` counters + getters | `src/nexus_orchestrator.{hpp,cpp}`, `src/bindings.cpp` | deep-path entry / full-match / miss / fallback accounting |
| `last_deep_n_past`, `last_deep_best_end` raw recording + getters | `src/nexus_orchestrator.{hpp,cpp}`, `src/bindings.cpp` | per-deep-call `(n_past, best_end)` so a harness can build the `best_end` histogram / `r = best_end/n_past` CDF offline |
| `NexusAgent.deep_path_telemetry()` | `src/nexus_agent.py` | Python accessor for all of the above |

**Metric contract (unchanged from approved plan):** `deep_path_l0_hit` is **opportunity only** (a *would-copy* full match), never a speedup, never consumed, never TTFT. `best_end` is in KV positions, not bytes/ms. Build verified: clean compile, links, all getters bound and callable. Counter invariant: `entered == l0_hit + l0_miss == text_fallback` (F0).

---

## 2. Code-derived structural cap on `best_end`

The L0 warm pool is populated by exactly one call site, `update_from_seq`, invoked **only on Path A** (`src/nexus_orchestrator.cpp:451`, `:453`), with:

```
update_from_seq(ctx, prefix_tokens = input_ids[:n_past], end_pos = n_past + schema_len, ...)
```

On Path A, `n_past ≤ max_splice_pos_ = 256`. Therefore the radix only ever indexes conversation-prefix token sequences of length `≤ 256`. The deep path (`n_past > 256`) is never a population source — F0 deliberately performs no deep-path write-back.

**Consequence:** for any deep request, the longest-common-prefix match against a pool that contains only `≤256`-length prefixes is bounded:

```
best_end ≤ 256   (for all n_past > 256)
⇒ r = best_end / n_past < 1,  and r → 0 as n_past grows
```

This is a **structural cap**, observable from the code without running anything. The Phase 2 instrumentation, if run, would quantify *where under 256* the residue sits — but cannot exceed it.

---

## 3. Why deep full-match is impossible under current population

A `deep_path_l0_hit` requires `best_end ≥ n_past`. With `best_end ≤ 256` and `n_past > 256`, the condition `best_end ≥ n_past` is **never satisfiable**. So:

- **Full deep-path match rate = 0, by construction** (not "approximately zero pending data" — structurally zero under the current population policy).
- Every deep partial is **Path A shallow residue** (`best_end ≤ 256`), which falls in the "noise-level partial" class (`best_end ≤ 256`) defined in the Phase 2 plan. Residue is worthless for the deep path: the first ≤256 tokens are already reachable by Path A and, on the live deep path, already resident in the context KV.
- This is the **H2 (population-suppressed) mechanism** the plan set out to test — confirmed in code. H1 vs H2 is therefore moot for a *go* decision: even if latent deep recurrence exists (H1-style opportunity), the current design cannot see or use it, and seeing it would require a population-policy change that is explicitly out of the approved scope.

**Second, independent reason L0-prefix is the wrong lever (residency):** L0's prefix tier reconstructs *conversation-prefix* KV. On the steady multi-turn deep path the prefix is already resident, so even a hypothetical full match saves no prefill. The cost Path B actually re-pays is **schema + query prefill at depth**, which the prefix tier does not address at all. So deep-path L0 is doubly non-viable: structurally capped, and aimed at the wrong cost component.

---

## 4. Why further measurement is low-ROI

- The decisive question ("can deep full-match exist?") is answered **No** from code; running E1/E2 would only attach a histogram to a foregone conclusion (`best_end ≤ 256`).
- The latency-relevance test would measure an upper bound on skippable prefix-reconstruction work that is **zero in the resident-prefix case** and, in the cold-prefix case, still aimed at the prefix component rather than the dominant schema+query prefill.
- Engine runs are blocked here anyway (no embed model) and would require building E1/E2/E3 drivers — additional implementation deliberately not undertaken.
- **Therefore measurement is deferred, not abandoned.** It becomes worthwhile only as **publication evidence** — i.e., if the paper needs an empirical `best_end`/`r` distribution to substantiate the structural claim for reviewers. In that case the landed instrumentation is sufficient; only the deep drivers + a host with the embed model would be needed, and the expected result is "all deep matches `≤256`, full-match rate 0." No product decision depends on it.

---

## 5. Pivot recommendation

**Stop Path B deep-prefix L0 work under the current design.** It is structurally non-viable and targets the wrong cost. Do not implement consume-on-hit, consume-on-partial, or deep-path population to rescue it; those would be effort spent fighting a population/residency mismatch rather than the real bottleneck.

**Pivot the deep-path question to Tool IR / deep schema-prefill reduction.** The measured-and-structural evidence points at one true Path B cost: text-prefilling the tool schema + query at depth (Path B ≡ the B3 baseline today). The high-leverage question is no longer "can we reuse deep KV?" but **"can we shrink the schema+query prefill that Path B must pay at any depth?"** — e.g., a compact tool representation (Tool IR) that the model can prefill in far fewer tokens than the raw JSON schema, with raw-schema fallback when fidelity is at risk. This works *with* RoPE (it is ordinary decode), helps at every depth, and does not depend on the splice/L0 machinery.

This pivot is recommended as a **review plan only** — design and decision gates first, no implementation — and is written up separately.

---

## Appendix — what would reverse this verdict

The verdict is scoped to **the current population policy**. It would need re-examination only if a future design deliberately (a) populates the pool with deep prefixes (a population-policy change, explicitly out of current scope) **and** (b) demonstrates that deep prefixes recur often enough **and** (c) shows the prefix is non-resident (cold-serving) so reuse saves real prefill. Absent all three, deep-path L0 stays non-viable. None of these is pursued here.
