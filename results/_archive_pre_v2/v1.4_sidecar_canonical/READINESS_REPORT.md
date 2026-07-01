# Semantic IR v1.4 — Readiness Report (NOT PROMOTED)

**Verdict: gate FAILED. v1.4 is not promoted. v1.3 remains canonical.**

IR v1.4 added, to `_compress_schema_to_ir`: (a) deterministic field ordering
(high-entropy identifiers first), (b) a type-aware one-shot JSON exemplar, plus an
anti-hallucination `ARG_GEN_DIRECTIVE` in the arg-gen prompt. Goal: fix owner↔repo
semantic slot confusion (the v1.3 residual) without breaking the latency/validity bars.

## Measured (n=30, this run)

| metric | v1.3 canonical | v1.4 | gate |
|---|---|---|---|
| specified_field_consistency | 0.926 | **0.894** | regressed |
| json_valid_rate_hybrid | 0.967 | **0.933** | ≥0.96 → FAIL |
| ttft_first_arg_token_ms_hybrid_p50 | 451.6 | **790.2** | <500 → FAIL |
| ttft oracle p50 | 720.5 | 1000.7 | — |
| speedup vs oracle | 1.59× | 1.27× | — |
| ir_tokens_p50 | 56 | 71 | <80 ok |
| routing_accuracy | 0.867 | 0.867 | — |
| coref_tool / coref_arg | 1.0 / 1.0 | 1.0 / 1.0 | — |

Per-case (specified_agreement): improved 0, regressed 3 (cases 2, 6, 9), same 23.

## Why it failed — three findings

1. **The fix actually worked, but the metric can't see it.** On the two canonical
   owner↔repo cases the IR arm is now MORE correct: case 2 v1.4 hybrid
   `owner="user", repo="run-tasks"` (correct) vs v1.3 `owner="run-tasks", repo="scripts"`
   (wrong); case 3 likewise `owner="user", repo="server-monitor"` (correct). But the
   **Oracle arm mis-slots those same cases** (`owner="run-tasks"` / `owner="server-monitor"`),
   so the now-correct hybrid *disagrees* with the wrong oracle and `specified_agreement`
   drops 0.5→0.33 / →None. The gate (IR-vs-Oracle agreement) **structurally punishes the
   very fix it asked for.** This is decisive evidence the metric is unsound for this task;
   a gold-args accuracy metric would score cases 2/3 as fixes.

2. **Exemplar placeholder leakage (new regression).** Generic-but-real exemplar values get
   copied literally: `repo="repo"` (cases 4, 6), and `branch="main"` in 6 cases. The
   exemplar's `"branch":"main"` **directly contradicts** the directive's "do not guess
   'main' for a branch" — and the exemplar wins. This leakage is the dominant new error and
   outweighs the slotting gains in the aggregate.

3. **Latency regressed past the gate.** IR grew +15 tok and the directive added ~55 prompt
   tokens; relative speedup fell 1.59×→1.27×. Absolute TTFT also inflated for BOTH arms
   (720→1000 on the unchanged oracle grounding), so host-level variance contributes — but
   790ms fails the <500ms bar as measured. (Run once per directive; not re-run to chase a
   nicer number.)

## Recommendation
Do not promote. The exemplar concept has merit (fixed real owner/repo slotting) but, as
specified, the literal placeholders leak and the gate metric cannot validate the fix.
Next-step options (user's call): (a) switch the gate to gold-args accuracy so real fixes are
measurable, then re-evaluate; (b) iterate the exemplar to non-literal placeholders
(`"<owner>"`) and drop branch/optional fields from it to stop the main conflict; (c) revert
v1.4. Environment note: run requires `DYLD_LIBRARY_PATH=build/external/llama.cpp/src`
(cb2463bb libllama with `llama_backend_sync`); the venv's bundled libllama is ABI-mismatched
with the prebuilt `nexus_fsm_ext.so`.
