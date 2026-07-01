# Semantic IR v1.5 — Readiness Report (NOT PROMOTED)

**Verdict: gate FAILED (3 of 4 conditions). v1.5 is not promoted. v1.3 remains canonical.**

V1.5 implemented the four approved fixes exactly:
1. Exemplar leakage fix — abstract `<...>` placeholders (`IR_EX_STR` → `{"owner":"<owner>",
   "repo":"<repo>","path":"<path>","name":"<name>"}`), exemplar capped to top-3 ranked
   fields, high-drift optionals (`branch`/`message`/`content`/`description`) dropped
   (`IR_EX_DROP`). For `create_or_update_file` the rendered exemplar is
   `EX:{"owner":"<owner>","repo":"<repo>","path":"<path>"}`.
2. Latency — `ARG_GEN_DIRECTIVE` compressed from ~55 tokens of English to the dense
   14-token rule.
3. Truncation — bench `--max-tokens` default 64 → 128.
4. Gate — abandoned Oracle-agreement; evaluated the 4 absolute conditions below.

Run: `DYLD_LIBRARY_PATH=build/external/llama.cpp/src PYTHONPATH=build:src:test
.venv/bin/python test/bench_sidecar_accuracy.py --limit 30 --out-dir
results/v1.5_sidecar_canonical` (exactly once, no re-runs).

## The 4 promotion gates (measured, n=30)

| # | Condition | Target | Measured | Result |
|---|---|---|---|---|
| 1 | Zero `<owner>`/`<repo>`/`<path>` leakage in hybrid args | 0 leaks | **1 record (case 4)** | **FAIL** |
| 2 | Cases 2/3 slot `owner="user"`, `repo="run-tasks"`/`"server-monitor"` | both | **neither** | **FAIL** |
| 3 | `ttft_first_arg_token_ms_hybrid_p50` | < 500 ms | **595.4 ms** | **FAIL** |
| 4 | `json_valid_rate_hybrid` | ≥ 0.96 | **0.9667** | **PASS** |

Supporting context: routing 0.867, coref tool/arg 1.0/1.0 (unchanged), ir_tokens_p50 73
(v1.4 71 / v1.3 56), ttft oracle p50 721 ms (speedup 1.21×).

## Why it failed — the decisive finding

**Conditions 1 and 2 are mutually contradictory as specified.** The directive assumed the
`<...>` placeholders would *preserve* the V1.4 `owner="user"` result while *stopping* the
leakage. The data shows those were the same mechanism:

- In V1.4 the exemplar contained the literal `"owner":"user"`. The reason cases 2/3 produced
  `owner="user"` was that the model **copied the literal "user" from the exemplar** — i.e. it
  was leakage, indistinguishable in kind from the `repo="repo"`/`branch="main"` leaks V1.4
  flagged. It only *looked* like a semantic fix because "user" is a plausible owner.
- V1.5 replaced `"user"` with `<owner>`, removing the only source of the string "user". So
  the owner slot is now unanchored and unstable across the 30 cases:
  `user123 ×3, alice, bob, run-tasks, <owner>, database-utils, open-source-project, "" `.
  Case 2 → `owner="run-tasks"` (repo name mis-slotted into owner), case 3 → `owner="user123"`.
  **Neither is `"user"`** → condition 2 fails.
- Meanwhile the abstract syntax reduced but did **not eliminate** copying: case 4
  ("…repo **auto-tests** on dev branch") emitted `{"owner":"<owner>","repo":"<repo>",
  "branch":"dev"}` — it leaked both placeholders **and ignored the specified `repo=auto-tests`**
  → condition 1 fails.

So the only way to satisfy condition 2 (`owner="user"`) is to put a literal default back in
the exemplar — which is exactly the leakage condition 1 forbids. The two gates cannot both
be met by tuning the exemplar.

**Latency (condition 3) improved but missed.** Compressing the directive moved
`ttft_hybrid_p50` 790 ms (v1.4) → 595 ms, but did not clear 500 ms. The 3-field exemplar
barely changed IR size (73 tok vs v1.4's 71) because the inline field *descriptions*, not the
exemplar, dominate the IR token count. Host-level variance (documented in prior runs) also
contributes to the absolute.

## Recommendation

Do not promote; v1.3 stays canonical. The leakage-vs-default contradiction means no exemplar
edit satisfies both arg-correctness gates simultaneously. Genuine next steps (user's call):
- **Drop the JSON exemplar entirely** and rely on the typed IR signature + a gold-args
  accuracy metric — the exemplar's only demonstrated effect is leakage, and "owner=user" was
  never a real inference.
- **Shrink the IR** by trimming/removing inline `/* descriptions */` (the real token cost) to
  attack latency, rather than the exemplar.
- **Replace the gate** with a small hand-labelled gold-args set (owner/repo/path expected
  values per query) so correctness is measured against truth, not against a literal the model
  happened to copy.

Artifact: `results/v1.5_sidecar_canonical/accuracy.json`. No `CANONICAL_RESULTS_MANIFEST.json`
written (gate not met).
