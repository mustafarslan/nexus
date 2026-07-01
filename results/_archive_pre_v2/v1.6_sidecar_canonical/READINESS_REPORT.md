# Semantic IR v1.6 — Readiness Report (NOT PROMOTED)

**Verdict: 3 of 4 gates PASS; gate 2 (arg accuracy) FAILS at 0.854. v1.6 is not promoted.
v1.3 remains canonical.**

V1.6 acted on the v1.5 finding that the exemplar only ever leaked and that inline
descriptions (not the exemplar) dominated IR token count. Changes:
- `src/nexus_agent.py` — `_compress_schema_to_ir`: removed the JSON exemplar entirely (and the
  `_ex_val` helper, `IR_EX_STR`/`IR_EX_DROP`); set `desc_chars` default 50→0 so no
  `/* desc */` comments render. The IR is now a bare typed signature, e.g.
  `create_or_update_file(owner: string, repo: string, path: string, branch: string,
  sha?: string, content: string, message: string)`.
- `test/bench_sidecar_accuracy.py` — replaced Oracle-consistency with a hand-labeled
  `GOLD_ARGS` (30 cases) + `compare_to_gold`: separator/case-insensitive string match
  (`_norm`), exact bool/int match, `<placeholder>` leak detection on values. Mis-routed cases
  (`tool_name != gold`) are recorded but excluded from the accuracy denominator.

Run (once): `DYLD_LIBRARY_PATH=build/external/llama.cpp/src PYTHONPATH=build:src:test
.venv/bin/python test/bench_sidecar_accuracy.py --limit 30 --out-dir
results/v1.6_sidecar_canonical`.

## The 4 promotion gates (measured, n=30)

| # | Gate | Target | Measured | Result |
|---|------|--------|----------|--------|
| 1 | `placeholder_leak_count` | 0 | **0** | ✅ PASS |
| 2 | `specified_arg_accuracy` | ≥ 0.95 | **0.854** (35/41) | ❌ FAIL |
| 3 | `ttft_first_arg_token_ms_hybrid_p50` | < 500 ms | **427.9 ms** | ✅ PASS |
| 4 | `json_valid_rate_hybrid` | ≥ 0.96 | **1.000** | ✅ PASS |

Context: IR shrank **73 → 19 tok p50** (the latency fix — first sub-500 ms TTFT since v1.3),
routing 0.867 (4 mis-routes excluded → 26 routed-ok cases, 41 scored fields), coref 1.0/1.0,
ttft oracle 652 ms (speedup 1.52×).

## Why gate 2 failed — the 6 wrong fields (of 41)

| case | query (abbrev) | field | gold | got | nature |
|---|---|---|---|---|---|
| 2 | "…file script.py to the repo **run-tasks**" | repo | run-tasks | `scripts` | genuine mis-extraction |
| 3 | "…log.txt in repo **server-monitor**" | repo | server-monitor | `logs` | genuine mis-extraction |
| 6 | "…in **project repo**" | repo | project-repo | `project` | partial (gold label debatable) |
| 8 | "…repo **open-source-project**" | repo | open-source-project | `main-repo` | genuine mis-extraction (hallucinated) |
| 16 | "…microservices in go" | query | microservices in go | `microservices language:go` | model added GitHub search syntax (arguably better) |
| 21 | "Create a **public** repo…" | private | False | `None` (omitted) | reasonable: public→omit, per "NO DEFAULTS" |

**Root cause is unchanged from v1.3/v1.5: repo-name slot extraction on underspecified
`create_or_update_file` queries** (cases 2/3/6/8). With no exemplar anchor the model
mis-extracts or hallucinates the repo name (e.g. pulls `scripts` from "script.py", invents
`main-repo`). Removing the exemplar did not regress this — it's the same residual the exemplar
was trying (and failing) to fix.

Two of the six are arguable gold-label issues, not model errors: case 16 (`microservices
language:go` is valid GitHub search syntax) and case 21 (omitting `private` for a *public*
repo is exactly what "NO HALLUCINATED DEFAULTS" asks). But even scoring both as correct gives
**37/41 = 0.902**, still < 0.95 — the four genuine repo errors alone cap the gate.

## Assessment

V1.6 is a clear net improvement: **latency is fixed** (428 ms, the headline goal), JSON
validity is perfect (1.0), leakage is eliminated (the exemplar removal worked), and the IR is
4× smaller. The only failing gate is arg accuracy, and it fails on a single, well-localized
problem — repo-name extraction on a handful of underspecified queries — not on a broad
regression.

Do not promote (gate 2 unmet); v1.3 stays canonical. Targeted next steps:
- Attack repo/owner extraction directly — e.g. a light retrieval/coreference pass that pins
  the entity named after "repo"/"repository" in the query into the `repo` slot, or a 1-line
  positional hint in the IR ("the value after 'repo' in the query is the repo name") that costs
  far fewer tokens than the failed exemplar.
- Tighten the gold labels for cases 16 and 21 (or make the metric credit valid-but-different
  GitHub query syntax and correct optional-field omission) so the gate measures real errors.

Artifact: `results/v1.6_sidecar_canonical/accuracy.json`. No `CANONICAL_RESULTS_MANIFEST.json`
written (gate 2 unmet).
