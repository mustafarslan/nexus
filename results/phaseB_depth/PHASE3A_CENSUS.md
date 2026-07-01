# Phase 3a — Schema Census & Token-Cost Survey

**Date:** 2026-06-20 · **Scope:** static analysis only — no engine run, no harness change, no IR format proposed.
**Corpus:** the canonical 10 GitHub MCP tools (`test/schemas/github_tools.json`, via `load_first_10_tools()`) — exactly the set the canonical gateway benchmark registers.
**Tokenizer:** production Qwen2.5-14B tokenizer, `vocab_only` (no weights). Serialization matches `nexus_agent._prefix_cache_text_fallback` exactly: `{name, description, inputSchema}`, `sort_keys=True`.
**Artifacts:** `results/phaseB_depth/census_3a.{py,json}`.

---

## 1. Per-tool token mass + features

| tool | raw tok | routing-only | exec-critical | props | required | enum vals | max depth | cond | genuine fb-trigger |
|---|---|---|---|---|---|---|---|---|---|
| create_or_update_file | 270 | 143 | 117 | 7 | 6 | 0 | 2 | 0 | no |
| search_repositories | 157 | 81 | 66 | 3 | 1 | 0 | 2 | 0 | no |
| create_repository | 160 | 83 | 67 | 4 | 1 | 0 | 2 | 0 | no |
| get_file_contents | 179 | 88 | 81 | 4 | 3 | 0 | 2 | 0 | no |
| push_files | 261 | 109 | 142 | 5 | 5 | 0 | **5** | 0 | **yes (depth)** |
| create_issue | 183 | 54 | 119 | 7 | 3 | 0 | 3 | 0 | no |
| create_pull_request | 273 | 146 | 117 | 8 | 5 | 0 | 2 | 0 | no |
| fork_repository | 131 | 51 | 69 | 3 | 2 | 0 | 2 | 0 | no |
| create_branch | 182 | 90 | 82 | 4 | 3 | 0 | 2 | 0 | no |
| list_commits | 139 | 42 | 87 | 5 | 2 | 0 | 2 | 0 | no |

## 2. Corpus-level breakdown

| Metric | Value |
|---|---|
| Tools | 10 |
| **Raw schema token mass (total / mean / range)** | **1935 / 193.5 / 131–273** tok |
| name | 46 tok (2.4%) |
| tool-level description | 385 tok (19.9%) |
| inputSchema (full) | 1403 tok (72.5%) |
| — inputSchema structure (exec-critical) | 947 tok (67.5% of inputSchema) |
| — inputSchema prose (routing-only) | 456 tok (32.5% of inputSchema) |
| JSON wrapper scaffold (residual) | ~101 tok (5.3%) |
| **Routing-only mass** (name + tool-desc + inputSchema prose) | **887 tok = 45.8%** |
| **Execution-critical mass** (inputSchema structure) | **947 tok = 48.9%** |

### Schema feature counts (corpus)

| Feature | Count | Note |
|---|---|---|
| enums / enum values | **0 / 0** | none in corpus |
| conditionals (oneOf/allOf/anyOf/not/if) | **0** | none |
| `$ref` | **0** | none |
| patterns | **0** | none |
| formats | **0** | none |
| `additionalProperties:false` | 11 | present on every tool; benign, 1 token to preserve |
| max nesting depth | 5 (push_files); rest 2–3 | only 1 tool is deep |
| properties / required | 50 / 31 | **required-field density ≈ 62%** |

## 3. Routing-only vs execution-critical (the two-IR split)

- **Routing-only fields (45.8%)** — tool name, tool-level description, and prose inside `inputSchema` (per-field `description`/`title`/`examples`/`default`). On Path B the tool is already selected upstream (`force_tool_id`), so this mass is **not needed to emit arguments** and is the theoretical headroom for an Execution IR.
- **Execution-critical fields (48.9%)** — field names, types, `required`, nesting structure. Must be preserved verbatim to keep argument fidelity. With 62% required-field density, most fields are mandatory — little can be dropped structurally.
- **Fallback triggers** — the census `fallback_trigger` flag reads 10/10, but that is an **artifact**: it fired solely on `additionalProperties:false`, which is not a compression hazard. **Genuine fidelity-blocking features (enums, conditionals, `$ref`, patterns, formats) = 0 across the corpus.** Only `push_files` (depth 5) is a real fallback candidate on the nesting axis. So this corpus is, structurally, *unusually safe* for compaction — there is almost nothing an Execution IR would have to bail out on.

## 4. Blunt verdict: **AMBIGUOUS, leaning LOW headroom for Execution IR**

Reasoning, no hedging:

- **Feasibility is good, magnitude is small.** The fidelity hazards that would block compaction (enums, conditionals, refs, patterns) are absent — an Execution IR would be *safe* here. But the absolute schema mass is tiny: **mean 193 tok/tool, max 273.** The routing-only headroom is ~46%, i.e. **~88 tok/tool** of theoretically droppable prefill. That is a real fraction of the *marginal* deep-path prefill but a small absolute number.
- **Schema is not the deep-path cost driver at this corpus size.** The Path B cost was framed as "schema + query prefill at depth." With ~193-token schemas, the schema is a minor contributor; the deep path's weight is the conversation context and generation, not a 193-token schema. Shrinking it ~46% is unlikely to move TTFT materially — but that is for a latency bridge (3d) to confirm, not assert.
- **Corpus is small and homogeneous.** Ten GitHub tools of near-identical shape. "Raw schema token mass" is highly corpus-dependent: the retired bloat strawman (`test/schemas/bloat/`, ~12.5k tok) would show huge mass, but it is not the production corpus. Generalization of this verdict to heterogeneous/large MCP toolsets is **unproven**.
- **High required density caps structural savings.** 62% of fields are required; the execution-critical 48.9% is mostly non-droppable.

**Net:** an Execution IR is *technically safe* on this corpus but the *headroom is small in absolute latency terms*, and the corpus is too small/uniform to claim a real win. This does **not** justify committing to a Tool IR format. The decision now is whether 3b/3d are worth pursuing at all, or whether the honest conclusion is that deep-path schema-prefill reduction is low-ROI for the canonical corpus and effort should go elsewhere (e.g., Path A hardening, paper positioning).

**Stop here for review before 3b.** No format proposed; no implementation; headroom is explicitly *not* asserted to exist.
