# Phase 3a′ — Bounded Real-Schema Census

**Date:** 2026-06-20 · **Scope:** static analysis only, production tokenizer (`vocab_only`), real published MCP servers only. No src/ or harness changes. Method identical to 3a.
**Artifacts:** `results/phaseB_depth/census_3a_prime.{py,json}`.

## Corpus (real, provenance-tagged)
- **github_mcp** — 26 tools, official GitHub MCP server (`test/schemas/github_tools.json`). The canonical 10 are a subset.
- **sqlite_mcp** — 6 tools, official MCP SQLite reference server (`test/schemas/sqlite_tools.json`).
- **Total: 32 real tools / 2 servers.**
- **Excluded as synthetic/strawman:** `bloat_metadata.json`, `slb_manifest.json` (10000 `nexus_*_api_N`), `massive_20_tools.json` (`manage_*_service_N`), `complex_tool.json`, `minimal_tool.json`, `mcp_tool_2.json`, `manage_email_service_5.json`, `list_commits.json` (dup). No padding/concatenation.

## Distribution (raw schema token mass, Path B serialization)

| Group | n | mean | median | p75 | p90 | min | max | routing-only% | exec-critical% |
|---|---|---|---|---|---|---|---|---|---|
| github_mcp | 26 | 180.9 | 158.5 | 194.0 | 271.5 | 100 | **445** | 34.7 | 59.7 |
| sqlite_mcp | 6 | 56.2 | 60.5 | 62.0 | 62.5 | 33 | 63 | 40.9 | 41.2 |
| **Corpus** | **32** | **157.5** | **143.0** | **185.0** | **269.1** | **33** | **445** | **35.2** | **58.5** |

## Schema features (corpus)

| Feature | Count | Where |
|---|---|---|
| enums / enum values | 7 tools / **45 values** | GitHub search/list tools (search_issues 13, list_pull_requests 9, list_issues 8, search_users 5, merge/review 3, update_issue/search_code 2) |
| conditionals (oneOf/anyOf/allOf) | **1** | create_pull_request_review |
| `$ref` | 0 | — |
| patterns / formats | 0 / 0 | — |
| `additionalProperties:false` | 29 | most tools (benign) |
| max nesting depth | **7** | create_pull_request_review |
| required-field density | **0.59** | — |

## What changed vs the canonical 10
The fuller, more representative GitHub set **surfaces structure the 10-tool subset hid**: enums appear (45 values across 7 tools), max depth rises to 7, one conditional appears, and the largest real tool is 445 tok. Crucially, the **droppable routing-only fraction fell** (45.8% on the 10 → 35.2% on the 32) while **fidelity-critical mass rose** (48.9% → 58.5%). The richer tools (search/list/review) carry more enum/structure and less prose — so there is *less* safe-to-drop mass, not more.

## Verdict: **KILL Tool IR (for this class), with a narrow revival trigger**

- **Mass stays small.** Median 143 tok, p90 269, max 445. Even the largest real schema is ~445 tok. Schema is not the deep-path cost driver at this scale.
- **Headroom shrank on the better corpus.** ~35% routing-only ≈ ~50 tok/tool droppable at the median — immaterial to TTFT.
- **Fidelity risk rose.** Enums, depth-7 nesting, and a conditional now present — exactly what an Execution IR must preserve verbatim, narrowing safe compaction further.
- **Decision-rule match:** "real schemas remain small → kill or strongly postpone." They remain small, and the trend is adverse.

**Corpus bias / limits (explicit):** this is 2 developer-tooling servers (GitHub, SQLite), which skew toward compact CLI-style schemas. It does **not** settle the general MCP population. Large enterprise-API servers (Stripe, AWS, Kubernetes, Salesforce) plausibly ship much larger schemas with big enums and deep nesting — the project's own synthetic `bloat/` set (`nexus_stripe_api`, `nexus_aws_ec2_api`, `nexus_kubernetes_api`) hints the authors believe so, but those files are synthetic and inadmissible as evidence.

**Revival trigger (the only condition that reopens Tool IR):** obtain **real, published** schemas from large enterprise-API MCP servers and show **both** (a) large mass (e.g., median ≫ ~800 tok) **and** (b) a substantial droppable routing/prose fraction that survives fidelity constraints. Absent that real evidence, Tool IR is not worth pursuing.
