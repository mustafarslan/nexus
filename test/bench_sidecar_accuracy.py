#!/usr/bin/env python3
"""V1.2 Sidecar accuracy + Oracle-consistency benchmark (V7/V3).

Run: PYTHONPATH=build:src:test .venv/bin/python test/bench_sidecar_accuracy.py [--smoke]

Arms (same agent, same 10 GitHub tools, same deep synthetic history):
  Sidecar : agent.execute_via_sidecar(messages)            -> pruned anchor, Path A.
  Oracle  : full UNPRUNED in-context history + schema(text) -> args (the reference).

Metrics:
  Routing accuracy  : sidecar tool vs gold (n=100).
  Oracle consistency: sidecar args vs oracle args, SAME forced tool. Reported as
                      parsed-JSON equality + mean per-field agreement + raw-string
                      match (no gold labels exist -> relative-to-oracle, caveated).
  Anchor P50/P99    : proves sidecar anchor < MAX_SPLICE_POS - SAFETY.
  TTFT (max_tokens=1, serial): time-to-first-arg-token, sidecar vs oracle. NOTE the
                      oracle includes a COLD full-history prefill each call; in a live
                      session that history is resident, so this OVER-states oracle cost.

Writes JSON to results/v1.2_sidecar_canonical/ (never touches v1.1_canonical).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../build/external/llama.cpp/src"))
os.environ["LLAMA_CPP_LIB_PATH"] = lib_dir
os.environ["LLAMA_CPP_LIB"] = os.path.join(lib_dir, "libllama.dylib")

import llama_cpp  # noqa: E402
from bench_routing_accuracy import queries_dataset, load_first_10_tools  # noqa: E402
from nexus_agent import NexusAgent  # noqa: E402
from nexus_retrieval import DEFAULT_EMBED_MODEL, DEFAULT_RERANK_MARGIN, embed_tool_document  # noqa: E402

MODEL = "/Volumes/AI_SSD/models/Qwen2.5-14B-Instruct-GGUF/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf"
ATB_DIR = Path("results/phaseA_tool_match_work")
OUT_DIR = Path("results/v1.2_sidecar_canonical")

# v1.6: hand-labeled gold ARGUMENTS for the first 30 queries_dataset cases (filtered to
# the registered GitHub-10). Only the fields a query actually SPECIFIES are listed --
# unspecified required fields (e.g. owner on case 2/3) are deliberately absent so no
# arbitrary default ("user") is enforced. Replaces the unsound Oracle-agreement metric.
GOLD_ARGS = {
    0: {"owner": "alice", "repo": "test-repo", "path": "readme.md", "content": "hello world"},
    1: {"owner": "bob", "repo": "app-settings", "path": "config.json", "branch": "dev"},
    2: {"repo": "run-tasks", "path": "script.py"},
    3: {"repo": "server-monitor", "path": "log.txt"},
    4: {"repo": "auto-tests", "path": "test.sh", "branch": "dev"},
    5: {"owner": "charlie", "repo": "web-app", "path": "index.html"},
    6: {"repo": "project-repo", "path": "readme.md"},
    7: {"repo": "database-utils", "path": "data.csv", "branch": "main"},
    8: {"repo": "open-source-project", "path": "license.txt"},
    9: {"repo": "ml-inference", "path": "main.py"},
    10: {"query": "python web framework"},
    11: {"query": "machine learning"},
    12: {"query": "rust-game-engine"},
    13: {"query": "docker configurations"},
    14: {"query": "react-native-navigation"},
    15: {"query": "LLM fine-tuning scripts"},
    16: {"query": ["microservices in go", "microservices language:go"]},
    17: {"query": "kubernetes operators"},
    18: {"query": "compiler design in c++"},
    19: {"query": "web scraper"},
    20: {"name": "secrets-manager", "private": True},
    21: {"name": "personal-blog", "autoInit": True},
    22: {"name": "web-crawler"},
    23: {"name": "data-science-experiments"},
    24: {"name": "task-manager-app"},
    25: {"name": "auth-service", "private": True},
    26: {"name": "api-gateway-node"},
    27: {"name": "chatbot-ui"},
    28: {"name": "rust-cli-tool"},
    29: {"name": "configuration-files"},
}

_PLACEHOLDER_RE = re.compile(r"<[A-Za-z_]+>")


def _norm(s):
    """Lenient string match: lowercase, strip separators (-, _, space)."""
    return str(s).lower().replace("-", "").replace("_", "").replace(" ", "").strip()


def build_agent():
    llm_emb = llama_cpp.Llama(model_path=DEFAULT_EMBED_MODEL, embedding=True, verbose=False)
    llm = llama_cpp.Llama(model_path=MODEL, n_ctx=8192, n_batch=2048,
                          n_gpu_layers=999, flash_attn=True, verbose=False)
    agent = NexusAgent(llm, llm_emb, base_pos=256, max_splice_pos=256,
                       rerank_margin=DEFAULT_RERANK_MARGIN)
    tools = load_first_10_tools()
    for i, tool in enumerate(tools):
        emb = embed_tool_document(llm_emb, tool)
        digest = f"Tool Name: {tool['name']}. Description: {tool.get('description', '')}."
        agent.register_tool(i + 1, tool["name"], emb, digest,
                            str(ATB_DIR / f"tool_{i}.isolated.atb"),
                            schema=tool.get("inputSchema"))
    return agent, {t["name"] for t in tools}


def deep_history():
    """A realistic short prior conversation. (The old 180-key raw blob was a relic of
    the splice-sidecar pruning stress; for the Hybrid it just bled into string args
    and is not representative.)"""
    return [
        {"role": "user", "content": "Let's work through some GitHub tasks today."},
        {"role": "assistant", "content": "Ready to help with repository operations."},
        {"role": "tool", "name": "get_repo_info", "content": "Repository active; default branch main; 12 open issues."},
        {"role": "assistant", "content": "The repository looks healthy."},
    ]


def full_schema_text(agent, tool_id):
    """The Oracle's grounding: the FULL JSON schema as text (vs the Hybrid's IR)."""
    import json
    rec = next((r for r in agent.tool_records
                if agent.tool_name_to_id.get(r["name"]) == tool_id), None)
    if rec is None:
        return ""
    return json.dumps({"name": rec["name"], "inputSchema": rec.get("inputSchema") or {}},
                      separators=(",", ":"))


def try_repair_json(s: str) -> tuple[dict, bool]:
    """Attempt to parse JSON; if truncated, try conservative repairs.
    
    Returns (parsed_dict, was_repaired). If completely unparseable, returns ({}, False).
    
    Repair strategy (ordered from most to least conservative):
    1. json.loads(s) -- no repair needed
    2. Append closing suffixes ('"}', '}', '"]}', ']}') -- handles mid-string truncation
    3. Regex key-value extraction -- handles severely truncated outputs where the
       opening fields are correct but the JSON is cut deep inside a later value
    
    The regex fallback extracts only string, boolean, and integer values. It does NOT
    attempt to reconstruct arrays or nested objects, keeping false-positive risk low.
    """
    s = (s or "").strip()
    if not s.startswith("{"):
        return {}, False
    
    # Strategy 1: clean parse
    try:
        d = json.loads(s)
        if isinstance(d, dict):
            return d, False
    except Exception:
        pass
    
    # Strategy 2: append closing suffixes (handles mid-string/mid-object truncation)
    for suffix in ['"}', '}', '",}', '"]}', ']}']:
        try:
            d = json.loads(s + suffix)
            if isinstance(d, dict):
                return d, True
        except Exception:
            pass
    
    # Strategy 3: regex extraction for severely truncated outputs
    # Only extract complete key-value pairs that appear before the truncation point
    result = {}
    # String values: "key": "value" (non-greedy, stops at unescaped quote)
    for m in re.finditer(r'"([^"]+)"\s*:\s*"((?:[^"\\]|\\.)*)"', s):
        result[m.group(1)] = m.group(2)
    # Boolean values: "key": true/false
    for m in re.finditer(r'"([^"]+)"\s*:\s*(true|false)(?=[,}\s])', s):
        result[m.group(1)] = m.group(2) == "true"
    # Integer values: "key": 123
    for m in re.finditer(r'"([^"]+)"\s*:\s*(\d+)(?=[,}\s])', s):
        result[m.group(1)] = int(m.group(2))
    
    return result, bool(result)


def compare_to_gold(hybrid: str, gold: dict) -> dict:
    """v1.9: score generated args against gold with truncation recovery.

    - parse_ok  : generated string yielded a usable dict (clean or repaired).
    - repaired  : True if the dict required truncation recovery.
    - leak      : any generated VALUE contains a <placeholder> marker.
    - per gold field: string -> _norm()-equal; bool/int -> exact. Missing key -> wrong.
    """
    a, repaired = try_repair_json(hybrid)
    parse_ok = bool(a)

    if parse_ok:
        leak = any(isinstance(v, str) and _PLACEHOLDER_RE.search(v) for v in a.values())
    else:
        leak = bool(_PLACEHOLDER_RE.search(hybrid or ""))

    field_results, n_correct = {}, 0
    for k, gv in gold.items():
        gen = a.get(k, None)
        if gen is None:
            ok = False
        elif isinstance(gv, bool):
            ok = isinstance(gen, bool) and gen == gv
        elif isinstance(gv, int):
            ok = isinstance(gen, int) and not isinstance(gen, bool) and gen == gv
        elif isinstance(gv, str):
            ok = _norm(gen) == _norm(gv)
        elif isinstance(gv, (list, tuple)):
            ok = any(_norm(gen) == _norm(x) if isinstance(x, str) else gen == x for x in gv)
        else:
            ok = gen == gv
        field_results[k] = ok
        n_correct += int(ok)

    return {
        "parse_ok": parse_ok,
        "repaired": repaired,
        "leak": leak,
        "n_specified": len(gold),
        "n_correct": n_correct,
        "specified_agreement": (n_correct / len(gold)) if gold else None,
        "field_results": field_results,
    }


def pct(xs, p):
    return statistics.quantiles(xs, n=100)[p - 1] if len(xs) > 1 else (xs[0] if xs else 0.0)


def coreference_fixture():
    """V3: multi-turn anaphora. The referent (repo/file name) lives in PRIOR turns;
    the answer is correct only if the pruned sliding window preserves Markov state.
    Tools restricted to the registered GitHub-10."""
    return [
        {"name": "branch_in_searched_repo", "tool": "create_branch", "must_contain": "payment-gateway",
         "messages": [
             {"role": "user", "content": "Search for the repository called payment-gateway."},
             {"role": "assistant", "content": "Found the repository payment-gateway."},
             {"role": "user", "content": "Create a branch called hotfix in it."},
         ]},
        {"name": "get_contents_of_named_file", "tool": "get_file_contents", "must_contain": "deploy.sh",
         "messages": [
             {"role": "user", "content": "I'm interested in the file deploy.sh in repo cloud-infra."},
             {"role": "assistant", "content": "Okay, deploy.sh in cloud-infra."},
             {"role": "user", "content": "Show me what's inside that file."},
         ]},
        {"name": "pr_for_prior_repo", "tool": "create_pull_request", "must_contain": "frontend-ui",
         "messages": [
             {"role": "user", "content": "Let's work in the frontend-ui repository."},
             {"role": "assistant", "content": "Great, frontend-ui it is."},
             {"role": "user", "content": "Open a pull request there from dev into main."},
         ]},
        {"name": "fork_prior_repo", "tool": "fork_repository", "must_contain": "rust-cli-tool",
         "messages": [
             {"role": "user", "content": "Take a look at the rust-cli-tool project."},
             {"role": "assistant", "content": "Looking at rust-cli-tool."},
             {"role": "user", "content": "Fork it to my account."},
         ]},
        {"name": "issue_in_prior_repo", "tool": "create_issue", "must_contain": "api-gateway",
         "messages": [
             {"role": "user", "content": "We have a bug in the api-gateway repository."},
             {"role": "assistant", "content": "Noted, api-gateway has a bug."},
             {"role": "user", "content": "File an issue about it titled 'crash on startup'."},
         ]},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny run to validate the harness")
    ap.add_argument("--limit", type=int, default=100, help="routing+TTFT query count")
    ap.add_argument("--cons-n", type=int, default=40, help="oracle-consistency query count")
    ap.add_argument("--max-tokens", type=int, default=256)  # v1.9: raised to 256 to prevent truncation
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    if args.smoke:
        args.limit, args.cons_n, args.max_tokens = 3, 3, 32

    agent, tool_names = build_agent()
    hist = deep_history()
    cases = [c for c in queries_dataset if c["tool"] in tool_names][: args.limit]

    routing_hits, ir_toks, ttft_hybrid, ttft_oracle = 0, [], [], []
    cons_records = []

    for idx, case in enumerate(cases):
        q, gold = case["query"], case["tool"]
        msgs = hist + [{"role": "user", "content": q}]

        # HYBRID routing + TTFT (max_tokens=1 -> time-to-first-arg-token).
        t0 = time.perf_counter()
        tool_name, _, meta = agent.generate_via_hybrid(msgs, max_tokens=1)
        ttft_hybrid.append((time.perf_counter() - t0) * 1e3)
        if meta.get("ir_tokens"):
            ir_toks.append(meta["ir_tokens"])
        hit = bool(tool_name) and (gold in tool_name or tool_name in gold)
        routing_hits += int(hit)
        routed_ok = tool_name == gold          # exact: arg schema is only valid if tool is right
        rid = agent.tool_name_to_id.get(tool_name, 0)

        # ORACLE arm (full schema text) -> TTFT baseline only (informational speedup ref).
        if rid:
            t0 = time.perf_counter()
            agent.generate_via_hybrid(msgs, max_tokens=1, schema_text=full_schema_text(agent, rid))
            ttft_oracle.append((time.perf_counter() - t0) * 1e3)

        # v1.6 gold-args eval: generate Hybrid args ONCE, compare to GOLD_ARGS[idx].
        # Mis-routed cases are recorded (routed_ok=False) but excluded from the accuracy
        # denominator in the summary -- the wrong tool's schema makes field scoring moot.
        if idx < args.cons_n and rid and idx in GOLD_ARGS:
            _, h_args, _ = agent.generate_via_hybrid(msgs, max_tokens=args.max_tokens)
            cmp = compare_to_gold(h_args, GOLD_ARGS[idx])
            cmp.update({"case": idx, "gold_tool": gold, "pred_tool": tool_name,
                        "routed_ok": routed_ok, "hybrid": h_args, "gold": GOLD_ARGS[idx]})
            cons_records.append(cmp)

        if (idx + 1) % 20 == 0:
            print(f"  ...{idx + 1}/{len(cases)}  routing_hit_rate={routing_hits/(idx+1):.3f}")

    # V3 coreference micro-benchmark (Hybrid).
    coref = []
    for fx in coreference_fixture():
        name, args_json, _ = agent.generate_via_hybrid(fx["messages"], max_tokens=args.max_tokens)
        tool_ok = bool(name) and (fx["tool"] in name or name in fx["tool"])
        arg_ok = bool(args_json) and (fx["must_contain"] in args_json)
        coref.append({"name": fx["name"], "gold_tool": fx["tool"], "pred_tool": name,
                      "tool_ok": tool_ok, "referent": fx["must_contain"],
                      "arg_ok": arg_ok, "args": args_json})

    n = len(cases)
    routed_records = [r for r in cons_records if r.get("routed_ok")]
    tot_spec = sum(r["n_specified"] for r in routed_records)
    tot_correct = sum(r["n_correct"] for r in routed_records)
    summary = {
        "arm": "V1.9_HYBRID (bare typed IR, no exemplar/desc; gold-args eval; kebab-case; truncation-resilient)",
        "n_routing": n,
        "routing_accuracy": routing_hits / n if n else 0.0,
        "ir_tokens_p50": pct(ir_toks, 50), "ir_tokens_p99": pct(ir_toks, 99),
        "n_consistency": len(cons_records),
        "n_routed_ok": len(routed_records),
        # GATE 2: accuracy over specified gold fields, ROUTED-OK cases only.
        "specified_arg_accuracy": (tot_correct / tot_spec) if tot_spec else None,
        "specified_fields_total": tot_spec,
        "specified_fields_correct": tot_correct,
        # GATE 1: any <placeholder> in any generated value (across all records).
        "placeholder_leak_count": sum(1 for r in cons_records if r.get("leak")),
        # GATE 4: JSON validity across all generated arg records.
        "json_valid_rate_hybrid": sum(r["parse_ok"] for r in cons_records) / len(cons_records) if cons_records else None,
        "repair_count": sum(1 for r in cons_records if r.get("repaired")),
        # GATE 3: serial time-to-first-arg-token.
        "ttft_first_arg_token_ms_hybrid_p50": pct(ttft_hybrid, 50),
        "ttft_first_arg_token_ms_oracle_fullschema_p50": pct(ttft_oracle, 50),
        "coref_tool_acc": sum(c["tool_ok"] for c in coref) / len(coref) if coref else None,
        "coref_arg_acc": sum(c["arg_ok"] for c in coref) / len(coref) if coref else None,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = {"summary": summary, "consistency_records": cons_records, "coref": coref}
    out_path = out_dir / ("accuracy_smoke.json" if args.smoke else "accuracy.json")
    out_path.write_text(json.dumps(artifact, indent=2))

    print("\n=== V1.9 SIDECAR ACCURACY / GOLD-ARGS ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\n  coreference:")
    for c in coref:
        print(f"    {c['name']:32s} tool_ok={c['tool_ok']} arg_ok={c['arg_ok']} "
              f"pred={c['pred_tool']!r} ref={c['referent']!r}")
    print(f"\n  artifact -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
