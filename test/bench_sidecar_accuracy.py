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

from bench_routing_accuracy import queries_dataset, load_n_tools, compile_atb_files  # noqa: E402

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


def build_agent_with_size(llm, llm_emb, size):
    agent = NexusAgent(llm, llm_emb, base_pos=256, max_splice_pos=256,
                       rerank_margin=DEFAULT_RERANK_MARGIN)
    # Using load_n_tools imported from bench_routing_accuracy
    tools = load_n_tools(size)
    compile_atb_files(tools, MODEL)
    
    for i, tool in enumerate(tools):
        emb = embed_tool_document(llm_emb, tool)
        digest = f"Tool Name: {tool['name']}. Description: {tool.get('desc', '')}."
        agent.register_tool(i + 1, tool["name"], emb, digest,
                            tool["atb_path"],
                            schema=tool.get("inputSchema") or tool.get("input_schema"))
    return agent, {t["name"] for t in tools}, tools


def deep_history():
    return [
        {"role": "user", "content": "Let's work through some GitHub tasks today."},
        {"role": "assistant", "content": "Ready to help with repository operations."},
        {"role": "tool", "name": "get_repo_info", "content": "Repository active; default branch main; 12 open issues."},
        {"role": "assistant", "content": "The repository looks healthy."},
    ]


def full_schema_text(agent, tool_id):
    import json
    rec = next((r for r in agent.tool_records
                if agent.tool_name_to_id.get(r["name"]) == tool_id), None)
    if rec is None:
        return ""
    return json.dumps({"name": rec["name"], "inputSchema": rec.get("inputSchema") or {}},
                      separators=(",", ":"))


def try_repair_json(s: str) -> tuple[dict, bool]:
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
    
    # Strategy 2: append closing suffixes
    for suffix in ['"}', '}', '",}', '"]}', ']}']:
        try:
            d = json.loads(s + suffix)
            if isinstance(d, dict):
                return d, True
        except Exception:
            pass
    
    # Strategy 3: regex extraction
    result = {}
    for m in re.finditer(r'"([^"]+)"\s*:\s*"((?:[^"\\]|\\.)*)"', s):
        result[m.group(1)] = m.group(2)
    for m in re.finditer(r'"([^"]+)"\s*:\s*(true|false)(?=[,}\s])', s):
        result[m.group(1)] = m.group(2) == "true"
    for m in re.finditer(r'"([^"]+)"\s*:\s*(\d+)(?=[,}\s])', s):
        result[m.group(1)] = int(m.group(2))
    
    return result, bool(result)


def compare_to_gold(hybrid: str, gold: dict) -> dict:
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
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--tool-sizes", default="10", help="Space-separated list of tool registry sizes, e.g. '10 20 26'")
    ap.add_argument("--runs", type=int, default=1, help="Number of trials per tool size")
    args = ap.parse_args()
    
    if args.smoke:
        args.limit, args.cons_n, args.max_tokens = 3, 3, 32
        
    sizes = [int(s) for s in args.tool_sizes.split()]
    runs = args.runs
    
    import shutil
    print(f"Pre-compiling all tools up to size {max(sizes)} on GPU...")
    max_tools = load_n_tools(max(sizes))
    compile_atb_files(max_tools, MODEL)
    print("Pre-compilation finished successfully.")
        
    print(f"Loading models once for sidecar benchmark...")
    global llama_cpp, NexusAgent, DEFAULT_EMBED_MODEL, DEFAULT_RERANK_MARGIN, embed_tool_document
    import llama_cpp
    from nexus_agent import NexusAgent
    from nexus_retrieval import DEFAULT_EMBED_MODEL, DEFAULT_RERANK_MARGIN, embed_tool_document
    llm_emb = llama_cpp.Llama(model_path=DEFAULT_EMBED_MODEL, embedding=True, verbose=False)
    llm = llama_cpp.Llama(model_path=MODEL, n_ctx=8192, n_batch=2048,
                          n_gpu_layers=999, flash_attn=True, verbose=False)
                          
    overall_results = {}
    hist = deep_history()
    
    for size in sizes:
        print(f"\n======================================================================")
        print(f"Evaluating Sidecar Generation Accuracy at N = {size}...")
        print(f"======================================================================")
        
        runs_routing_acc = []
        runs_arg_acc_routed = []
        runs_arg_acc_e2e = []
        runs_placeholder_leaks = []
        runs_json_valid = []
        runs_ttft_hybrid = []
        runs_ttft_oracle = []
        runs_token_savings = []
        last_ir_toks = []
        last_counts = {}

        for run_idx in range(runs):
            print(f"  Run {run_idx + 1}/{runs}...")
            agent, tool_names, tools = build_agent_with_size(llm, llm_emb, size)
            cases = [c for c in queries_dataset if c["tool"] in tool_names][: args.limit]
            
            routing_hits, ir_toks, ttft_hybrid, ttft_oracle = 0, [], [], []
            cons_records = []
            
            for idx, case in enumerate(cases):
                q, gold = case["query"], case["tool"]
                msgs = hist + [{"role": "user", "content": q}]

                # HYBRID routing + TTFT (max_tokens=1)
                t0 = time.perf_counter()
                tool_name, _, meta = agent.generate_via_hybrid(msgs, max_tokens=1)
                ttft_hybrid.append((time.perf_counter() - t0) * 1e3)
                if meta.get("ir_tokens"):
                    ir_toks.append(meta["ir_tokens"])
                hit = bool(tool_name) and (gold in tool_name or tool_name in gold)
                routing_hits += int(hit)
                routed_ok = tool_name == gold
                rid = agent.tool_name_to_id.get(tool_name, 0)

                # ORACLE arm (full schema text) -> TTFT baseline
                if rid:
                    t0 = time.perf_counter()
                    agent.generate_via_hybrid(msgs, max_tokens=1, schema_text=full_schema_text(agent, rid))
                    ttft_oracle.append((time.perf_counter() - t0) * 1e3)

                # gold-args eval
                if idx < args.cons_n and rid and idx in GOLD_ARGS:
                    _, h_args, _ = agent.generate_via_hybrid(msgs, max_tokens=args.max_tokens)
                    cmp = compare_to_gold(h_args, GOLD_ARGS[idx])
                    cmp.update({"case": idx, "gold_tool": gold, "pred_tool": tool_name,
                                "routed_ok": routed_ok, "hybrid": h_args, "gold": GOLD_ARGS[idx]})
                    cons_records.append(cmp)
                    
            # Compute token compression ratio dynamically for this registry
            raw_lens, ir_lens = [], []
            for tool in tools:
                raw_schema = json.dumps({"name": tool["name"], "inputSchema": tool.get("inputSchema") or tool.get("input_schema") or {}}, separators=(",", ":"))
                raw_tok_len = len(llm.tokenize(raw_schema.encode("utf-8"), add_bos=False, special=False))
                raw_lens.append(raw_tok_len)
                
                sig = agent._compress_schema_to_ir(tool)
                ir_tok_len = len(llm.tokenize(sig.encode("utf-8"), add_bos=False, special=False))
                ir_lens.append(ir_tok_len)
            avg_raw = statistics.mean(raw_lens) if raw_lens else 1.0
            avg_ir = statistics.mean(ir_lens) if ir_lens else 1.0
            token_savings = (1.0 - (avg_ir / avg_raw)) * 100.0
            runs_token_savings.append(token_savings)
            
            # Calculate accuracies
            n = len(cases)
            routed_records = [r for r in cons_records if r.get("routed_ok")]
            
            # 1. Routed-Only Parameter Accuracy
            tot_spec = sum(r["n_specified"] for r in routed_records)
            tot_correct = sum(r["n_correct"] for r in routed_records)
            arg_acc_routed = (tot_correct / tot_spec) * 100.0 if tot_spec else 0.0
            
            # 2. End-to-End (E2E) Parameter Accuracy
            tot_spec_e2e = sum(r["n_specified"] for r in cons_records)
            tot_correct_e2e = sum(r["n_correct"] if r.get("routed_ok") else 0 for r in cons_records)
            arg_acc_e2e = (tot_correct_e2e / tot_spec_e2e) * 100.0 if tot_spec_e2e else 0.0

            # Exact integer counts for honest Wilson CIs (case-level and argument-level denominators)
            last_counts = {
                "n_cases": n, "routing_hits": routing_hits, "n_cons_records": len(cons_records),
                "n_routed_records": len(routed_records),
                "tot_spec_routed": tot_spec, "tot_correct_routed": tot_correct,
                "tot_spec_e2e": tot_spec_e2e, "tot_correct_e2e": tot_correct_e2e,
            }
            
            runs_routing_acc.append((routing_hits / n) * 100.0 if n else 0.0)
            runs_arg_acc_routed.append(arg_acc_routed)
            runs_arg_acc_e2e.append(arg_acc_e2e)
            runs_placeholder_leaks.append(sum(1 for r in cons_records if r.get("leak")))
            runs_json_valid.append((sum(r["parse_ok"] for r in cons_records) / len(cons_records) * 100.0) if cons_records else 0.0)
            runs_ttft_hybrid.append(pct(ttft_hybrid, 50))
            runs_ttft_oracle.append(pct(ttft_oracle, 50))
            last_ir_toks = ir_toks
                
        # Aggregate runs
        def stats(vals):
            if not vals:
                return 0.0, 0.0
            return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)
            
        m_routing, s_routing = stats(runs_routing_acc)
        m_arg_routed, s_arg_routed = stats(runs_arg_acc_routed)
        m_arg_e2e, s_arg_e2e = stats(runs_arg_acc_e2e)
        m_leak, _ = stats(runs_placeholder_leaks)
        m_valid, s_valid = stats(runs_json_valid)
        m_ttft_h, _ = stats(runs_ttft_hybrid)
        m_ttft_o, _ = stats(runs_ttft_oracle)
        m_savings, _ = stats(runs_token_savings)
        
        _ir_all = [t for t in (last_ir_toks or []) if t]
        overall_results[size] = {
            "routing_accuracy": m_routing, "routing_accuracy_std": s_routing,
            "arg_accuracy_routed_only": m_arg_routed, "arg_accuracy_routed_only_std": s_arg_routed,
            "arg_accuracy_e2e": m_arg_e2e, "arg_accuracy_e2e_std": s_arg_e2e,
            "placeholder_leak_count": m_leak,
            "json_valid_rate_hybrid": m_valid, "json_valid_rate_hybrid_std": s_valid,
            "ir_tokens_p50": (statistics.median(_ir_all) if _ir_all else 0.0),
            "ir_tokens_p99": (pct(_ir_all, 99) if _ir_all else 0.0),
            "ttft_hybrid_p50_ms": m_ttft_h,
            "ttft_oracle_p50_ms": m_ttft_o,
            "token_savings_pct": m_savings,
            "counts": last_counts,
        }
        
    # Write summary artifact
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ("accuracy_smoke.json" if args.smoke else "accuracy.json")
    out_path.write_text(json.dumps(overall_results, indent=2))
    
    print("\n" + "="*80)
    print("                      SIDECAR COMPARATIVE SCALING CURVE")
    print("="*80)
    print(f"{'Size (N)':<8} | {'Route Acc':<12} | {'Arg (Routed)':<14} | {'Arg (E2E)':<12} | {'TTFT (H/O)':<16} | {'Token Save':<10}")
    print("-"*80)
    for size in sizes:
        res = overall_results[size]
        route_str = f"{res['routing_accuracy']:.1f}%"
        arg_r_str = f"{res['arg_accuracy_routed_only']:.1f}%"
        arg_e_str = f"{res['arg_accuracy_e2e']:.1f}%"
        ttft_str = f"{res['ttft_hybrid_p50_ms']:.0f}/{res['ttft_oracle_p50_ms']:.0f} ms"
        tok_str = f"{res['token_savings_pct']:.1f}%"
        print(f"{size:<8} | {route_str:<12} | {arg_r_str:<14} | {arg_e_str:<12} | {ttft_str:<16} | {tok_str:<10}")
    print("="*80)
    print(f"Wrote JSON artifact: {out_path}")
    
    if os.path.exists("test/schemas/jit_bloat"):
        import shutil
        shutil.rmtree("test/schemas/jit_bloat")
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
