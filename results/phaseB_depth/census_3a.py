#!/usr/bin/env python3
"""Phase 3a — schema census & token-cost survey (static analysis ONLY).

No engine run, no harness change, no optimization. Loads the canonical 10-tool
corpus and tokenizes each schema with the PRODUCTION tokenizer (Qwen2.5, vocab_only),
serialized exactly as Path B prefills it (nexus_agent._prefix_cache_text_fallback).
Reports per-tool + corpus token mass by category and schema-feature counts.
"""
import json, sys, os

MODEL = "/Volumes/AI_SSD/models/Qwen2.5-14B-Instruct-GGUF/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf"
sys.path.insert(0, "test")
from bench_routing_accuracy import load_first_10_tools

from llama_cpp import Llama
llm = Llama(model_path=MODEL, vocab_only=True, verbose=False)

def tok(s: str) -> int:
    return len(llm.tokenize(s.encode("utf-8"), add_bos=False, special=False))

# ---- Path B serialization (must match _prefix_cache_text_fallback exactly) ----
def pathb_schema_str(rec):
    return json.dumps({
        "name": rec["name"],
        "description": rec.get("description", ""),
        "inputSchema": rec.get("inputSchema") or rec.get("input_schema") or {},
    }, sort_keys=True)

# ---- strip routing-only prose from a schema object (keep execution structure) ----
PROSE_KEYS = {"description", "title", "examples", "example", "$comment", "default"}
def strip_prose(obj):
    if isinstance(obj, dict):
        return {k: strip_prose(v) for k, v in obj.items() if k not in PROSE_KEYS}
    if isinstance(obj, list):
        return [strip_prose(v) for v in obj]
    return obj

# ---- schema feature walk ----
COND_KEYS = {"oneOf", "allOf", "anyOf", "not", "if", "then", "else"}
def walk(obj, depth=0, feat=None):
    if feat is None:
        feat = {"max_depth":0,"enums":0,"enum_values":0,"conditionals":0,
                "patterns":0,"formats":0,"refs":0,"addl_false":0}
    if isinstance(obj, dict):
        feat["max_depth"] = max(feat["max_depth"], depth)
        if "enum" in obj and isinstance(obj["enum"], list):
            feat["enums"] += 1; feat["enum_values"] += len(obj["enum"])
        if "pattern" in obj: feat["patterns"] += 1
        if "format" in obj: feat["formats"] += 1
        if "$ref" in obj: feat["refs"] += 1
        if obj.get("additionalProperties") is False: feat["addl_false"] += 1
        for k in COND_KEYS:
            if k in obj: feat["conditionals"] += 1
        for v in obj.values():
            walk(v, depth+1, feat)
    elif isinstance(obj, list):
        for v in obj:
            walk(v, depth+1, feat)
    return feat

tools = load_first_10_tools()
rows = []
for rec in tools:
    isch = rec.get("inputSchema") or rec.get("input_schema") or {}
    name_s = json.dumps(rec["name"])
    desc_s = json.dumps(rec.get("description", ""))
    isch_full_s = json.dumps(isch, sort_keys=True)
    isch_stripped_s = json.dumps(strip_prose(isch), sort_keys=True)

    raw_total = tok(pathb_schema_str(rec))
    t_name = tok(name_s)
    t_desc = tok(desc_s)
    t_isch_full = tok(isch_full_s)
    t_isch_struct = tok(isch_stripped_s)          # execution structure (prose removed)
    t_isch_prose = max(0, t_isch_full - t_isch_struct)

    props = (isch.get("properties") or {})
    n_props = len(props)
    req = isch.get("required") or []
    n_req = len(req)
    feat = walk(isch)

    routing_only = t_name + t_desc + t_isch_prose           # not needed to emit args (tool already selected)
    exec_critical = t_isch_struct                            # field names/types/enums/required/nesting
    fallback_trigger = (feat["conditionals"]>0 or feat["refs"]>0 or feat["patterns"]>0
                        or feat["max_depth"]>=4 or feat["enum_values"]>=12 or feat["addl_false"]>0)

    rows.append({
        "name": rec["name"], "raw_total_tok": raw_total,
        "name_tok": t_name, "tool_desc_tok": t_desc,
        "isch_full_tok": t_isch_full, "isch_struct_tok": t_isch_struct, "isch_prose_tok": t_isch_prose,
        "routing_only_tok": routing_only, "exec_critical_tok": exec_critical,
        "n_props": n_props, "n_required": n_req,
        "req_density": round(n_req/n_props,2) if n_props else None,
        "enums": feat["enums"], "enum_values": feat["enum_values"],
        "max_nest_depth": feat["max_depth"], "conditionals": feat["conditionals"],
        "patterns": feat["patterns"], "formats": feat["formats"],
        "refs": feat["refs"], "addl_false": feat["addl_false"],
        "fallback_trigger": fallback_trigger,
    })

def s(k): return sum(r[k] for r in rows)
n = len(rows)
corpus = {
    "n_tools": n,
    "raw_total_tok": s("raw_total_tok"),
    "raw_total_tok_mean": round(s("raw_total_tok")/n,1),
    "name_tok": s("name_tok"), "tool_desc_tok": s("tool_desc_tok"),
    "isch_full_tok": s("isch_full_tok"), "isch_struct_tok": s("isch_struct_tok"),
    "isch_prose_tok": s("isch_prose_tok"),
    "routing_only_tok": s("routing_only_tok"), "exec_critical_tok": s("exec_critical_tok"),
    "routing_only_pct": round(100*s("routing_only_tok")/s("raw_total_tok"),1),
    "exec_critical_pct": round(100*s("exec_critical_tok")/s("raw_total_tok"),1),
    "enums": s("enums"), "enum_values": s("enum_values"),
    "conditionals": s("conditionals"), "patterns": s("patterns"),
    "formats": s("formats"), "refs": s("refs"), "addl_false": s("addl_false"),
    "max_nest_depth": max(r["max_nest_depth"] for r in rows),
    "n_props": s("n_props"), "n_required": s("n_required"),
    "fallback_trigger_tools": sum(1 for r in rows if r["fallback_trigger"]),
}

out = {"corpus": corpus, "per_tool": rows,
       "note": "Static census. routing_only = name+tool_desc+inputSchema prose (tool already "
               "selected upstream on Path B). exec_critical = inputSchema structure (names/types/"
               "enums/required/nesting). Component token sums are independent-substring estimates "
               "(boundary effects small); raw_total_tok is exact for the Path B serialization."}
os.makedirs("results/phaseB_depth", exist_ok=True)
with open("results/phaseB_depth/census_3a.json","w") as f:
    json.dump(out, f, indent=2)

print(f"{'tool':28} {'raw':>5} {'rout':>5} {'exec':>5} {'props':>5} {'req':>4} {'enum':>4} {'depth':>5} {'cond':>4} {'fb':>3}")
for r in rows:
    print(f"{r['name']:28} {r['raw_total_tok']:5} {r['routing_only_tok']:5} {r['exec_critical_tok']:5} "
          f"{r['n_props']:5} {r['n_required']:4} {r['enum_values']:4} {r['max_nest_depth']:5} "
          f"{r['conditionals']:4} {'Y' if r['fallback_trigger'] else '-':>3}")
print("-"*90)
c=corpus
print(f"{'CORPUS('+str(n)+')':28} {c['raw_total_tok']:5} {c['routing_only_tok']:5} {c['exec_critical_tok']:5} "
      f"{c['n_props']:5} {c['n_required']:4} {c['enum_values']:4} {c['max_nest_depth']:5} {c['conditionals']:4} "
      f"{c['fallback_trigger_tools']:3}")
print(f"\nrouting_only = {c['routing_only_pct']}%  exec_critical = {c['exec_critical_pct']}%  of raw schema mass")
print(f"mean raw schema = {c['raw_total_tok_mean']} tok/tool")
