#!/usr/bin/env python3
"""Phase 3a' — bounded REAL-schema census (static analysis only).

Reuses the 3a method exactly. Real published MCP servers only:
  - github_tools.json  (26 tools, official GitHub MCP server)
  - sqlite_tools.json  (6 tools, official MCP SQLite reference server)
Excludes all synthetic/strawman schemas (bloat_metadata, slb_manifest, massive_20_tools,
complex_tool, minimal_tool, mcp_tool_2, manage_email_service_5). No padding/concatenation.
"""
import json, sys, os, statistics as st

MODEL = "/Volumes/AI_SSD/models/Qwen2.5-14B-Instruct-GGUF/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf"
SERVERS = {
    "github_mcp": "test/schemas/github_tools.json",
    "sqlite_mcp": "test/schemas/sqlite_tools.json",
}

from llama_cpp import Llama
llm = Llama(model_path=MODEL, vocab_only=True, verbose=False)
def tok(s): return len(llm.tokenize(s.encode("utf-8"), add_bos=False, special=False))

def pathb(rec):
    return json.dumps({"name": rec["name"], "description": rec.get("description",""),
                       "inputSchema": rec.get("inputSchema") or rec.get("input_schema") or {}}, sort_keys=True)
PROSE={"description","title","examples","example","$comment","default"}
def strip(o):
    if isinstance(o,dict): return {k:strip(v) for k,v in o.items() if k not in PROSE}
    if isinstance(o,list): return [strip(v) for v in o]
    return o
COND={"oneOf","allOf","anyOf","not","if","then","else"}
def walk(o,d=0,f=None):
    if f is None: f={"max_depth":0,"enums":0,"enum_values":0,"conditionals":0,"patterns":0,"formats":0,"refs":0,"addl_false":0}
    if isinstance(o,dict):
        f["max_depth"]=max(f["max_depth"],d)
        if isinstance(o.get("enum"),list): f["enums"]+=1; f["enum_values"]+=len(o["enum"])
        if "pattern" in o: f["patterns"]+=1
        if "format" in o: f["formats"]+=1
        if "$ref" in o: f["refs"]+=1
        if o.get("additionalProperties") is False: f["addl_false"]+=1
        for k in COND:
            if k in o: f["conditionals"]+=1
        for v in o.values(): walk(v,d+1,f)
    elif isinstance(o,list):
        for v in o: walk(v,d+1,f)
    return f

rows=[]
for server,path in SERVERS.items():
    data=json.load(open(path)); tools=data.get("tools",data)
    for rec in tools:
        isch=rec.get("inputSchema") or rec.get("input_schema") or {}
        raw=tok(pathb(rec))
        t_name=tok(json.dumps(rec["name"])); t_desc=tok(json.dumps(rec.get("description","")))
        t_if=tok(json.dumps(isch,sort_keys=True)); t_is=tok(json.dumps(strip(isch),sort_keys=True))
        t_ip=max(0,t_if-t_is)
        props=isch.get("properties") or {}; req=isch.get("required") or []
        f=walk(isch)
        rows.append({"server":server,"name":rec["name"],"raw":raw,
                     "routing_only":t_name+t_desc+t_ip,"exec_critical":t_is,
                     "n_props":len(props),"n_req":len(req),
                     "req_density":round(len(req)/len(props),2) if props else None,
                     "enums":f["enums"],"enum_values":f["enum_values"],"max_depth":f["max_depth"],
                     "conditionals":f["conditionals"],"patterns":f["patterns"],"formats":f["formats"],
                     "refs":f["refs"],"addl_false":f["addl_false"]})

def pct(v,p):
    v=sorted(v); k=(len(v)-1)*p/100; lo=int(k); return round(v[lo]+(v[min(lo+1,len(v)-1)]-v[lo])*(k-lo),1)
raws=[r["raw"] for r in rows]
def grp(rs):
    n=len(rs); R=[r["raw"] for r in rs]
    return {"n_tools":n,"raw_total":sum(R),"raw_mean":round(sum(R)/n,1),
            "raw_median":pct(R,50),"raw_p75":pct(R,75),"raw_p90":pct(R,90),"raw_min":min(R),"raw_max":max(R),
            "routing_only_pct":round(100*sum(r["routing_only"] for r in rs)/sum(R),1),
            "exec_critical_pct":round(100*sum(r["exec_critical"] for r in rs)/sum(R),1),
            "enums":sum(r["enums"] for r in rs),"enum_values":sum(r["enum_values"] for r in rs),
            "conditionals":sum(r["conditionals"] for r in rs),"refs":sum(r["refs"] for r in rs),
            "patterns":sum(r["patterns"] for r in rs),"formats":sum(r["formats"] for r in rs),
            "addl_false":sum(r["addl_false"] for r in rs),"max_depth":max(r["max_depth"] for r in rs),
            "n_props":sum(r["n_props"] for r in rs),"n_req":sum(r["n_req"] for r in rs)}
out={"corpus":grp(rows),"by_server":{s:grp([r for r in rows if r["server"]==s]) for s in SERVERS},
     "per_tool":rows,"excluded_synthetic":["bloat_metadata.json","slb_manifest.json","massive_20_tools.json",
       "complex_tool.json","minimal_tool.json","mcp_tool_2.json","manage_email_service_5.json","list_commits.json(dup)"],
     "note":"Real published MCP servers only. Same method as 3a. No padding/concat."}
os.makedirs("results/phaseB_depth",exist_ok=True)
json.dump(out,open("results/phaseB_depth/census_3a_prime.json","w"),indent=2)

print(f"{'server':12}{'tool':30}{'raw':>5}{'rout':>5}{'exec':>5}{'props':>5}{'req':>4}{'enum':>5}{'dep':>4}{'cond':>4}")
for r in rows:
    print(f"{r['server']:12}{r['name'][:29]:30}{r['raw']:5}{r['routing_only']:5}{r['exec_critical']:5}{r['n_props']:5}{r['n_req']:4}{r['enum_values']:5}{r['max_depth']:4}{r['conditionals']:4}")
for s in SERVERS:
    g=out["by_server"][s]; print(f"\n[{s}] n={g['n_tools']} mean={g['raw_mean']} median={g['raw_median']} p75={g['raw_p75']} p90={g['raw_p90']} max={g['raw_max']} | rout={g['routing_only_pct']}% exec={g['exec_critical_pct']}% | enums={g['enum_values']} cond={g['conditionals']} refs={g['refs']} maxdepth={g['max_depth']}")
c=out["corpus"]
print(f"\n=== CORPUS (real, n={c['n_tools']}) ===")
print(f"raw tok: mean={c['raw_mean']} median={c['raw_median']} p75={c['raw_p75']} p90={c['raw_p90']} min={c['raw_min']} max={c['raw_max']}")
print(f"routing_only={c['routing_only_pct']}%  exec_critical={c['exec_critical_pct']}%")
print(f"features: enums={c['enum_values']} cond={c['conditionals']} refs={c['refs']} patterns={c['patterns']} formats={c['formats']} addlFalse={c['addl_false']} maxdepth={c['max_depth']} reqDensity={round(c['n_req']/c['n_props'],2)}")
