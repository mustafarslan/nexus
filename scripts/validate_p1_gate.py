#!/usr/bin/env python3
"""Phase 1 acceptance gate: recall@1 >= 0.95 and documents e2e tool-hit."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "test"))

lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../build/external/llama.cpp/src"))
os.environ["LLAMA_CPP_LIB_PATH"] = lib_dir
os.environ["LLAMA_CPP_LIB"] = os.path.join(lib_dir, "libllama.dylib")

import llama_cpp  # noqa: E402
from bench_routing_accuracy import load_first_10_tools, queries_dataset  # noqa: E402
from nexus_retrieval import (  # noqa: E402
    DEFAULT_EMBED_MODEL,
    CrossEncoderReranker,
    embed_query,
    embed_tools,
    margin_gated_retrieve,
    retrieve_hybrid,
)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 1 retrieval gate.")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--recall-target", type=float, default=0.95)
    p.add_argument("--e2e-json", default="results/bench_e2e.json")
    p.add_argument("--output", default="results/phase1_gate.json")
    p.add_argument("--cross-encoder", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    tools = load_first_10_tools()
    tool_by_name = {t["name"]: t for t in tools}
    cases = [c for c in queries_dataset if c["tool"] in tool_by_name]

    llm_emb = llama_cpp.Llama(model_path=args.embed_model, embedding=True, verbose=False)
    tool_embs, tool_names = embed_tools(llm_emb, tools)

    reranker = CrossEncoderReranker() if args.cross_encoder else None
    hits = 0
    rerank_latencies = []
    for case in cases:
        q = case["query"]
        gold = case["tool"]
        q_emb = embed_query(llm_emb, q)
        if reranker:
            topk, picked, lat = margin_gated_retrieve(
                q, q_emb, tool_embs, tool_names, tools,
                k=5, reranker=reranker, tools_by_name=tool_by_name,
            )
            if lat:
                rerank_latencies.append(lat)
            pred = topk[0]
        else:
            pred = retrieve_hybrid(q, q_emb, tool_embs, tool_names, tools, 1)[0]
        if pred == gold:
            hits += 1

    recall = hits / max(len(cases), 1)
    e2e = json.loads(Path(args.e2e_json).read_text()) if Path(args.e2e_json).exists() else {}
    by_arm = e2e.get("extra", {}).get("by_arm", {})
    n1_acc = by_arm.get("N1", {}).get("tool_accuracy")
    b1_acc = by_arm.get("B1", {}).get("tool_accuracy")

    verdict = {
        "recall_at_1": recall,
        "recall_target": args.recall_target,
        "recall_pass": recall >= args.recall_target,
        "n1_tool_hit": n1_acc,
        "b1_tool_hit": b1_acc,
        "e2e_pass": (n1_acc or 0) >= (b1_acc or 0) - 0.01,
        "rerank_p50_us": float(sorted(rerank_latencies)[len(rerank_latencies) // 2]) if rerank_latencies else None,
        "pass": recall >= args.recall_target,
    }
    Path(args.output).write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2))
    return 0 if verdict["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
