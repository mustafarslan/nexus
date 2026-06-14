#!/usr/bin/env python3
"""Gateway/orchestrator e2e arm — exercises NexusAgent.route_and_splice (production path)."""
from __future__ import annotations

import argparse
import os
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
from benchmark_results import file_sha256, write_artifact  # noqa: E402
from benchmark_stats import summarize_arm_stats  # noqa: E402
from bench_routing_accuracy import queries_dataset, load_first_10_tools  # noqa: E402
from nexus_agent import NexusAgent  # noqa: E402
from nexus_retrieval import DEFAULT_EMBED_MODEL, DEFAULT_RERANK_MARGIN, embed_tool_document  # noqa: E402

DEFAULT_MODEL = "/Volumes/AI_SSD/models/Qwen2.5-14B-Instruct-GGUF/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf"


def tool_hit(pred: str | None, gold: str) -> bool:
    if not pred:
        return False
    p = pred.strip().replace("`", "")
    return gold in p or p in gold or p == gold


def parse_args():
    p = argparse.ArgumentParser(description="Gateway orchestrator e2e benchmark.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--atb-workdir", default="results/phaseA_tool_match_work")
    p.add_argument("--query-limit", type=int, default=100)
    p.add_argument("--output", default="results/bench_gateway_e2e.json")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not Path(args.model).exists():
        print(f"Model missing: {args.model}")
        return 1

    tools = load_first_10_tools()
    tool_by_name = {t["name"]: t for t in tools}
    cases = [c for c in queries_dataset if c["tool"] in tool_by_name][: args.query_limit]

    llm_emb = llama_cpp.Llama(model_path=args.embed_model, embedding=True, verbose=False)
    llm = llama_cpp.Llama(
        model_path=args.model, n_ctx=8192, n_batch=2048, n_gpu_layers=999, flash_attn=True, verbose=False,
    )
    llm_rerank = llama_cpp.Llama(
        model_path=args.model, n_ctx=4096, n_batch=512, n_gpu_layers=999, flash_attn=True, verbose=False,
    )

    agent = NexusAgent(llm, llm_emb, base_pos=256, max_splice_pos=256, rerank_margin=DEFAULT_RERANK_MARGIN)
    agent.configure_reranker(llm_rerank=llm_rerank)
    agent.pin_hot_tools(list(range(1, 6)))

    atb_dir = Path(args.atb_workdir)
    for i, tool in enumerate(tools):
        emb = embed_tool_document(llm_emb, tool)
        digest = f"Tool Name: {tool['name']}. Description: {tool.get('description', '')}."
        atb = atb_dir / f"tool_{i}.isolated.atb"
        agent.register_tool(i + 1, tool["name"], emb, digest, str(atb), schema=tool.get("inputSchema"))

    records = []
    for idx, case in enumerate(cases):
        q = case["query"]
        gold = case["tool"]

        t0 = time.perf_counter()
        resolved_id, tool_name, meta = agent.route_with_retrieval(q)
        ttft_us = (time.perf_counter() - t0) * 1e6

        records.append({
            "arm": "GW_route",
            "case_id": idx,
            "gold": gold,
            "pred": tool_name or "",
            "tool_hit": tool_hit(tool_name, gold),
            "retrieval_hit": meta.get("selected") == gold if "selected" in meta else tool_hit(tool_name, gold),
            "ttft_us": ttft_us,
            "decode_us": 0.0,
            "rerank_used": meta.get("rerank_used", False),
            "resolved_id": resolved_id,
        })

        if (idx + 1) % 20 == 0:
            print(f"  {idx + 1}/{len(cases)}")

    by_arm = {"GW_route": summarize_arm_stats(records, "GW_route")}
    out_path = write_artifact(
        args.output,
        "bench_gateway_e2e",
        model_hash=file_sha256(args.model),
        config=vars(args),
        metrics={},
        records=records,
        extra={"by_arm": by_arm},
        smoke=args.smoke,
        force=args.force,
    )
    b = by_arm["GW_route"]
    print(f"Wrote {out_path}")
    print(f"GW_route acc={b['tool_accuracy']:.3f} ttft={b['ttft_us_mean']/1000:.1f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
