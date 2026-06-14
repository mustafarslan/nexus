#!/usr/bin/env python3
"""Calibrate dense-margin rerank gate from embed top-1 margins on a query set."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "test"))
sys.path.insert(0, str(ROOT / "build"))
lib_dir = ROOT / "build/external/llama.cpp/src"
os.environ["LLAMA_CPP_LIB_PATH"] = str(lib_dir)
os.environ["LLAMA_CPP_LIB"] = str(lib_dir / "libllama.dylib")

import llama_cpp  # noqa: E402
from bench_routing_accuracy import load_first_10_tools, queries_dataset  # noqa: E402
from nexus_retrieval import (  # noqa: E402
    DEFAULT_EMBED_MODEL,
    calibrate_margin_threshold,
    dense_margin,
    embed_query,
    embed_tools,
    retrieve_hybrid,
)

DEFAULT_OUTPUT = "results/margin_calibration.json"


def main() -> int:
    p = argparse.ArgumentParser(description="Calibrate margin gate for cross-encoder router.")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--query-limit", type=int, default=100)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    args = p.parse_args()

    tools = load_first_10_tools()
    tool_names = [t["name"] for t in tools]
    gold_set = set(tool_names)
    cases = [c for c in queries_dataset if c["tool"] in gold_set][: args.query_limit]

    llm_emb = llama_cpp.Llama(model_path=args.embed_model, embedding=True, verbose=False)
    tool_embs, _ = embed_tools(llm_emb, tools)

    margins = []
    embed_hits = []
    for case in cases:
        q_emb = embed_query(llm_emb, case["query"])
        margins.append(dense_margin(q_emb, tool_embs))
        top1 = retrieve_hybrid(case["query"], q_emb, tool_embs, tool_names, tools, 1)[0]
        embed_hits.append(top1 == case["tool"])

    cal = calibrate_margin_threshold(margins, embed_hits)
    roc = []
    for thr in sorted(set(margins)):
        fire = sum(1 for m in margins if m < thr) / len(margins)
        misses = [i for i, h in enumerate(embed_hits) if not h]
        miss_cov = sum(1 for i in misses if margins[i] < thr) / len(misses) if misses else 1.0
        roc.append({"threshold": thr, "fire_rate": fire, "miss_coverage": miss_cov})
    payload = {
        "n_queries": len(cases),
        "margins": {
            "min": min(margins) if margins else 0.0,
            "median": float(sorted(margins)[len(margins) // 2]) if margins else 0.0,
            "max": max(margins) if margins else 0.0,
        },
        "embed_recall_at_1": sum(embed_hits) / len(embed_hits) if embed_hits else 0.0,
        "calibration": cal,
        "roc_curve": roc,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
