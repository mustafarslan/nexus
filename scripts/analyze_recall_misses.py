#!/usr/bin/env python3
"""M3: Analyze retrieval misses on hold-out queries_dataset with calibrated CE gating."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../test")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../build/external/llama.cpp/src"))
os.environ["LLAMA_CPP_LIB_PATH"] = lib_dir
os.environ["LLAMA_CPP_LIB"] = os.path.join(lib_dir, "libllama.dylib")

import llama_cpp  # noqa: E402
from bench_routing_accuracy import load_first_10_tools, queries_dataset  # noqa: E402
from nexus_retrieval import (  # noqa: E402
    DEFAULT_EMBED_MODEL,
    DEFAULT_RERANK_MARGIN,
    CrossEncoderReranker,
    dense_margin,
    embed_query,
    embed_tools,
    rerank_with_llm,
    retrieve_hybrid,
    route_tool,
)

DEFAULT_MODEL = "/Volumes/AI_SSD/models/Qwen2.5-14B-Instruct-GGUF/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf"
DEFAULT_CE_V3 = "results/tool_cross_encoder_finetuned_v3"


def _default_margin_threshold() -> float:
    try:
        from nexus_calibration import CALIBRATED_MARGIN_THRESHOLD
        return float(CALIBRATED_MARGIN_THRESHOLD)
    except ImportError:
        return DEFAULT_RERANK_MARGIN


def parse_args():
    p = argparse.ArgumentParser(description="Hold-out retrieval miss analysis + M3 CE gating.")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output", default="results/recall_miss_analysis_v2.json")
    p.add_argument("--rerank", action="store_true")
    p.add_argument("--cross-encoder", default=DEFAULT_CE_V3)
    p.add_argument("--margin-threshold", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    margin_threshold = args.margin_threshold if args.margin_threshold is not None else _default_margin_threshold()

    ce_path = Path(args.cross_encoder)
    if not ce_path.is_dir():
        raise RuntimeError(
            f"FATAL: CrossEncoder not found at {ce_path}. "
            "Run: python scripts/train_cross_encoder.py --output results/tool_cross_encoder_finetuned_v3"
        )

    tools = load_first_10_tools()
    tool_by_name = {t["name"]: t for t in tools}
    cases = [c for c in queries_dataset if c["tool"] in tool_by_name]

    llm_emb = llama_cpp.Llama(model_path=args.embed_model, embedding=True, verbose=False)
    tool_embs, tool_names = embed_tools(llm_emb, tools)

    llm = None
    if args.rerank and Path(args.model).exists():
        llm = llama_cpp.Llama(model_path=args.model, n_ctx=4096, n_gpu_layers=999, verbose=False)

    ce_reranker = CrossEncoderReranker(str(ce_path))
    if not ce_reranker.available:
        raise RuntimeError(f"FATAL: CrossEncoder failed to load from {ce_path}")

    misses, hits = [], []
    rerank_hits = []
    ce_hits = []
    ce_rerank_used = 0

    for case in cases:
        q = case["query"]
        gold = case["tool"]
        q_emb = embed_query(llm_emb, q)
        margin = dense_margin(q_emb, tool_embs)
        top1 = retrieve_hybrid(q, q_emb, tool_embs, tool_names, tools, 1)[0]
        top5 = retrieve_hybrid(q, q_emb, tool_embs, tool_names, tools, 5)
        row = {
            "query": q,
            "gold": gold,
            "top1": top1,
            "top5": top5,
            "hit@1": top1 == gold,
            "hit@5": gold in top5,
            "dense_margin": margin,
        }

        pred_ce, used_ce, _, _, _ = route_tool(
            q, q_emb, tool_embs, tool_names, tools, tool_by_name,
            margin_threshold=margin_threshold,
            cross_encoder=ce_reranker,
            rerank_mode="cross_encoder",
        )
        row["ce_top1"] = pred_ce
        row["ce_hit@1"] = pred_ce == gold
        row["ce_rerank_used"] = used_ce
        ce_hits.append(row["ce_hit@1"])
        if used_ce:
            ce_rerank_used += 1

        if top1 == gold:
            hits.append(row)
        else:
            misses.append(row)
        if llm is not None:
            pred = rerank_with_llm(llm, q, top5, tool_by_name)
            rerank_hits.append(pred == gold)

    hits_at_5 = sum(
        1 for case in cases
        if case["tool"] in retrieve_hybrid(
            case["query"],
            embed_query(llm_emb, case["query"]),
            tool_embs,
            tool_names,
            tools,
            5,
        )
    )
    ce_recall = float(np.mean(ce_hits)) if ce_hits else None
    summary = {
        "n_queries": len(cases),
        "recall_at_1": len(hits) / len(cases),
        "recall_at_5": hits_at_5 / len(cases),
        "ce_recall_at_1": ce_recall,
        "ce_rerank_fraction": ce_rerank_used / len(cases) if cases else None,
        "margin_threshold": margin_threshold,
        "n_misses": len(misses),
        "misses": misses,
        "rerank_recall_at_1": float(np.mean(rerank_hits)) if rerank_hits else None,
        "target_recall_at_1": 0.90,
        "meets_target": ce_recall is not None and ce_recall >= 0.90,
        "honest_ceiling_note": (
            "M3: P20 calibrated on synthetic queries; ambiguous hard-pair clamp; "
            "CE v3 trained on synthetic paraphrases only; hold-out queries_dataset for eval."
        ),
    }

    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "misses"}, indent=2))
    print(f"Wrote {args.output} ({len(misses)} misses)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
