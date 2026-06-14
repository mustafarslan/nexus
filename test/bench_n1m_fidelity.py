#!/usr/bin/env python3
"""M3: Multi-splice (3 schemas) fidelity vs multi-schema text prefill."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../build/external/llama.cpp/src"))
os.environ["LLAMA_CPP_LIB_PATH"] = lib_dir
os.environ["LLAMA_CPP_LIB"] = os.path.join(lib_dir, "libllama.dylib")

import llama_cpp  # noqa: E402
import nexus_fsm_ext  # noqa: E402
from benchmark_results import file_sha256, write_artifact  # noqa: E402
from bench_routing_accuracy import load_first_10_tools, queries_dataset  # noqa: E402
from nexus_recompute import recompute_fused_multi  # noqa: E402
from nexus_retrieval import DEFAULT_EMBED_MODEL, embed_query, embed_tools, retrieve_hybrid  # noqa: E402

DEFAULT_MODEL = "/Volumes/AI_SSD/models/Qwen2.5-14B-Instruct-GGUF/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf"
P_START = 256


def tokenize(llm, text: str) -> list[int]:
    return [int(t) for t in llm.tokenize(text.encode("utf-8"), add_bos=False, special=False)]


def schema_text(tool: dict) -> str:
    return json.dumps({
        "name": tool["name"],
        "description": tool.get("description") or tool.get("desc", ""),
        "inputSchema": tool.get("inputSchema") or {},
    }, sort_keys=True)


def format_query(q: str) -> str:
    return (
        "You are a tool routing agent. Given the tool schema above, select the tool "
        "that matches the user's intent. You MUST output ONLY the tool name.\n\n"
        f"Query: {q}\nSelected Tool Name:"
    )


def kl_div(ref: np.ndarray, test: np.ndarray) -> float:
    ref = ref.astype(np.float64)
    test = test.astype(np.float64)
    ref = ref - ref.max()
    test = test - test.max()
    ref_p = np.exp(ref) / np.exp(ref).sum()
    test_lp = test - math.log(np.exp(test).sum())
    ref_lp = ref - math.log(np.exp(ref).sum())
    return float(np.sum(ref_p * (ref_lp - test_lp)))


def tool_hit(pred: str, gold: str) -> bool:
    return gold.strip().lower() in pred.strip().lower() or pred.strip().lower() in gold.strip().lower()


def parse_args():
    p = argparse.ArgumentParser(description="N1m multi-splice fidelity validation.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--cases", type=int, default=20)
    p.add_argument("--output", default="results/bench_n1m_fidelity.json")
    p.add_argument("--atb-dir", default="results/phaseA_tool_match_work")
    p.add_argument("--recompute-pct", type=float, default=5.0)
    p.add_argument("--arm", default="fused_multi", choices=("fused_multi",))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    tools = load_first_10_tools()
    tool_by_name = {t["name"]: t for t in tools}
    cases = [c for c in queries_dataset if c["tool"] in tool_by_name][: args.cases]
    if args.smoke:
        cases = cases[:3]

    llm_emb = llama_cpp.Llama(model_path=args.embed_model, embedding=True, verbose=False)
    tool_embs, tool_names = embed_tools(llm_emb, tools)
    llm = llama_cpp.Llama(
        model_path=args.model, n_ctx=8192, n_batch=2048, n_ubatch=2048,
        n_gpu_layers=999, flash_attn=True, verbose=False,
    )
    ctx = llm._ctx.ctx
    cache = nexus_fsm_ext.NexusBlockCache(8 * 1024 * 1024 * 1024)
    preceding = tokenize(llm, "You are a tool-using assistant.\n") + [1] * max(0, P_START - 8)
    preceding = preceding[:P_START]
    schema_tok = {t["name"]: tokenize(llm, schema_text(t)) for t in tools}
    atb_by_name = {t["name"]: Path(args.atb_dir) / f"tool_{i}.isolated.atb" for i, t in enumerate(tools)}

    records = []
    for idx, case in enumerate(cases):
        q = case["query"]
        gold = case["tool"]
        q_emb = embed_query(llm_emb, q)
        top3 = retrieve_hybrid(q, q_emb, tool_embs, tool_names, tools, 3)
        query_tokens = tokenize(llm, format_query(q))
        concat_tokens = []
        for name in top3:
            concat_tokens.extend(schema_tok[name])

        nexus_fsm_ext.clear_kv_cache(ctx)
        if preceding:
            nexus_fsm_ext.decode_tokens(ctx, preceding, 0, 0)
        ref_logits = np.array(
            nexus_fsm_ext.decode_tokens(ctx, concat_tokens + query_tokens, P_START, 0),
            dtype=np.float32,
        )
        ref_text = llm.detokenize([int(np.argmax(ref_logits))]).decode("utf-8", errors="replace")

        def read_kv(layer, p0, p1):
            flat, seq_len, n_head_kv, d_head = nexus_fsm_ext.read_kv_slice(ctx, layer, p0, p1, False)
            return np.array(flat, dtype=np.float32).reshape(seq_len, n_head_kv, d_head)

        nexus_fsm_ext.clear_kv_cache(ctx)
        if preceding:
            nexus_fsm_ext.decode_tokens(ctx, preceding, 0, 0)
        nexus_fsm_ext.decode_tokens(ctx, concat_tokens, P_START, 0)
        true_l1 = read_kv(1, P_START, P_START + len(concat_tokens))

        nexus_fsm_ext.clear_kv_cache(ctx)
        if preceding:
            nexus_fsm_ext.decode_tokens(ctx, preceding, 0, 0)
        offset = 0
        chunks: list[tuple[list[int], int]] = []
        handles = []
        for name in top3:
            toks = schema_tok[name]
            handle = cache.get_or_load(str(atb_by_name[name]))
            handles.append(handle)
            chunks.append((toks, P_START + offset))
            offset += len(toks)
        if args.arm == "fused_multi":
            nexus_fsm_ext.inject_tool_pages_multi(ctx, handles, P_START, 0)
        else:
            for i, name in enumerate(top3):
                nexus_fsm_ext.inject_tool_page(ctx, handles[i], P_START + sum(len(schema_tok[n]) for n in top3[:i]), 0)
        splice_l1 = read_kv(1, P_START, P_START + len(concat_tokens))
        layer1_dev = np.linalg.norm((true_l1 - splice_l1).reshape(len(concat_tokens), -1), axis=1).tolist()
        recompute_fused_multi(ctx, chunks, layer1_dev, args.recompute_pct)
        splice_logits = np.array(
            nexus_fsm_ext.decode_tokens(ctx, query_tokens, P_START + offset, 0),
            dtype=np.float32,
        )
        splice_text = llm.detokenize([int(np.argmax(splice_logits))]).decode("utf-8", errors="replace")

        kl = kl_div(ref_logits, splice_logits)
        records.append({
            "case_id": idx,
            "gold": gold,
            "top3": top3,
            "kl_ref_to_splice": kl,
            "top1_agreement": int(np.argmax(ref_logits)) == int(np.argmax(splice_logits)),
            "ref_tool_hit": tool_hit(ref_text, gold),
            "splice_tool_hit": tool_hit(splice_text, gold),
        })
        print(f"  case {idx}: KL={kl:.4f} ref_hit={records[-1]['ref_tool_hit']} splice_hit={records[-1]['splice_tool_hit']}")

    mean_kl = float(np.mean([r["kl_ref_to_splice"] for r in records])) if records else 999.0
    top1_rate = float(np.mean([1.0 if r["top1_agreement"] else 0.0 for r in records])) if records else 0.0
    splice_hit_rate = float(np.mean([1.0 if r["splice_tool_hit"] else 0.0 for r in records])) if records else 0.0
    arm_valid = mean_kl < 0.1 and top1_rate >= 0.98 and splice_hit_rate >= 0.5

    out_path = write_artifact(
        args.output,
        "bench_n1m_fidelity",
        model_hash=file_sha256(args.model) if Path(args.model).exists() else "",
        config={"n_cases": len(cases), "recompute_pct": args.recompute_pct, "k_schemas": 3},
        metrics={"kl_ref_to_splice": {"unit": "nats", "raw_samples": [r["kl_ref_to_splice"] for r in records]}},
        records=records,
        extra={
            "verdict": {
                "arm_valid": arm_valid,
                "mean_kl": mean_kl,
                "top1_agreement_rate": top1_rate,
                "splice_tool_hit_rate": splice_hit_rate,
                "recommendation": "keep_N1m_arm" if arm_valid else "drop_N1m_arm",
            },
        },
        smoke=args.smoke,
        force=args.force,
    )
    print(f"Wrote {out_path} arm_valid={arm_valid}")
    return 0 if arm_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
