#!/usr/bin/env python3
"""Dynamic-context benchmark: splice vs prefix-cache under varying multi-session histories.

Thesis: position-independent KV splice sustains O(1) tool inject cost while prefix-cache
hit rate collapses as concurrent session count S grows.
"""
from __future__ import annotations

import argparse
import json
import os
import random
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
import nexus_fsm_ext  # noqa: E402
from bench_e2e import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_SYSTEM,
    SchemaPrefixCache,
    cache_preceding,
    compile_atbs,
    format_query,
    nexus_splice_recompute,
    tokenize,
)
from bench_routing_accuracy import load_first_10_tools, queries_dataset  # noqa: E402
from benchmark_results import write_artifact  # noqa: E402
from benchmark_stats import summarize_arm_stats  # noqa: E402
from nexus_retrieval import DEFAULT_EMBED_MODEL, embed_query, embed_tools, retrieve_hybrid, schema_text  # noqa: E402


def session_history(llm, session_id: int, min_len: int, max_len: int, rng: random.Random) -> list[int]:
    """Unique multi-turn history per session (prefix-cache cannot share across sessions)."""
    turns = [
        f"User (session {session_id}): debugging auth flow attempt {rng.randint(1, 99)}.",
        f"Assistant: check token expiry and refresh rotation policy for session {session_id}.",
        f"User: pipeline stage {rng.randint(1, 12)} failed with exit code {rng.randint(1, 9)}.",
        f"Assistant: inspect logs and retry with exponential backoff (session {session_id}).",
    ]
    target = rng.randint(min_len, max_len)
    text = DEFAULT_SYSTEM + "\n"
    i = 0
    toks = tokenize(llm, text)
    while len(toks) < target:
        text += turns[i % len(turns)] + "\n"
        i += 1
        toks = tokenize(llm, text)
    return toks[:target]


def estimate_prefix_cache_bytes(seq_len: int, n_layers: int, n_head_kv: int, d_head: int) -> int:
    per_layer = 2 * seq_len * n_head_kv * d_head * 2  # K+V fp16
    return per_layer * n_layers


def main() -> int:
    p = argparse.ArgumentParser(description="Dynamic multi-session context benchmark.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--sessions", default="1,8,32", help="Comma-separated concurrent session counts.")
    p.add_argument("--query-limit", type=int, default=50)
    p.add_argument("--history-min", type=int, default=128)
    p.add_argument("--history-max", type=int, default=512)
    p.add_argument("--output", default="results/bench_dynamic_context.json")
    p.add_argument("--atb-workdir", default="results/phaseA_tool_match_work")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    session_counts = [int(x) for x in args.sessions.split(",") if x.strip()]
    tools = load_first_10_tools()
    tool_by_name = {t["name"]: t for t in tools}
    cases = [c for c in queries_dataset if c["tool"] in tool_by_name][: args.query_limit]

    llm_emb = llama_cpp.Llama(model_path=args.embed_model, embedding=True, verbose=False)
    tool_embs, tool_names = embed_tools(llm_emb, tools)

    llm = llama_cpp.Llama(
        model_path=args.model, n_ctx=8192, n_batch=2048, n_ubatch=2048,
        n_gpu_layers=999, flash_attn=True, verbose=False,
    )
    ctx = llm._ctx.ctx
    block_cache = nexus_fsm_ext.NexusBlockCache(8 * 1024 * 1024 * 1024)
    atb_by_tool = compile_atbs(tools, args.model, Path(args.atb_workdir))
    schema_tok = {t["name"]: tokenize(llm, schema_text(t)) for t in tools}

    n_layer = int(llm.metadata.get("llama.block_count", 48))
    n_head_kv = int(llm.metadata.get("llama.attention.head_count_kv", 2))
    d_head = int(llm.metadata.get("llama.embedding_length", 5120)) // max(
        int(llm.metadata.get("llama.attention.head_count", 40)), 1
    )

    records = []
    rng = random.Random(1337)

    for s_count in session_counts:
        prefix_cache = SchemaPrefixCache()
        histories = [
            session_history(llm, sid, args.history_min, args.history_max, rng)
            for sid in range(s_count)
        ]

        for idx, case in enumerate(cases):
            session_id = idx % s_count
            preceding = histories[session_id]
            p_start = len(preceding)
            query_tokens = tokenize(llm, format_query(case["query"]))
            q_emb = embed_query(llm_emb, case["query"])
            top1 = retrieve_hybrid(case["query"], q_emb, tool_embs, tool_names, tools, 1)[0]

            cache_preceding(ctx, preceding)
            _, ttft_n1, _ = nexus_splice_recompute(
                ctx, llm, block_cache, atb_by_tool[top1], preceding,
                schema_tok[top1], query_tokens, 5.0,
            )
            records.append({
                "arm": "N1", "sessions": s_count, "case_id": idx, "session_id": session_id,
                "p_start": p_start, "ttft_us": ttft_n1, "gold": case["tool"], "retrieved": top1,
            })

            same_history_key = (session_id, top1)
            was_warm = getattr(prefix_cache, "_last_hit_key", None) == same_history_key
            if was_warm:
                prefix_cache.apply_hit(ctx, preceding, schema_tok[top1], top1, p_start)
                t0 = time.perf_counter()
                nexus_fsm_ext.decode_tokens(ctx, query_tokens, p_start + len(schema_tok[top1]), 0)
                ttft_pc = (time.perf_counter() - t0) * 1e6
                arm = "B3pc_hit"
            else:
                prefix_cache.warm(ctx, preceding, schema_tok[top1], top1, p_start)
                cache_preceding(ctx, preceding)
                t0 = time.perf_counter()
                if schema_tok[top1]:
                    nexus_fsm_ext.decode_tokens(ctx, schema_tok[top1], p_start, 0)
                nexus_fsm_ext.decode_tokens(ctx, query_tokens, p_start + len(schema_tok[top1]), 0)
                ttft_pc = (time.perf_counter() - t0) * 1e6
                arm = "B3pc_miss"
            prefix_cache._last_hit_key = same_history_key  # noqa: SLF001
            records.append({
                "arm": arm, "sessions": s_count, "case_id": idx, "session_id": session_id,
                "p_start": p_start, "ttft_us": ttft_pc, "cache_hit": arm == "B3pc_hit",
            })

        mean_p = int(np.mean([len(h) + np.mean([len(schema_tok[t["name"]]) for t in tools]) for h in histories]))
        records.append({
            "arm": "memory_estimate",
            "sessions": s_count,
            "prefix_cache_kv_bytes_est": estimate_prefix_cache_bytes(mean_p, n_layer, n_head_kv, d_head) * s_count,
            "atb_store_bytes": sum(Path(p).stat().st_size for p in atb_by_tool.values()),
        })

    by_key = {}
    for s in session_counts:
        for arm in ("N1", "B3pc_hit", "B3pc_miss"):
            rows = [r for r in records if r.get("sessions") == s and r.get("arm") == arm]
            by_key[f"S{s}_{arm}"] = summarize_arm_stats(rows, arm)

    out_path = write_artifact(
        args.output,
        "bench_dynamic_context",
        config={"sessions": session_counts, "n_queries": len(cases)},
        records=records,
        extra={"by_arm_session": by_key},
        smoke=args.smoke,
    )
    print(f"Wrote {out_path}")
    for s in session_counts:
        n1 = by_key.get(f"S{s}_N1", {})
        miss = by_key.get(f"S{s}_B3pc_miss", {})
        hit = by_key.get(f"S{s}_B3pc_hit", {})
        n1_p50 = n1.get("ttft_us_p50", 0) / 1000
        miss_p50 = miss.get("ttft_us_p50", 0) / 1000
        hit_p50 = hit.get("ttft_us_p50", 0) / 1000
        ratio = miss_p50 / n1_p50 if n1_p50 else 0
        hit_rate = hit.get("n", 0) / max(len(cases), 1)
        print(
            f"S={s}: N1={n1_p50:.0f}ms  B3pc_miss={miss_p50:.0f}ms  "
            f"B3pc_hit={hit_p50:.0f}ms  miss/N1={ratio:.1f}x  pc_hit_rate={hit_rate:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
