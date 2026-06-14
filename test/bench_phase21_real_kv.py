#!/usr/bin/env python3
"""Phase C: real-KV TTFT with honest baselines, splice+recompute, cold vs warm load."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
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
from benchmark_results import file_sha256, write_artifact  # noqa: E402
from bench_routing_accuracy import load_first_10_tools  # noqa: E402
from nexus_recompute import plan_suffix_recompute, recompute_tail  # noqa: E402

DEFAULT_MODEL = "/Volumes/AI_SSD/models/Qwen2.5-14B-Instruct-GGUF/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf"
DEFAULT_ATB = "results/phaseA_tool_match_work/tool_0.isolated.atb"
P_START = 256
RECOMPUTE_PCT = 5.0


def tokenize(llm, text: str) -> list[int]:
    return [int(t) for t in llm.tokenize(text.encode("utf-8"), add_bos=False, special=False)]


def schema_text(tool: dict) -> str:
    return json.dumps({
        "name": tool["name"],
        "description": tool.get("description") or tool.get("desc", ""),
        "inputSchema": tool.get("inputSchema") or {},
    }, sort_keys=True)


def build_bloat_tokens(llm, tools, target: int) -> list[int]:
    chunks = []
    for t in tools:
        chunks.extend(tokenize(llm, schema_text(t)))
    out = []
    while len(out) < target:
        out.extend(chunks)
    return out[:target]


def cache_preceding(ctx, preceding: list[int]) -> None:
    nexus_fsm_ext.clear_kv_cache(ctx)
    if preceding:
        nexus_fsm_ext.decode_tokens(ctx, preceding, 0, 0)


def prefill_ms(ctx, tokens: list[int], start: int = 0) -> float:
    if not tokens:
        return 0.0
    t0 = time.perf_counter()
    chunk = 512
    for i in range(0, len(tokens), chunk):
        part = tokens[i : i + chunk]
        nexus_fsm_ext.decode_tokens(ctx, part, start + i, 0)
    return (time.perf_counter() - t0) * 1000


def splice_ms(ctx, cache, atb_path: str, p_start: int, cold: bool = False) -> float:
    if cold:
        cache = nexus_fsm_ext.NexusBlockCache(8 * 1024 * 1024 * 1024)
    t0 = time.perf_counter()
    handle = cache.get_or_load(atb_path)
    nexus_fsm_ext.inject_tool_page(ctx, handle, p_start, 0)
    return (time.perf_counter() - t0) * 1000


def splice_recompute_ms(ctx, cache, atb_path: str, schema_tokens: list[int], p_start: int,
                        recompute_pct: float, cold: bool = False) -> float:
    if cold:
        cache = nexus_fsm_ext.NexusBlockCache(8 * 1024 * 1024 * 1024)
    t0 = time.perf_counter()
    handle = cache.get_or_load(atb_path)
    nexus_fsm_ext.inject_tool_page(ctx, handle, p_start, 0)
    plan = plan_suffix_recompute("tail", len(schema_tokens), recompute_pct)
    recompute_tail(ctx, schema_tokens, p_start, recompute_pct)
    return (time.perf_counter() - t0) * 1000, plan.actual_recompute_tokens


def summarize(xs: list[float]) -> dict:
    a = np.array(xs, dtype=np.float64)
    return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)), "p99": float(np.percentile(a, 99))}


def parse_args():
    p = argparse.ArgumentParser(description="Phase C real-KV TTFT benchmark.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--atb", default=DEFAULT_ATB)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--output", default="results/bench_phase21_ttft_real.json")
    p.add_argument("--recompute-pct", type=float, default=RECOMPUTE_PCT)
    p.add_argument("--bloat-every", type=int, default=10,
                   help="Measure 12.5k bloat/single-schema baselines every N iterations (splice every iter).")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--tag", default="")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    tools = load_first_10_tools()
    tool0 = tools[0]
    llm = llama_cpp.Llama(
        model_path=args.model, n_ctx=16384, n_batch=2048, n_ubatch=2048,
        n_gpu_layers=999, flash_attn=True, verbose=False,
    )
    ctx = llm._ctx.ctx
    cache = nexus_fsm_ext.NexusBlockCache(4 * 1024 * 1024 * 1024)
    schema_tokens = tokenize(llm, schema_text(tool0))
    bloat_tokens = build_bloat_tokens(llm, tools, 12500)
    preceding = tokenize(llm, "You are a tool-using assistant.\n") + [1] * max(0, P_START - 8)
    preceding = preceding[:P_START]
    query_tokens = tokenize(llm, "Query: list commits\nSelected Tool Name:")

    metrics = {
        "baseline_bloat_prefill_ms": [],
        "baseline_single_schema_prefill_ms": [],
        "warm_splice_ms": [],
        "cold_splice_ms": [],
        "splice_recompute_ms": [],
        "actual_recompute_tokens": [],
        "true_nexus_ttft_ms": [],
    }
    recompute_plan_info = plan_suffix_recompute("tail", len(schema_tokens), args.recompute_pct)

    for _ in range(args.warmup):
        cache_preceding(ctx, preceding)
        prefill_ms(ctx, bloat_tokens, P_START)
        splice_ms(ctx, cache, args.atb, P_START, cold=False)

    for i in range(args.iterations):
        if i % args.bloat_every == 0:
            cache_preceding(ctx, preceding)
            metrics["baseline_bloat_prefill_ms"].append(prefill_ms(ctx, bloat_tokens, P_START))
            cache_preceding(ctx, preceding)
            metrics["baseline_single_schema_prefill_ms"].append(prefill_ms(ctx, schema_tokens, P_START))

        cache_preceding(ctx, preceding)
        metrics["warm_splice_ms"].append(splice_ms(ctx, cache, args.atb, P_START, cold=False))

        cache_preceding(ctx, preceding)
        metrics["cold_splice_ms"].append(splice_ms(ctx, cache, args.atb, P_START, cold=True))

        cache_preceding(ctx, preceding)
        srec, actual_tok = splice_recompute_ms(ctx, cache, args.atb, schema_tokens, P_START, args.recompute_pct, cold=False)
        metrics["splice_recompute_ms"].append(srec)
        metrics["actual_recompute_tokens"].append(actual_tok)
        cache_preceding(ctx, preceding)
        q_ms = prefill_ms(ctx, query_tokens, P_START + len(schema_tokens))
        metrics["true_nexus_ttft_ms"].append(metrics["warm_splice_ms"][-1] + srec + q_ms)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{args.iterations}", flush=True)

    bloat_p50 = np.percentile(metrics["baseline_bloat_prefill_ms"], 50)
    single_p50 = np.percentile(metrics["baseline_single_schema_prefill_ms"], 50)
    nexus_p50 = np.percentile(metrics["true_nexus_ttft_ms"], 50)

    summary = {k: summarize(v) for k, v in metrics.items()}
    summary["speedup_vs_bloat_p50"] = float(bloat_p50 / max(nexus_p50, 1e-6))
    summary["speedup_vs_single_schema_p50"] = float(single_p50 / max(nexus_p50, 1e-6))

    out_path = write_artifact(
        args.output,
        "bench_phase21_ttft_real",
        model_hash=file_sha256(args.model) if Path(args.model).exists() else "",
        config={
            "model_path": args.model,
            "atb_path": args.atb,
            "atb_payload": "real_compiled_kv",
            "baseline_bloat_tokens": 12500,
            "single_schema_tokens": len(schema_tokens),
            "p_start": P_START,
            "recompute_pct": args.recompute_pct,
            "actual_recompute_tokens_nominal": recompute_plan_info.actual_recompute_tokens,
            "actual_recompute_pct_nominal": recompute_plan_info.actual_recompute_pct,
            "iterations": args.iterations,
            "bloat_every": args.bloat_every,
        },
        metrics={k: {"unit": "ms" if k != "actual_recompute_tokens" else "tokens", "raw_samples": v} for k, v in metrics.items()},
        records=[],
        extra={"summary": summary},
        smoke=args.smoke,
        tag=args.tag or None,
        force=args.force,
    )

    print(f"Wrote {out_path}")
    print(f"speedup vs bloat P50: {summary['speedup_vs_bloat_p50']:.2f}x")
    print(f"speedup vs single-schema P50: {summary['speedup_vs_single_schema_p50']:.2f}x")
    print(f"warm splice P50: {summary['warm_splice_ms']['p50']:.2f} ms")
    print(f"cold splice P50: {summary['cold_splice_ms']['p50']:.2f} ms")


if __name__ == "__main__":
    main()
