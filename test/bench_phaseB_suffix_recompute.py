#!/usr/bin/env python3
"""Phase L1: suffix-fraction selective recompute ablation with honest cost axis.

Design space: suffix fraction s% with selectors {tail, hkvd_suffix_start, oracle_suffix_start}.
Cost axis = actual_recompute_tokens (causal suffix decode from min selected).

Harness optimizations (Phase 1):
  - Reuse preceding KV via invalidate_sequence (not full clear+prefill per config)
  - P-outer loop: 3 preceding prefills per run instead of thousands
  - Periodic defrag reset every N cases within a P tier
  - JSONL checkpoint + --resume
"""
from __future__ import annotations

import argparse
import json
import math
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
from bench_routing_accuracy import queries_dataset, load_first_10_tools  # noqa: E402
from nexus_recompute import (  # noqa: E402
    DEFAULT_SUFFIX_PCTS,
    SUFFIX_SELECTORS,
    plan_suffix_recompute,
    recompute_plan,
)

DEFAULT_MODEL = "/Volumes/AI_SSD/models/Qwen2.5-14B-Instruct-GGUF/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf"
DEFAULT_SYSTEM_PROMPT = "You are a tool-using assistant. Use the provided tool schema to answer the user."
KV_RESET_END = -1

_FILLER_TURNS = [
    "User: I'm building a small data pipeline and I keep running into timezone bugs.",
    "Assistant: Store everything in UTC and convert only at display boundaries.",
    "User: We also have a flaky integration test that fails about one run in twenty.",
    "Assistant: Flaky tests are often hidden ordering or timing assumptions.",
]


def tokenize(llm, text: str) -> list[int]:
    return [int(t) for t in llm.tokenize(text.encode("utf-8"), add_bos=False, special=False)]


def log_softmax(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    x = x - np.max(x)
    return x - math.log(float(np.exp(x).sum()))


def kl_ref_to_splice(ref_logits: np.ndarray, splice_logits: np.ndarray) -> float:
    ref_logp = log_softmax(ref_logits)
    splice_logp = log_softmax(splice_logits)
    ref_p = np.exp(ref_logp)
    return float(np.sum(ref_p * (ref_logp - splice_logp)))


def build_preceding_tokens(llm, system_prompt: str, target_len: int) -> list[int]:
    if target_len <= 0:
        return []
    text = system_prompt + "\n"
    idx = 0
    toks = tokenize(llm, text)
    while len(toks) < target_len:
        text += _FILLER_TURNS[idx % len(_FILLER_TURNS)] + "\n"
        idx += 1
        toks = tokenize(llm, text)
        if idx > 10000:
            break
    return toks[:target_len]


def schema_text_for(tool: dict) -> str:
    return json.dumps({
        "name": tool["name"],
        "description": tool.get("description") or tool.get("desc", ""),
        "inputSchema": tool.get("inputSchema") or {},
    }, sort_keys=True)


def format_query(user_query: str) -> str:
    return (
        "You are a tool routing agent. Given the tool schema above, select the tool "
        "that matches the user's intent. You MUST output ONLY the tool name.\n\n"
        f"Query: {user_query}\nSelected Tool Name:"
    )


def prefill_preceding(ctx, preceding: list[int]) -> None:
    nexus_fsm_ext.clear_kv_cache(ctx)
    if preceding:
        nexus_fsm_ext.decode_tokens(ctx, preceding, 0, 0)


def reset_from_p_start(ctx, p_start: int) -> None:
    """Drop schema/query KV from p_start onward; keep preceding [0, p_start)."""
    nexus_fsm_ext.invalidate_sequence(ctx, 0, p_start, KV_RESET_END)


def cache_preceding(ctx, preceding: list[int]) -> None:
    """Legacy: full clear + preceding prefill (for A/B validation)."""
    prefill_preceding(ctx, preceding)


def prepare_experiment(ctx, preceding: list[int], p_start: int, *, legacy_cache: bool) -> None:
    if legacy_cache:
        cache_preceding(ctx, preceding)
    else:
        reset_from_p_start(ctx, p_start)


def reference_logits(ctx, preceding, schema_tokens, query_tokens, p_start: int, *, legacy_cache: bool) -> np.ndarray:
    prepare_experiment(ctx, preceding, p_start, legacy_cache=legacy_cache)
    return np.array(
        nexus_fsm_ext.decode_tokens(ctx, schema_tokens + query_tokens, p_start, 0),
        dtype=np.float32,
    )


def token_deviation_l2(true_k, splice_k) -> np.ndarray:
    diff = true_k - splice_k
    return np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1)


def read_kv_slice(ctx, layer, p0, p1):
    flat, seq_len, n_head_kv, d_head = nexus_fsm_ext.read_kv_slice(ctx, layer, p0, p1, False)
    return np.array(flat, dtype=np.float32).reshape(seq_len, n_head_kv, d_head)


def compute_deviations(ctx, cache, atb_path, preceding, schema_tokens, p_start, schema_len, n_layer, *, legacy_cache: bool):
    prepare_experiment(ctx, preceding, p_start, legacy_cache=legacy_cache)
    nexus_fsm_ext.decode_tokens(ctx, schema_tokens, p_start, 0)
    true_by_layer = {il: read_kv_slice(ctx, il, p_start, p_start + schema_len) for il in range(n_layer)}
    prepare_experiment(ctx, preceding, p_start, legacy_cache=legacy_cache)
    handle = cache.get_or_load(str(atb_path))
    nexus_fsm_ext.inject_tool_page(ctx, handle, p_start, 0)
    layer1_dev = token_deviation_l2(true_by_layer[1], read_kv_slice(ctx, 1, p_start, p_start + schema_len))
    oracle_dev = np.zeros(schema_len, dtype=np.float64)
    for il in range(n_layer):
        oracle_dev += token_deviation_l2(true_by_layer[il], read_kv_slice(ctx, il, p_start, p_start + schema_len))
    return layer1_dev, oracle_dev


def splice_recompute_with_plan(ctx, cache, atb_path, preceding, schema_tokens, query_tokens, p_start, plan, *, legacy_cache: bool):
    prepare_experiment(ctx, preceding, p_start, legacy_cache=legacy_cache)
    t0 = time.perf_counter()
    handle = cache.get_or_load(str(atb_path))
    nexus_fsm_ext.inject_tool_page(ctx, handle, p_start, 0)
    actual = recompute_plan(ctx, schema_tokens, p_start, plan) if plan.actual_recompute_tokens else 0
    logits = np.array(
        nexus_fsm_ext.decode_tokens(ctx, query_tokens, p_start + len(schema_tokens), 0),
        dtype=np.float32,
    )
    return logits, (time.perf_counter() - t0) * 1e6, actual


def summarize(values):
    if not values:
        return {"n": 0}
    arr = np.array(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "std": float(arr.std()),
    }


def record_key(case_id: str, p_start: int) -> str:
    return f"{case_id}|P{p_start}"


def load_completed_keys(checkpoint_path: Path) -> set[str]:
    if not checkpoint_path.exists():
        return set()
    done: set[str] = set()
    with checkpoint_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done.add(record_key(row["case_id"], row["p_start"]))
    return done


def append_checkpoint(checkpoint_path: Path, case_records: list[dict]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("a") as f:
        for row in case_records:
            f.write(json.dumps(row) + "\n")


def load_records_from_checkpoint(checkpoint_path: Path) -> list[dict]:
    if not checkpoint_path.exists():
        return []
    records = []
    with checkpoint_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_by_config(records, p_starts, selectors, suffix_pcts):
    by_config = {}
    for p_start in p_starts:
        for selector in selectors:
            for s_pct in suffix_pcts:
                key = f"P{p_start}_{selector}_s{s_pct}"
                rows = [
                    r for r in records
                    if r["p_start"] == p_start and r["selector"] == selector and r["nominal_suffix_pct"] == s_pct
                ]
                by_config[key] = {
                    "p_start": p_start,
                    "selector": selector,
                    "nominal_suffix_pct": s_pct,
                    "actual_recompute_tokens_mean": float(np.mean([r["actual_recompute_tokens"] for r in rows])) if rows else 0,
                    "kl_ref_to_splice": summarize([r["kl_ref_to_splice"] for r in rows]),
                    "top1_agreement_rate": float(np.mean([1.0 if r["top1_agreement"] else 0.0 for r in rows])) if rows else 0,
                    "splice_recompute_us": summarize([r["splice_recompute_us"] for r in rows]),
                }
    return by_config


def build_canaries(records, p_starts):
    canaries = {}
    for p_start in p_starts:
        r100 = [r for r in records if r["p_start"] == p_start and r["nominal_suffix_pct"] == 100.0]
        mean_kl = float(np.mean([r["kl_ref_to_splice"] for r in r100])) if r100 else None
        top1 = float(np.mean([1.0 if r["top1_agreement"] else 0.0 for r in r100])) if r100 else None
        canaries[str(p_start)] = {
            "r100_mean_kl": mean_kl,
            "r100_top1_agreement_rate": top1,
            "pass": bool(r100) and mean_kl is not None and mean_kl < 0.01 and top1 == 1.0,
        }
    pooled_r100 = [r for r in records if r["nominal_suffix_pct"] == 100.0]
    pooled = {
        "r100_mean_kl": float(np.mean([r["kl_ref_to_splice"] for r in pooled_r100])) if pooled_r100 else None,
        "r100_top1_agreement_rate": float(np.mean([1.0 if r["top1_agreement"] else 0.0 for r in pooled_r100])) if pooled_r100 else None,
        "pass": bool(pooled_r100) and float(np.mean([r["kl_ref_to_splice"] for r in pooled_r100])) < 0.01,
    }
    return {"per_p": canaries, "pooled": pooled}


def parse_args():
    p = argparse.ArgumentParser(description="Suffix-fraction recompute ablation (L1).")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--p-starts", default="256,1024,4096")
    p.add_argument("--selectors", default=",".join(SUFFIX_SELECTORS))
    p.add_argument("--suffix-pcts", default=",".join(str(x) for x in DEFAULT_SUFFIX_PCTS))
    p.add_argument("--case-limit", type=int, default=50)
    p.add_argument("--output", default="results/bench_phaseB_suffix_recompute_full.json")
    p.add_argument("--checkpoint-jsonl", default="results/bench_phaseB_suffix_recompute_full.records.jsonl")
    p.add_argument("--atb-workdir", default="results/phaseA_tool_match_work")
    p.add_argument("--workdir", default="results/phaseB_recompute_work")
    p.add_argument("--n-ctx", type=int, default=8192)
    p.add_argument("--defrag-every", type=int, default=10,
                   help="Full preceding re-prefill every N cases within a P tier (KV defrag guard).")
    p.add_argument("--resume", action="store_true", help="Skip (case_id, P) pairs already in checkpoint JSONL.")
    p.add_argument("--legacy-cache", action="store_true",
                   help="Use full clear+prefill per config (slow; for A/B equivalence checks).")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--tag", default="")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    p_starts = [int(x) for x in args.p_starts.split(",") if x.strip()]
    selectors = [x.strip() for x in args.selectors.split(",") if x.strip()]
    suffix_pcts = [float(x) for x in args.suffix_pcts.split(",") if x.strip()]
    if args.quick or args.smoke:
        p_starts = [256, 1024]
        selectors = ["tail", "hkvd_suffix_start"]
        suffix_pcts = [0, 5, 10, 20, 100]
        if args.smoke:
            args.case_limit = min(args.case_limit, 3)

    tools = load_first_10_tools()
    tool_by_name = {t["name"]: t for t in tools}
    cases = [c for c in queries_dataset if c["tool"] in tool_by_name][: args.case_limit]
    checkpoint_path = Path(args.checkpoint_jsonl)
    completed = load_completed_keys(checkpoint_path) if args.resume else set()
    mode = "legacy-cache" if args.legacy_cache else "kv-reuse"
    print(f"Suffix ablation: {len(cases)} cases P={p_starts} selectors={selectors} s%={suffix_pcts} mode={mode}")
    if completed:
        print(f"  resume: skipping {len(completed)} completed (case,P) pairs")

    llm = llama_cpp.Llama(
        model_path=args.model, n_ctx=args.n_ctx,
        n_batch=min(args.n_ctx, max(2048, max(p_starts) + 640)),
        n_ubatch=min(args.n_ctx, max(2048, max(p_starts) + 640)),
        n_gpu_layers=999, flash_attn=True, verbose=False,
    )
    ctx = llm._ctx.ctx
    cache = nexus_fsm_ext.NexusBlockCache(8 * 1024 * 1024 * 1024)
    n_layer = int(llm.metadata.get("llama.block_count", llm.metadata.get("qwen2.block_count", 48)))

    atb_by_tool, schema_tokens_by_tool, schema_len_by_tool = {}, {}, {}
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    atb_src = Path(args.atb_workdir)
    for i, tool in enumerate(tools):
        text = schema_text_for(tool)
        atb_path = atb_src / f"tool_{i}.isolated.atb"
        if not atb_path.exists():
            schema_path = workdir / f"tool_{i}.schema.json"
            schema_path.write_text(text)
            atb_path = workdir / f"tool_{i}.isolated.atb"
            subprocess.run(
                ["./build/nexus_kv_compiler", "--model", args.model, "--schema", str(schema_path), "--output", str(atb_path)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        toks = tokenize(llm, text)
        atb_by_tool[tool["name"]] = atb_path
        schema_tokens_by_tool[tool["name"]] = toks
        schema_len_by_tool[tool["name"]] = len(toks)

    preceding_by_pos = {p: build_preceding_tokens(llm, DEFAULT_SYSTEM_PROMPT, p) for p in p_starts}
    dev_cache: dict[tuple[str, int], tuple] = {}
    records = load_records_from_checkpoint(checkpoint_path) if args.resume else []
    cases_done = 0

    # P-outer loop: preceding prefilled once per P tier (or on defrag reset).
    for p_start in p_starts:
        preceding = preceding_by_pos[p_start]
        if not args.legacy_cache:
            prefill_preceding(ctx, preceding)
        cases_in_tier = 0

        for idx, case in enumerate(cases):
            tool_name = case["tool"]
            cid = f"{tool_name}-{idx}"
            if record_key(cid, p_start) in completed:
                continue

            schema_tokens = schema_tokens_by_tool[tool_name]
            schema_len = schema_len_by_tool[tool_name]
            query_tokens = tokenize(llm, format_query(case["query"]))
            atb_path = atb_by_tool[tool_name]

            if not args.legacy_cache and cases_in_tier > 0 and cases_in_tier % args.defrag_every == 0:
                prefill_preceding(ctx, preceding)

            ref = reference_logits(
                ctx, preceding, schema_tokens, query_tokens, p_start, legacy_cache=args.legacy_cache)
            top1_ref = int(np.argmax(ref))

            dev_key = (tool_name, p_start)
            if dev_key not in dev_cache:
                dev_cache[dev_key] = compute_deviations(
                    ctx, cache, atb_path, preceding, schema_tokens, p_start, schema_len, n_layer,
                    legacy_cache=args.legacy_cache)
            layer1_dev, oracle_dev = dev_cache[dev_key]

            case_records = []
            for selector in selectors:
                for s_pct in suffix_pcts:
                    if s_pct <= 0:
                        prepare_experiment(ctx, preceding, p_start, legacy_cache=args.legacy_cache)
                        t0 = time.perf_counter()
                        handle = cache.get_or_load(str(atb_path))
                        nexus_fsm_ext.inject_tool_page(ctx, handle, p_start, 0)
                        splice = np.array(
                            nexus_fsm_ext.decode_tokens(ctx, query_tokens, p_start + schema_len, 0),
                            dtype=np.float32,
                        )
                        lat_us = (time.perf_counter() - t0) * 1e6
                        actual = 0
                        start_idx = None
                        actual_pct = 0.0
                    else:
                        plan = plan_suffix_recompute(
                            selector, schema_len, s_pct, layer1_dev, oracle_dev)
                        splice, lat_us, actual = splice_recompute_with_plan(
                            ctx, cache, atb_path, preceding, schema_tokens, query_tokens, p_start, plan,
                            legacy_cache=args.legacy_cache)
                        start_idx = plan.suffix_start_idx
                        actual_pct = plan.actual_recompute_pct

                    kl = kl_ref_to_splice(ref, splice)
                    case_records.append({
                        "case_id": cid,
                        "p_start": p_start,
                        "selector": selector,
                        "nominal_suffix_pct": s_pct,
                        "suffix_start_idx": start_idx,
                        "actual_recompute_tokens": actual,
                        "actual_recompute_pct": actual_pct,
                        "schema_tokens": schema_len,
                        "kl_ref_to_splice": kl,
                        "top1_agreement": top1_ref == int(np.argmax(splice)),
                        "splice_recompute_us": lat_us,
                    })

            records.extend(case_records)
            append_checkpoint(checkpoint_path, case_records)
            cases_in_tier += 1
            cases_done += 1
            if cases_done % 5 == 0:
                print(f"  {cases_done}/{len(cases) * len(p_starts)} (case,P) pairs")

    by_config = build_by_config(records, p_starts, selectors, suffix_pcts)
    canary = build_canaries(records, p_starts)

    out_path = write_artifact(
        args.output,
        "bench_phaseB_suffix_recompute",
        model_hash=file_sha256(args.model) if Path(args.model).exists() else "",
        config={
            "model_path": args.model,
            "p_starts": p_starts,
            "selectors": selectors,
            "suffix_pcts": suffix_pcts,
            "n_cases": len(cases),
            "cost_axis": "actual_recompute_tokens",
            "harness_mode": mode,
            "defrag_every": args.defrag_every,
        },
        metrics={
            "kl_ref_to_splice": {"unit": "nats", "raw_samples": [r["kl_ref_to_splice"] for r in records]},
            "actual_recompute_tokens": {"unit": "tokens", "raw_samples": [r["actual_recompute_tokens"] for r in records]},
        },
        records=records,
        extra={"by_config": by_config, "correctness_canary": canary},
        smoke=args.smoke,
        tag=args.tag or None,
        force=args.force,
    )
    print(f"Wrote {out_path}")
    print(f"Canary per-P: {json.dumps(canary['per_p'], indent=2)}")


if __name__ == "__main__":
    main()
