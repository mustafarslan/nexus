#!/usr/bin/env python3
"""P0: Bisect P=1024 splice canary failure — per-layer KV divergence vs text prefill."""
from __future__ import annotations

import argparse
import json
import math
import os
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
from benchmark_results import write_artifact  # noqa: E402
from bench_routing_accuracy import load_first_10_tools  # noqa: E402
from nexus_recompute import plan_suffix_recompute, recompute_plan, schema_invalidate_end  # noqa: E402

DEFAULT_MODEL = "/Volumes/AI_SSD/models/Qwen2.5-14B-Instruct-GGUF/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf"
P_TIERS = (256, 384, 512, 768, 1024)


def tokenize(llm, text: str) -> list[int]:
    return [int(t) for t in llm.tokenize(text.encode("utf-8"), add_bos=False, special=False)]


def schema_text(tool: dict) -> str:
    return json.dumps({
        "name": tool["name"],
        "description": tool.get("description") or tool.get("desc", ""),
        "inputSchema": tool.get("inputSchema") or {},
    }, sort_keys=True)


def build_preceding(llm, target_len: int) -> list[int]:
    text = "You are a tool-using assistant.\n"
    i = 0
    toks = tokenize(llm, text)
    while len(toks) < target_len:
        text += f"User: filler turn {i}\nAssistant: ack {i}\n"
        i += 1
        toks = tokenize(llm, text)
    return toks[:target_len]


def read_kv(ctx, layer, p0, p1):
    flat, seq_len, n_head_kv, d_head = nexus_fsm_ext.read_kv_slice(ctx, layer, p0, p1, False)
    return np.array(flat, dtype=np.float32).reshape(seq_len, n_head_kv, d_head)


def token_l2(true_k, splice_k) -> np.ndarray:
    diff = true_k - splice_k
    return np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1)


def run_case(ctx, cache, atb, preceding, schema_tokens, p_start, n_layer, suffix_pct):
    schema_len = len(schema_tokens)
    nexus_fsm_ext.clear_kv_cache(ctx)
    if preceding:
        nexus_fsm_ext.decode_tokens(ctx, preceding, 0, 0)
    nexus_fsm_ext.decode_tokens(ctx, schema_tokens, p_start, 0)
    true_layers = {il: read_kv(ctx, il, p_start, p_start + schema_len) for il in range(n_layer)}

    nexus_fsm_ext.clear_kv_cache(ctx)
    if preceding:
        nexus_fsm_ext.decode_tokens(ctx, preceding, 0, 0)
    handle = cache.get_or_load(str(atb))
    nexus_fsm_ext.inject_tool_page(ctx, handle, p_start, 0)
    plan = plan_suffix_recompute("tail", schema_len, suffix_pct)
    recompute_plan(ctx, schema_tokens, p_start, plan)
    splice_layers = {il: read_kv(ctx, il, p_start, p_start + schema_len) for il in range(n_layer)}

    per_layer = {}
    first_bad = None
    for il in range(n_layer):
        dev = token_l2(true_layers[il], splice_layers[il])
        mean_dev = float(dev.mean())
        max_dev = float(dev.max())
        per_layer[str(il)] = {"mean_l2": mean_dev, "max_l2": max_dev, "p95_l2": float(np.percentile(dev, 95))}
        if first_bad is None and max_dev > 1e-3:
            first_bad = {"layer": il, "token": int(np.argmax(dev)), "l2": max_dev}

    ref_logits = np.array(
        nexus_fsm_ext.decode_tokens(ctx, schema_tokens[:1], p_start + schema_len - 1, 0),
        dtype=np.float32,
    )
    _ = ref_logits  # placeholder for future logit compare

    return {
        "suffix_pct": suffix_pct,
        "actual_recompute_tokens": plan.actual_recompute_tokens,
        "invalidate_end_exclusive": schema_invalidate_end(p_start, schema_len),
        "per_layer": per_layer,
        "first_divergence": first_bad,
    }


def parse_args():
    p = argparse.ArgumentParser(description="P1024 canary KV bisection harness.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--atb-dir", default="results/phaseA_tool_match_work")
    p.add_argument("--output", default="results/bench_p1024_canary_bisect.json")
    p.add_argument("--suffix-pct", type=float, default=100.0)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if not Path(args.model).exists():
        print(f"Model missing: {args.model}")
        return 1

    tools = load_first_10_tools()[:2 if args.smoke else 3]
    llm = llama_cpp.Llama(
        model_path=args.model, n_ctx=8192, n_batch=2048, n_ubatch=2048,
        n_gpu_layers=999, flash_attn=True, verbose=False,
    )
    ctx = llm._ctx.ctx
    cache = nexus_fsm_ext.NexusBlockCache(4 * 1024 * 1024 * 1024)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from atb_io import read_header  # noqa: E402

    records = []
    for tool_idx, tool in enumerate(tools):
        atb = Path(args.atb_dir) / f"tool_{tool_idx}.isolated.atb"
        if not atb.exists():
            continue
        n_layer = read_header(atb.read_bytes()[:128])["n_layer"]
        schema_tokens = tokenize(llm, schema_text(tool))
        for p_start in (P_TIERS if not args.smoke else (256, 1024)):
            preceding = build_preceding(llm, p_start)
            row = run_case(ctx, cache, atb, preceding, schema_tokens, p_start, n_layer, args.suffix_pct)
            row.update({"tool": tool["name"], "p_start": p_start})
            records.append(row)
            print(f"  {tool['name']} P={p_start} r{args.suffix_pct}% first_bad={row['first_divergence']}")

    out = write_artifact(
        args.output,
        "bench_p1024_canary_bisect",
        config={"suffix_pct": args.suffix_pct, "p_tiers": list(P_TIERS), "fix": "exclusive invalidate end"},
        metrics={},
        records=records,
        extra={
            "root_cause_note": (
                "recompute_selected used inclusive end (p+len-1) but llama_kv_cache_seq_rm is half-open [p0,p1); "
                "last schema token retained stale splice KV — fixed in nexus_recompute.schema_invalidate_end"
            ),
        },
        smoke=args.smoke,
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
