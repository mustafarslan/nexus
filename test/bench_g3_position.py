#!/usr/bin/env python3
"""G3: Root-cause position-dependent splice error experiments."""
from __future__ import annotations

import argparse
import json
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
from benchmark_results import file_sha256, write_artifact  # noqa: E402
from bench_phaseB_selective_recompute import (  # noqa: E402
    DEFAULT_MODEL,
    build_preceding_tokens,
    format_query,
    kl_ref_to_splice,
    reference_logits,
    schema_text_for,
    splice_recompute_query,
    tokenize,
)
from bench_routing_accuracy import load_first_10_tools, queries_dataset  # noqa: E402
from nexus_recompute import recompute_tail  # noqa: E402


def bare_splice_logits(ctx, cache, atb, preceding, schema_tokens, query_tokens, p_start):
    selected = []
    logits, _ = splice_recompute_query(
        ctx, cache, atb, preceding, schema_tokens, query_tokens, p_start, selected,
    )
    return logits


def tail_r5_logits(ctx, cache, atb, preceding, schema_tokens, query_tokens, p_start):
    n = max(1, int(len(schema_tokens) * 0.05))
    selected = list(range(len(schema_tokens) - n, len(schema_tokens)))
    logits, _ = splice_recompute_query(
        ctx, cache, atb, preceding, schema_tokens, query_tokens, p_start, selected,
    )
    return logits


def parse_args():
    p = argparse.ArgumentParser(description="G3 position-dependent splice error study.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--p-starts", default="256,1024,2048")
    p.add_argument("--output", default="results/bench_g3_position.json")
    return p.parse_args()


def main():
    args = parse_args()
    if not Path(args.model).exists():
        print(f"Model missing: {args.model}")
        return 1

    tool = load_first_10_tools()[0]
    atb = Path("results/phaseA_tool_match_work/tool_0.isolated.atb")
    if not atb.exists():
        print("Missing ATB; run phase A compile first")
        return 1

    p_starts = [int(x) for x in args.p_starts.split(",")]
    llm = llama_cpp.Llama(
        model_path=args.model, n_ctx=8192, n_batch=4096, n_ubatch=4096,
        n_gpu_layers=999, flash_attn=True, verbose=False,
    )
    ctx = llm._ctx.ctx
    cache = nexus_fsm_ext.NexusBlockCache(4 * 1024 * 1024 * 1024)
    schema_tokens = tokenize(llm, schema_text_for(tool))
    case = queries_dataset[0]
    query_tokens = tokenize(llm, format_query(case["query"]))

    records = []
    for p_start in p_starts:
        preceding = build_preceding_tokens(llm, "You are a tool-using assistant.", p_start)
        ref = reference_logits(ctx, preceding, schema_tokens, query_tokens)
        bare = bare_splice_logits(ctx, cache, atb, preceding, schema_tokens, query_tokens, p_start)
        tail5 = tail_r5_logits(ctx, cache, atb, preceding, schema_tokens, query_tokens, p_start)
        records.append({
            "p_start": p_start,
            "schema_tokens": len(schema_tokens),
            "kl_bare_splice": kl_ref_to_splice(ref, bare),
            "kl_tail_r5": kl_ref_to_splice(ref, tail5),
            "top1_ref": int(np.argmax(ref)),
            "top1_bare": int(np.argmax(bare)),
            "top1_tail_r5": int(np.argmax(tail5)),
        })
        print(f"P={p_start}: KL bare={records[-1]['kl_bare_splice']:.3f} tail5={records[-1]['kl_tail_r5']:.3f}")

    write_artifact(
        args.output,
        "bench_g3_position",
        model_hash=file_sha256(args.model),
        config={"model_path": args.model, "p_starts": p_starts},
        metrics={},
        records=records,
        extra={"hypothesis": "attention_sink / position-dependent splice drift; compare KL vs P_start"},
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
