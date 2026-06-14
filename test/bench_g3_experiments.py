#!/usr/bin/env python3
"""L3: G3 root-cause experiments — sink prefix, compile-at-position, position sweep."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
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
from bench_phaseB_suffix_recompute import (  # noqa: E402
    DEFAULT_MODEL,
    build_preceding_tokens,
    compute_deviations,
    format_query,
    kl_ref_to_splice,
    reference_logits,
    schema_text_for,
    tokenize,
)
from bench_routing_accuracy import load_first_10_tools, queries_dataset  # noqa: E402
from nexus_recompute import plan_suffix_recompute, recompute_plan  # noqa: E402

SINK_PREFIX = (
    "The following is a stable system context block used for attention calibration. "
    "It repeats filler content to occupy early context positions. "
) * 8


def splice_tail5_logits(ctx, cache, atb, preceding, schema_tokens, query_tokens, p_start):
    plan = plan_suffix_recompute("tail", len(schema_tokens), 5.0)
    cache_preceding = lambda c, p: (nexus_fsm_ext.clear_kv_cache(c), nexus_fsm_ext.decode_tokens(c, p, 0, 0) if p else None)
    cache_preceding(ctx, preceding)
    handle = cache.get_or_load(str(atb))
    nexus_fsm_ext.inject_tool_page(ctx, handle, p_start, 0)
    recompute_plan(ctx, schema_tokens, p_start, plan)
    return np.array(
        nexus_fsm_ext.decode_tokens(ctx, query_tokens, p_start + len(schema_tokens), 0),
        dtype=np.float32,
    )


def compile_atb(model, schema_path, output_path, base_pos: int = 0) -> Path:
    cmd = ["./build/nexus_kv_compiler", "--model", model, "--schema", str(schema_path),
           "--output", str(output_path), "--base-pos", str(base_pos)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return Path(output_path)


def parse_args():
    p = argparse.ArgumentParser(description="G3 root-cause experiments.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output", default="results/bench_g3_experiments.json")
    p.add_argument("--p-starts", default="256,1024")
    return p.parse_args()


def main():
    args = parse_args()
    if not Path(args.model).exists():
        print(f"Model missing: {args.model}")
        return 1

    tool = load_first_10_tools()[0]
    case = queries_dataset[0]
    p_starts = [int(x) for x in args.p_starts.split(",") if x.strip()]
    work = Path("results/g3_experiments_work")
    work.mkdir(parents=True, exist_ok=True)

    llm = llama_cpp.Llama(
        model_path=args.model, n_ctx=8192, n_batch=4096, n_ubatch=4096,
        n_gpu_layers=999, flash_attn=True, verbose=False,
    )
    ctx = llm._ctx.ctx
    cache = nexus_fsm_ext.NexusBlockCache(4 * 1024 * 1024 * 1024)

    schema_plain = schema_text_for(tool)
    schema_sink = json.dumps({
        "sink_prefix": SINK_PREFIX,
        **json.loads(schema_plain),
    }, sort_keys=True)

    schema_path = work / "tool_0.schema.json"
    schema_path.write_text(schema_plain)
    atb_default = work / "tool_0_pos0.atb"
    if not atb_default.exists():
        compile_atb(args.model, schema_path, atb_default, base_pos=0)

    schema_tokens = tokenize(llm, schema_plain)
    query_tokens = tokenize(llm, format_query(case["query"]))
    records = []

    for p_start in p_starts:
        preceding = build_preceding_tokens(llm, "You are a tool-using assistant.", p_start)
        ref = reference_logits(ctx, preceding, schema_tokens, query_tokens, p_start, legacy_cache=True)
        bare = splice_tail5_logits(ctx, cache, atb_default, preceding, schema_tokens, query_tokens, p_start)
        records.append({
            "experiment": "baseline_default_atb",
            "p_start": p_start,
            "kl_bare_tail5": kl_ref_to_splice(ref, bare),
            "top1_ref": int(np.argmax(ref)),
            "top1_splice": int(np.argmax(bare)),
        })

        # Compile-at-canonical-position: compile with filler at base_pos=p_start
        atb_canon = work / f"tool_0_compile_at_{p_start}.atb"
        if not atb_canon.exists():
            compile_atb(args.model, schema_path, atb_canon, base_pos=p_start)
        canon = splice_tail5_logits(ctx, cache, atb_canon, preceding, schema_tokens, query_tokens, p_start)
        records.append({
            "experiment": "compile_at_p_start",
            "p_start": p_start,
            "compile_base_pos": p_start,
            "kl_tail5": kl_ref_to_splice(ref, canon),
            "delta_pos": 0,
        })

        # Sink-prefix schema text (longer compile, same splice offset)
        sink_path = work / "tool_0_sink.schema.json"
        sink_path.write_text(schema_sink)
        atb_sink = work / "tool_0_sink.atb"
        if not atb_sink.exists():
            compile_atb(args.model, sink_path, atb_sink, base_pos=0)
        sink_toks = tokenize(llm, schema_sink)
        ref_sink = reference_logits(ctx, preceding, sink_toks, query_tokens, p_start, legacy_cache=True)
        bare_sink = splice_tail5_logits(ctx, cache, atb_sink, preceding, sink_toks, query_tokens, p_start)
        records.append({
            "experiment": "sink_prefix_schema",
            "p_start": p_start,
            "schema_tokens": len(sink_toks),
            "kl_tail5_vs_sink_ref": kl_ref_to_splice(ref_sink, bare_sink),
        })

        print(f"P={p_start}: baseline KL={records[-3]['kl_bare_tail5']:.4f} "
              f"compile_at={records[-2]['kl_tail5']:.4f} sink={records[-1]['kl_tail5_vs_sink_ref']:.4f}")

    write_artifact(
        args.output,
        "bench_g3_experiments",
        model_hash=file_sha256(args.model),
        config={"model_path": args.model, "p_starts": p_starts},
        metrics={},
        records=records,
        extra={
            "hypotheses": [
                "attention_sink: longer sink-prefix schema changes KL vs P",
                "compile_at_position: compile base_pos=p_start reduces RoPE delta to 0",
                "fp32_re_rotation: deferred (requires C++ branch in apply_relative_rope_shift)",
            ],
        },
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
