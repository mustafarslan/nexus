#!/usr/bin/env python3
"""F4: llama-bench throughput sanity gate vs harness prefill measurements."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../test")))

lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../build/external/llama.cpp/src"))
os.environ["LLAMA_CPP_LIB_PATH"] = lib_dir
os.environ["LLAMA_CPP_LIB"] = os.path.join(lib_dir, "libllama.dylib")

import llama_cpp  # noqa: E402
import nexus_fsm_ext  # noqa: E402
from benchmark_results import file_sha256, write_artifact  # noqa: E402
from nexus_retrieval import schema_text  # noqa: E402
from bench_routing_accuracy import load_first_10_tools  # noqa: E402

DEFAULT_MODEL = "/Volumes/AI_SSD/models/Qwen2.5-14B-Instruct-GGUF/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf"


def find_llama_bench() -> Path | None:
    candidates = [
        Path("build/bin/llama-bench"),
        Path("build/external/llama.cpp/bin/llama-bench"),
        Path("build/external/llama.cpp/examples/llama-bench/llama-bench"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def run_llama_bench(bench_bin: Path, model: str, n_prompt: int = 512) -> dict:
    cmd = [
        str(bench_bin),
        "-m", model,
        "-ngl", "999",
        "-fa", "1",
        "-p", str(n_prompt),
        "-n", "0",
        "-r", "3",
    ]
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    pp_tps = None
    for line in out.splitlines():
        m = re.search(r"pp\d+\s+\|\s+([\d.]+)\s+\±", line)
        if m:
            pp_tps = float(m.group(1))
    return {"raw_output_tail": out.splitlines()[-5:], "pp512_tps": pp_tps, "n_prompt": n_prompt}


def harness_prefill_tps(model: str, n_tokens: int, p_start: int = 256, iterations: int = 5) -> dict:
    tools = load_first_10_tools()
    llm = llama_cpp.Llama(
        model_path=model, n_ctx=16384, n_batch=2048, n_ubatch=2048,
        n_gpu_layers=999, flash_attn=True, verbose=False,
    )
    ctx = llm._ctx.ctx
    schema_tokens = [
        int(t) for t in llm.tokenize(schema_text(tools[0]).encode(), add_bos=False, special=False)
    ]
    tokens = schema_tokens[:n_tokens] if len(schema_tokens) >= n_tokens else (
        schema_tokens * ((n_tokens // len(schema_tokens)) + 1)
    )[:n_tokens]
    preceding = [1] * p_start
    latencies = []
    for _ in range(iterations):
        nexus_fsm_ext.clear_kv_cache(ctx)
        if preceding:
            nexus_fsm_ext.decode_tokens(ctx, preceding, 0, 0)
        t0 = time.perf_counter()
        chunk = 512
        for i in range(0, len(tokens), chunk):
            part = tokens[i : i + chunk]
            nexus_fsm_ext.decode_tokens(ctx, part, p_start + i, 0)
        latencies.append(time.perf_counter() - t0)
    mean_s = float(np.mean(latencies))
    tps = n_tokens / mean_s if mean_s > 0 else 0.0
    return {"n_tokens": n_tokens, "mean_ms": mean_s * 1000, "tps": tps, "iterations": iterations}


def parse_args():
    p = argparse.ArgumentParser(description="Throughput sanity gate (F4).")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output", default="results/bench_throughput_gate.json")
    p.add_argument("--ratio-threshold", type=float, default=0.5,
                   help="Fail if harness_tps / llama_bench_tps < threshold")
    return p.parse_args()


def main():
    args = parse_args()
    if not Path(args.model).exists():
        print(f"Model missing: {args.model}")
        return 1

    bench_bin = find_llama_bench()
    llama_bench = None
    if bench_bin:
        try:
            llama_bench = run_llama_bench(bench_bin, args.model)
        except subprocess.CalledProcessError as e:
            llama_bench = {"error": str(e), "pp512_tps": None}
    else:
        llama_bench = {"error": "llama-bench binary not found", "pp512_tps": None}

    harness_220 = harness_prefill_tps(args.model, 220)
    harness_512 = harness_prefill_tps(args.model, 512)

    bench_tps = llama_bench.get("pp512_tps") if llama_bench else None
    ratio = None
    pass_gate = False
    if bench_tps and harness_512["tps"]:
        ratio = harness_512["tps"] / bench_tps
        pass_gate = ratio >= args.ratio_threshold
    elif bench_tps is None:
        pass_gate = False

    summary = {
        "llama_bench_pp512_tps": bench_tps,
        "harness_220_tps": harness_220["tps"],
        "harness_512_tps": harness_512["tps"],
        "harness_to_bench_ratio": ratio,
        "pass_gate": pass_gate,
        "threshold": args.ratio_threshold,
    }

    write_artifact(
        args.output,
        "bench_throughput_gate",
        model_hash=file_sha256(args.model),
        config={"model_path": args.model, "ratio_threshold": args.ratio_threshold},
        metrics={
            "harness_512_tps": {"unit": "tok/s", "raw_samples": [harness_512["tps"]]},
            "harness_220_tps": {"unit": "tok/s", "raw_samples": [harness_220["tps"]]},
        },
        records=[llama_bench or {}, harness_220, harness_512],
        extra={"summary": summary},
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output}")
    return 0 if pass_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
