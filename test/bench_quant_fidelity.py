#!/usr/bin/env python3
"""N3: q8/q4 quantization fidelity through splice path on gold ATBs."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../build/external/llama.cpp/src"))
os.environ["LLAMA_CPP_LIB_PATH"] = lib_dir
os.environ["LLAMA_CPP_LIB"] = os.path.join(lib_dir, "libllama.dylib")

import llama_cpp  # noqa: E402
import nexus_fsm_ext  # noqa: E402
from benchmark_results import file_sha256, write_artifact  # noqa: E402
from bench_routing_accuracy import load_first_10_tools  # noqa: E402
from nexus_recompute import plan_suffix_recompute, recompute_plan  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from atb_io import make_quantized_atb, read_kv_fp16  # noqa: E402

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


def log_softmax(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    x = x - np.max(x)
    return x - math.log(float(np.exp(x).sum()))


def kl_divergence(ref: np.ndarray, test: np.ndarray) -> float:
    ref_lp = log_softmax(ref)
    test_lp = log_softmax(test)
    ref_p = np.exp(ref_lp)
    return float(np.sum(ref_p * (ref_lp - test_lp)))


def parse_args():
    p = argparse.ArgumentParser(description="Quantized ATB splice fidelity (N3).")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--atb-dir", default="results/phaseA_tool_match_work")
    p.add_argument("--output", default="results/bench_quant_fidelity.json")
    p.add_argument("--quants", default="fp16,q8,q4")
    p.add_argument("--suffix-pct", type=float, default=5.0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if not Path(args.model).exists():
        print(f"Model missing: {args.model}")
        return 1

    tools = load_first_10_tools()
    if args.smoke:
        tools = tools[:2]
    quants = [q.strip() for q in args.quants.split(",") if q.strip()]

    llm = llama_cpp.Llama(
        model_path=args.model, n_ctx=8192, n_batch=2048, n_ubatch=2048,
        n_gpu_layers=999, flash_attn=True, verbose=False,
    )
    ctx = llm._ctx.ctx
    cache = nexus_fsm_ext.NexusBlockCache(4 * 1024 * 1024 * 1024)
    preceding = tokenize(llm, "You are a tool-using assistant.\n") + [1] * max(0, P_START - 8)
    preceding = preceding[:P_START]
    query = tokenize(llm, "Query: perform task\nSelected Tool Name:")

    work = Path("results/quant_fidelity_work")
    work.mkdir(parents=True, exist_ok=True)
    records, storage_rows = [], []

    for i, tool in enumerate(tools):
        src = Path(args.atb_dir) / f"tool_{i}.isolated.atb"
        if not src.exists():
            print(f"Skip missing {src}")
            continue
        schema_tokens = tokenize(llm, schema_text(tool))
        nexus_fsm_ext.clear_kv_cache(ctx)
        if preceding:
            nexus_fsm_ext.decode_tokens(ctx, preceding, 0, 0)
        ref_logits = np.array(
            nexus_fsm_ext.decode_tokens(ctx, schema_tokens + query, P_START, 0),
            dtype=np.float32,
        )
        plan = plan_suffix_recompute("tail", len(schema_tokens), args.suffix_pct)

        # FP16 splice baseline (same repair path, isolates quant delta).
        nexus_fsm_ext.clear_kv_cache(ctx)
        if preceding:
            nexus_fsm_ext.decode_tokens(ctx, preceding, 0, 0)
        fp16_handle = cache.get_or_load(str(src))
        nexus_fsm_ext.inject_tool_page(ctx, fp16_handle, P_START, 0)
        recompute_plan(ctx, schema_tokens, P_START, plan)
        fp16_splice_logits = np.array(
            nexus_fsm_ext.decode_tokens(ctx, query, P_START + len(schema_tokens), 0),
            dtype=np.float32,
        )
        kl_splice_error = kl_divergence(ref_logits, fp16_splice_logits)

        for quant in quants:
            if quant == "fp16":
                atb_path = src
            else:
                atb_path = work / f"tool_{i}.{quant}.atb"
                if not atb_path.exists() or args.force:
                    make_quantized_atb(src, atb_path, quant)
            hdr, _, _ = read_kv_fp16(atb_path if quant != "fp16" else src)
            storage_rows.append({
                "tool": tool["name"],
                "quant": quant,
                "file_bytes": atb_path.stat().st_size,
                "kv_elements": hdr["k_total_bytes"] // 2 + hdr["v_total_bytes"] // 2,
            })

            nexus_fsm_ext.clear_kv_cache(ctx)
            if preceding:
                nexus_fsm_ext.decode_tokens(ctx, preceding, 0, 0)
            handle = cache.get_or_load(str(atb_path))
            nexus_fsm_ext.inject_tool_page(ctx, handle, P_START, 0)
            recompute_plan(ctx, schema_tokens, P_START, plan)
            splice_logits = np.array(
                nexus_fsm_ext.decode_tokens(ctx, query, P_START + len(schema_tokens), 0),
                dtype=np.float32,
            )
            kl = kl_divergence(ref_logits, splice_logits)
            kl_vs_fp16 = kl_divergence(fp16_splice_logits, splice_logits)
            records.append({
                "tool": tool["name"],
                "quant": quant,
                "suffix_pct": args.suffix_pct,
                "actual_recompute_tokens": plan.actual_recompute_tokens,
                "kl_ref_to_splice": kl,
                "kl_splice_error_fp16": kl_splice_error,
                "kl_quant_vs_fp16_splice": kl_vs_fp16,
                "top1_agreement": int(np.argmax(ref_logits)) == int(np.argmax(splice_logits)),
            })
            print(
                f"  {tool['name']} {quant}: KL={kl:.6f} splice_err={kl_splice_error:.6f} "
                f"quant_delta={kl_vs_fp16:.6f} top1={records[-1]['top1_agreement']}"
            )

    per_tool_mb = {}
    for quant in quants:
        rows = [r for r in storage_rows if r["quant"] == quant]
        if rows:
            per_tool_mb[quant] = float(np.mean([r["file_bytes"] for r in rows])) / 1e6

    economics = {
        "per_tool_mb": per_tool_mb,
        "total_1e3_tools_gb": {q: per_tool_mb.get(q, 0) * 1e3 / 1e3 for q in quants},
        "total_1e4_tools_gb": {q: per_tool_mb.get(q, 0) * 1e4 / 1e3 for q in quants},
        "total_1e5_tools_tb": {q: per_tool_mb.get(q, 0) * 1e5 / 1e6 for q in quants},
        "note": "FP16 runtime splice; q8/q4 measured via dequant->temp FP16 ATB roundtrip",
    }

    out_path = write_artifact(
        args.output,
        "bench_quant_fidelity",
        model_hash=file_sha256(args.model),
        config={"model_path": args.model, "quants": quants, "suffix_pct": args.suffix_pct, "n_tools": len(records) // max(len(quants), 1)},
        metrics={"kl_ref_to_splice": {"unit": "nats", "raw_samples": [r["kl_ref_to_splice"] for r in records]}},
        records=records,
        extra={"storage_economics": economics, "storage_rows": storage_rows},
        smoke=args.smoke,
        force=args.force,
    )
    print(f"Wrote {out_path}")
    print(json.dumps(economics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
