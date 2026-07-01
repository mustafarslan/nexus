#!/usr/bin/env python3
"""D_KL vs. placement-offset sweep (referee item P0-3).

Measures the *raw* off-anchor fidelity drift of a spliced schema block as a function of
placement offset Delta_pos, with recompute DISABLED (bare splice, no R-repair). This is the
curve that justifies the MAX_SPLICE_POS=256 threshold: it shows where the next-token
distribution departs from the compile anchor before Eq.(2)'s depth-adaptive repair engages.

The compiled block tool_0.isolated.atb has base_pos=0 (RoPE anchor at position 0), so the
injection position equals Delta_pos directly. For each offset we prefill a preceding context of
that length, inject the block bare at that position, decode the query one step, and compute
D_KL(p_prefill || p_splice) and top-1 agreement against a full-prefill reference at the same
depth. Several queries per offset give a per-offset sample for bootstrap CIs.

Reuses the exact fidelity machinery of bench_v2_capstone.py (reference_logits, kl_ref_to_splice,
inject_tool_page). Bare splice = we simply never call recompute_tail.

Run: python test/bench_dkl_sweep.py [--offsets "0 64 128 192 256 320 384 512 768 1024"] [--queries 12]
Skips if the chat model / compiled ATB are absent.
"""
from __future__ import annotations
import argparse, os, sys, json, platform, subprocess, datetime as _dt
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.insert(0, os.path.dirname(__file__))
lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../build/external/llama.cpp/src"))
os.environ["LLAMA_CPP_LIB_PATH"] = lib_dir
os.environ["LLAMA_CPP_LIB"] = os.path.join(lib_dir, "libllama.dylib")

import llama_cpp
import nexus_fsm_ext
from bench_phaseB_selective_recompute import (DEFAULT_MODEL, tokenize, build_preceding_tokens,
    schema_text_for, format_query, reference_logits, kl_ref_to_splice)
from bench_routing_accuracy import load_first_10_tools, queries_dataset


def _boot_ci(xs, iters=10000):
    """Deterministic percentile bootstrap 95% CI of the mean (fixed seed)."""
    a = np.asarray(xs, dtype=np.float64)
    if a.size < 2:
        return [float(a.min()) if a.size else 0.0, float(a.max()) if a.size else 0.0]
    rng = np.random.default_rng(20260701)
    means = [float(np.mean(rng.choice(a, size=a.size, replace=True))) for _ in range(iters)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _capture(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--offsets", default="0 64 128 192 256 320 384 512 768 1024")
    ap.add_argument("--queries", type=int, default=12, help="queries averaged per offset")
    ap.add_argument("--atb", default="results/phaseA_tool_match_work/tool_0.isolated.atb")
    ap.add_argument("--output", default="results/v2.0_canonical/raw/dkl_sweep.json")
    args = ap.parse_args()
    if not Path(args.model).exists():
        print(f"SKIP: model not found {args.model}"); return
    atb = Path(args.atb)
    if not atb.exists():
        print(f"SKIP: missing compiled ATB {atb}"); return

    offsets = [int(x) for x in args.offsets.split()]
    llm = llama_cpp.Llama(model_path=args.model, n_ctx=4096, n_batch=2048, n_ubatch=2048,
                          n_gpu_layers=999, flash_attn=True, verbose=False)
    ctx = llm._ctx.ctx
    cache = nexus_fsm_ext.NexusBlockCache(2 * 1024 * 1024 * 1024)  # must outlive `handle`
    handle = cache.get_or_load(str(atb))

    tool = load_first_10_tools()[0]
    st = tokenize(llm, schema_text_for(tool))
    queries = [q["query"] for q in queries_dataset[:args.queries]]
    qts = [tokenize(llm, format_query(q)) for q in queries]

    print(f"\nD_KL vs placement offset (bare splice, base_pos=0) | schema_len={len(st)} "
          f"| queries/offset={len(qts)} | model={Path(args.model).name}")
    print(f"{'offset':>7} {'dkl_mean':>10} {'dkl_ci95':>22} {'top1':>6} {'kl_max':>9}")
    print("-" * 62)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    def write(rows):
        """Atomic incremental write so a crash never loses completed offsets."""
        artifact = {
            "schema_version": 1,
            "benchmark": "bench_dkl_placement_offset_sweep",
            "description": "Bare-splice (recompute disabled) next-token D_KL vs placement offset; "
                           "block base_pos=0 so injection position == Delta_pos. Justifies MAX_SPLICE_POS=256.",
            "git_sha": _capture(["git", "rev-parse", "HEAD"]),
            "llama_cpp_sha": _capture(["git", "-C", "external/llama.cpp", "rev-parse", "HEAD"]),
            "hardware": f"Apple M4 Max, {platform.system()} {platform.release()} {platform.machine()} (UMA/Metal)",
            "timestamp": _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "config": {"model": Path(args.model).name, "atb": atb.name, "base_pos": 0,
                       "schema_len": len(st), "queries_per_offset": len(qts),
                       "max_splice_pos": 256, "recompute": "disabled (bare splice)"},
            "complete": len(rows) == len(offsets),
            "offset": [r["offset"] for r in rows],
            "dkl_mean": [r["dkl_mean"] for r in rows],
            "dkl_ci95": [r["dkl_ci95"] for r in rows],
            "top1": [r["top1_agree"] for r in rows],
            "curve": rows,
        }
        tmp = str(args.output) + ".tmp"
        Path(tmp).write_text(json.dumps(artifact, indent=2) + "\n")
        os.replace(tmp, args.output)

    rows = []
    for off in offsets:
        kls, top1s = [], []
        for qt in qts:
            pre = build_preceding_tokens(llm, "You are a tool-using assistant.", off)
            ref = reference_logits(ctx, pre, st, qt)             # full-prefill reference at depth
            nexus_fsm_ext.clear_kv_cache(ctx)
            if pre:
                nexus_fsm_ext.decode_tokens(ctx, pre, 0, 0)
            nexus_fsm_ext.inject_tool_page(ctx, handle, len(pre), 0)  # BARE: no recompute_tail
            spl = np.array(nexus_fsm_ext.decode_tokens(ctx, qt, len(pre) + len(st), 0), dtype=np.float32)
            kls.append(float(kl_ref_to_splice(ref, spl)))
            top1s.append(int(int(np.argmax(ref)) == int(np.argmax(spl))))
        dkl_mean = float(np.mean(kls))
        ci = _boot_ci(kls)
        top1 = float(np.mean(top1s))
        rows.append({"offset": off, "dkl_mean": dkl_mean, "dkl_ci95": ci,
                     "top1_agree": top1, "kl_max": float(np.max(kls)), "kl_raw": kls, "n": len(kls)})
        print(f"{off:>7} {dkl_mean:>10.4f} [{ci[0]:>8.4f},{ci[1]:>8.4f}] {top1:>6.2f} {np.max(kls):>9.4f}", flush=True)
        write(rows)  # checkpoint after every offset

    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
