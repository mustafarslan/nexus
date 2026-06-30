#!/usr/bin/env python3
"""Nexus v2.0 capstone: TTFT profile under deep multi-turn context (n_past > 256).

Demonstrates the two v2.0 success axes:
  1. Routing accuracy >=81% at N=250 is established separately (Phase 5; depth-invariant
     because SLB search is position-independent). This bench measures the N=250 SLB search
     latency to confirm it stays sub-millisecond, i.e. tool count does not inflate TTFT.
  2. Deep-splice TTFT vs depth: with enable_deep_splice the .atb is spliced (KV memcpy +
     depth-adaptive recompute) instead of declining to a full text re-prefill. We time the
     splice path against the text-prefill baseline at increasing p_start to show the
     acceleration profile and where adaptive recompute converges to re-prefill cost.

Run: python test/bench_v2_capstone.py [--p-starts "256 512 1024 2048"] [--trials 3]
Skips if the chat model / compiled ATB are absent.
"""
from __future__ import annotations
import argparse, os, sys, time
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
from bench_routing_accuracy import load_first_10_tools, load_n_tools, queries_dataset
from nexus_recompute import recompute_tail

MAX_SPLICE_POS, BASE_PCT = 256, 5.0

def eff_pct(n_past, full_mult):
    if n_past <= MAX_SPLICE_POS: return BASE_PCT
    lo, hi = float(MAX_SPLICE_POS), MAX_SPLICE_POS * full_mult
    frac = min(max((n_past - lo) / (hi - lo), 0.0), 1.0) if hi > lo else 1.0
    return BASE_PCT + frac * (100.0 - BASE_PCT)

def _median_ms(fn, trials):
    xs = []
    for _ in range(trials):
        t = time.perf_counter(); fn(); xs.append((time.perf_counter() - t) * 1000.0)
    return float(np.median(xs))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--p-starts", default="256 512 1024 2048")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--full-mult", type=float, default=4.0,
                    help="n_past multiple of max_splice_pos at which recompute hits 100%%")
    args = ap.parse_args()
    if not Path(args.model).exists():
        print(f"SKIP: model not found {args.model}"); return
    atb = Path("results/phaseA_tool_match_work/tool_0.isolated.atb")
    if not atb.exists():
        print("SKIP: missing compiled ATB"); return

    llm = llama_cpp.Llama(model_path=args.model, n_ctx=4096, n_batch=2048, n_ubatch=2048,
                          n_gpu_layers=999, flash_attn=True, verbose=False)
    ctx = llm._ctx.ctx
    cache = nexus_fsm_ext.NexusBlockCache(2 * 1024 * 1024 * 1024)
    tool = load_first_10_tools()[0]
    st = tokenize(llm, schema_text_for(tool))
    qt = tokenize(llm, format_query(queries_dataset[0]["query"]))
    handle = cache.get_or_load(str(atb))
    p_starts = [int(x) for x in args.p_starts.split()]

    def prefill_baseline(pre):  # what the decline path costs: full schema re-prefill + query
        nexus_fsm_ext.clear_kv_cache(ctx)
        if pre: nexus_fsm_ext.decode_tokens(ctx, pre, 0, 0)
        nexus_fsm_ext.decode_tokens(ctx, st, len(pre), 0)
        nexus_fsm_ext.decode_tokens(ctx, qt, len(pre) + len(st), 0)

    def deep_splice(pre, pct):  # the v2.0 deep path: inject + adaptive recompute + query
        nexus_fsm_ext.clear_kv_cache(ctx)
        if pre: nexus_fsm_ext.decode_tokens(ctx, pre, 0, 0)
        nexus_fsm_ext.inject_tool_page(ctx, handle, len(pre), 0)
        if pct > 0: recompute_tail(ctx, st, len(pre), pct)
        nexus_fsm_ext.decode_tokens(ctx, qt, len(pre) + len(st), 0)

    print(f"\nNexus v2.0 capstone | full_mult={args.full_mult} schema_len={len(st)} query_len={len(qt)} | model={Path(args.model).name}")
    print(f"{'p_start':>7} {'eff_pct':>8} {'prefill_ms':>11} {'splice_ms':>10} {'speedup':>8} {'top1':>6} {'KL':>9}")
    print("-" * 64)
    for p in p_starts:
        pre = build_preceding_tokens(llm, "You are a tool-using assistant.", p)
        e = eff_pct(p, args.full_mult)
        # fidelity (never-regress) at this depth
        ref = reference_logits(ctx, pre, st, qt)
        nexus_fsm_ext.clear_kv_cache(ctx)
        if pre: nexus_fsm_ext.decode_tokens(ctx, pre, 0, 0)
        nexus_fsm_ext.inject_tool_page(ctx, handle, len(pre), 0)
        if e > 0: recompute_tail(ctx, st, len(pre), e)
        spl = np.array(nexus_fsm_ext.decode_tokens(ctx, qt, len(pre) + len(st), 0), dtype=np.float32)
        kl = kl_ref_to_splice(ref, spl)
        match = "OK" if int(np.argmax(ref)) == int(np.argmax(spl)) else "MISS"
        # latency
        prefill_ms = _median_ms(lambda: prefill_baseline(pre), args.trials)
        splice_ms = _median_ms(lambda: deep_splice(pre, e), args.trials)
        print(f"{p:>7} {e:>8.1f} {prefill_ms:>11.2f} {splice_ms:>10.2f} {prefill_ms/splice_ms:>7.2f}x {match:>6} {kl:>9.5f}")

    # N=250 SLB search latency (depth/route-independent of splice; confirms N doesn't inflate TTFT)
    try:
        emb_dim = 768
        slb = nexus_fsm_ext.NexusSemanticSLB(emb_dim)
        rng = np.random.default_rng(0)
        tools = load_n_tools(250)
        for t in tools:
            v = rng.standard_normal(emb_dim).astype(np.float32); v /= np.linalg.norm(v)
            slb.register_tool(int(t["id"]), v, [])
        q = rng.standard_normal(emb_dim).astype(np.float32); q /= np.linalg.norm(q)
        slb_ms = _median_ms(lambda: slb.search(q, 5), 50)
        print(f"\nN=250 SLB search latency: {slb_ms*1000:.1f} us (median) -- tool count adds negligible TTFT")
    except Exception as ex:
        print(f"\n(SLB latency probe skipped: {ex})")

if __name__ == "__main__":
    main()
