#!/usr/bin/env python3
"""Nexus v2.0 capstone: TTFT profile under deep multi-turn context (n_past > 256).

Demonstrates the two v2.0 success axes:
  1. Deep-splice TTFT vs depth: with enable_deep_splice the .atb is spliced (KV memcpy +
     depth-adaptive recompute) instead of declining to a full text re-prefill. We time the
     splice path against the text-prefill baseline at increasing p_start to show the
     acceleration profile and where adaptive recompute converges to re-prefill cost.
  2. Never-regress fidelity: per depth we record next-token top-1 agreement and Logit-KL of
     the spliced distribution vs the recompute reference, confirming accuracy never regresses.

This version writes a committed JSON artifact (median + bootstrap 95% CI over trials) so the
paper's Table I traces to a reproducible file, not console output.

Run: python test/bench_v2_capstone.py [--p-starts "256 512 1024 2048"] [--full-mults "4 16"] [--trials 15]
Skips if the chat model / compiled ATB are absent.
"""
from __future__ import annotations
import argparse, os, sys, time, json, platform, subprocess, datetime as _dt
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

def _samples_ms(fn, trials):
    xs = []
    for _ in range(trials):
        t = time.perf_counter(); fn(); xs.append((time.perf_counter() - t) * 1000.0)
    return xs

def _boot_ci(xs, iters=2000):
    """Deterministic bootstrap 95% CI of the median (fixed-seed RNG; no Math.random dependency)."""
    a = np.asarray(xs, dtype=np.float64)
    if a.size < 2:
        return [float(a.min()) if a.size else 0.0, float(a.max()) if a.size else 0.0]
    rng = np.random.default_rng(20260701)
    meds = [float(np.median(rng.choice(a, size=a.size, replace=True))) for _ in range(iters)]
    return [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]

def _capture(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--p-starts", default="256 512 1024 2048")
    ap.add_argument("--full-mults", default="4 16", help="recompute_full_mult curves to sweep (K)")
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--output", default="results/v2.0_canonical/raw/deep_splice_ttft.json")
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
    full_mults = [float(x) for x in args.full_mults.split()]

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

    curves = {}
    print(f"\nNexus v2.0 capstone | schema_len={len(st)} query_len={len(qt)} trials={args.trials} | model={Path(args.model).name}")
    for fm in full_mults:
        rows = []
        print(f"\n== curve K(full_mult)={fm:g} ==")
        print(f"{'p_start':>7} {'eff_pct':>8} {'prefill_ms':>11} {'splice_ms':>10} {'speedup':>8} {'top1':>6} {'KL':>9}")
        print("-" * 64)
        for p in p_starts:
            pre = build_preceding_tokens(llm, "You are a tool-using assistant.", p)
            e = eff_pct(p, fm)
            # fidelity (never-regress) at this depth
            ref = reference_logits(ctx, pre, st, qt)
            nexus_fsm_ext.clear_kv_cache(ctx)
            if pre: nexus_fsm_ext.decode_tokens(ctx, pre, 0, 0)
            nexus_fsm_ext.inject_tool_page(ctx, handle, len(pre), 0)
            if e > 0: recompute_tail(ctx, st, len(pre), e)
            spl = np.array(nexus_fsm_ext.decode_tokens(ctx, qt, len(pre) + len(st), 0), dtype=np.float32)
            kl = float(kl_ref_to_splice(ref, spl))
            match = int(int(np.argmax(ref)) == int(np.argmax(spl)))
            # latency
            pf = _samples_ms(lambda: prefill_baseline(pre), args.trials)
            sp = _samples_ms(lambda: deep_splice(pre, e), args.trials)
            pf_med, sp_med = float(np.median(pf)), float(np.median(sp))
            rows.append({
                "p_start": p, "eff_pct": round(e, 1),
                "prefill_ms_median": pf_med, "prefill_ms_ci95": _boot_ci(pf),
                "splice_ms_median": sp_med, "splice_ms_ci95": _boot_ci(sp),
                "speedup": pf_med / sp_med if sp_med else 0.0,
                "top1_agree": match, "kl_nats": kl,
            })
            print(f"{p:>7} {e:>8.1f} {pf_med:>11.1f} {sp_med:>10.1f} {pf_med/sp_med:>7.2f}x {'OK' if match else 'MISS':>6} {kl:>9.5f}")
        curves[f"full_mult_{fm:g}"] = rows

    artifact = {
        "schema_version": 1,
        "benchmark": "bench_v2_capstone_deep_splice_ttft",
        "git_sha": _capture(["git", "rev-parse", "HEAD"]),
        "llama_cpp_sha": _capture(["git", "-C", "external/llama.cpp", "rev-parse", "HEAD"]),
        "hardware": f"{platform.system()} {platform.release()} {platform.machine()} (UMA/Metal)",
        "timestamp": _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "config": {"model": Path(args.model).name, "trials": args.trials, "schema_len": len(st),
                   "query_len": len(qt), "p_starts": p_starts, "full_mults": full_mults,
                   "max_splice_pos": MAX_SPLICE_POS, "base_recompute_pct": BASE_PCT},
        "curves": curves,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"\nWrote {args.output}")

    # N=250 SLB search latency (synthetic random unit vectors; pure timing of the INT8 scan,
    # NOT an accuracy measurement -- labelled synthetic in the paper).
    try:
        emb_dim = 768
        slb = nexus_fsm_ext.NexusSemanticSLB(emb_dim)
        rng = np.random.default_rng(0)
        tools = load_n_tools(250)
        for t in tools:
            v = rng.standard_normal(emb_dim).astype(np.float32); v /= np.linalg.norm(v)
            slb.register_tool(int(t["id"]), v, [])
        q = rng.standard_normal(emb_dim).astype(np.float32); q /= np.linalg.norm(q)
        xs = _samples_ms(lambda: slb.search(q, 5), 200)
        slb_us = float(np.median(xs)) * 1000.0
        (Path(args.output).parent / "slb_latency.json").write_text(json.dumps({
            "benchmark": "slb_search_latency_synthetic", "n_tools": 250, "emb_dim": emb_dim,
            "trials": 200, "median_us": slb_us, "ci95_us": [c * 1000.0 for c in _boot_ci(xs)],
            "note": "synthetic unit-norm random vectors; pure INT8 SIMD scan timing, not accuracy",
            "git_sha": artifact["git_sha"], "timestamp": artifact["timestamp"],
        }, indent=2) + "\n")
        print(f"N=250 SLB search latency: {slb_us:.1f} us (median, synthetic)")
    except Exception as ex:
        print(f"(SLB latency probe skipped: {ex})")

if __name__ == "__main__":
    main()
