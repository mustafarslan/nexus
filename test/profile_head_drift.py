#!/usr/bin/env python3
"""Phase 2 Component B + go/no-go: per-head splice drift calibration & proxy correlation.

TRUE per-head drift (needs reference, OK offline):
    drift[il,h] = || K_true_in_context[:, h, :] - K_reanchored_splice[:, h, :] ||_2
REFERENCE-FREE proxy (what the runtime micro-gate could see):
    livevar[il,h] = variance of the live PRECEDING-context K for that head
    leverage(Delta) = analytical RoPE long-range sensitivity (constant across heads per sample)

GO/NO-GO: if per-head livevar does NOT rank-correlate with true drift, the online micro-gate
cannot predict catastrophic divergence -> Components C/D are NOT shipped (degrade to Phase 1).
Also reports whether catastrophic drift even occurs (if drift is uniformly low, the gate has
little to do for this workload).
"""
import os, sys
sys.path.insert(0, os.path.abspath("build")); sys.path.insert(0, os.path.abspath("src")); sys.path.insert(0, os.path.abspath("test"))
lib_dir = os.path.abspath("build/external/llama.cpp/src")
os.environ["LLAMA_CPP_LIB_PATH"]=lib_dir; os.environ["LLAMA_CPP_LIB"]=os.path.join(lib_dir,"libllama.dylib")
import numpy as np, llama_cpp, nexus_fsm_ext
from pathlib import Path
from bench_phaseB_selective_recompute import (DEFAULT_MODEL, tokenize, build_preceding_tokens,
    schema_text_for, format_query, reference_logits)
from bench_routing_accuracy import load_first_10_tools, queries_dataset

ATB = Path("results/phaseA_tool_match_work/tool_0.isolated.atb")

def read_head(ctx, il, p0, p1):
    flat, seq, nh, dh = nexus_fsm_ext.read_kv_slice(ctx, il, p0, p1, False)
    return np.array(flat, dtype=np.float32).reshape(seq, nh, dh)

def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0: return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])

def main():
    if not ATB.exists():
        print("SKIP: missing ATB"); return
    llm = llama_cpp.Llama(model_path=DEFAULT_MODEL, n_ctx=4096, n_batch=2048, n_ubatch=2048,
                          n_gpu_layers=999, flash_attn=True, verbose=False)
    ctx = llm._ctx.ctx
    cache = nexus_fsm_ext.NexusBlockCache(2*1024*1024*1024)
    handle = cache.get_or_load(str(ATB))
    tool = load_first_10_tools()[0]
    st = tokenize(llm, schema_text_for(tool)); slen = len(st)
    n_layer = int(llm.metadata.get("llama.block_count", 48))
    layers = list(range(1, n_layer, max(1, n_layer // 8)))  # sample ~8 layers
    contexts = ["You are a tool-using assistant.",
                "Discuss quantum chromodynamics and lattice gauge theory in depth."]  # on/off-topic

    print(f"schema_len={slen} n_layer={n_layer} sampled_layers={layers}\n")
    print(f"{'p_start':>7} {'ctx':>4} {'maxDrift':>9} {'meanDrift':>9} {'spearman(livevar,drift)':>24}")
    print("-"*60)
    all_corr = []
    for p in [256, 1024, 2048]:
        for ci, cprompt in enumerate(contexts):
            pre = build_preceding_tokens(llm, cprompt, p)
            true_h, splice_h, live_h = [], [], []
            for il in layers:
                # true in-context K for schema region
                nexus_fsm_ext.clear_kv_cache(ctx)
                nexus_fsm_ext.decode_tokens(ctx, pre, 0, 0)
                nexus_fsm_ext.decode_tokens(ctx, st, len(pre), 0)
                k_true = read_head(ctx, il, len(pre), len(pre)+slen)           # (slen, nh, dh)
                live = read_head(ctx, il, 0, len(pre))                         # preceding region
                # spliced (reanchored) K for schema region
                nexus_fsm_ext.clear_kv_cache(ctx)
                nexus_fsm_ext.decode_tokens(ctx, pre, 0, 0)
                nexus_fsm_ext.inject_tool_page(ctx, handle, len(pre), 0)
                k_splice = read_head(ctx, il, len(pre), len(pre)+slen)
                nh = k_true.shape[1]
                for h in range(nh):
                    true_h.append(np.linalg.norm(k_true[:, h, :] - k_splice[:, h, :]))  # per-head drift
                    live_h.append(np.var(live[:, h, :]))                       # proxy: live variance
            drift = np.array(true_h); proxy = np.array(live_h)
            # normalize drift by per-head true magnitude to get relative drift
            corr = spearman(proxy, drift)
            all_corr.append(corr)
            print(f"{p:>7} {ci:>4} {drift.max():>9.3f} {drift.mean():>9.3f} {corr:>24.3f}")
    mc = float(np.mean(all_corr))
    print(f"\nmean Spearman(proxy livevar, true drift) = {mc:.3f}")
    print("GO/NO-GO:", "GO (proxy predicts drift; wire C++ micro-gate)" if mc >= 0.4
          else "NO-GO (proxy does not predict drift; ship Component A only, keep Phase 1 depth curve)")

if __name__ == "__main__":
    main()
