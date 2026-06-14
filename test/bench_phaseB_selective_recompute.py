#!/usr/bin/env python3
"""Phase B-ext: principled selective recompute with HKVD/oracle ablations.

Selector ablation matrix:
  selection in {tail, front, random, hkvd_layer1, oracle_full_deviation}
  r in {0, 2, 5, 10, 20}
  P_start in {256, 1024}

Latency timer = splice + recompute + query decode only (preceding context pre-cached).
Anchors: full schema+query prefill, r=100% schema reprefill after splice.
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
from nexus_recompute import contiguous_runs, recompute_selected  # noqa: E402

DEFAULT_MODEL = "/Volumes/AI_SSD/models/Qwen2.5-14B-Instruct-GGUF/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf"
DEFAULT_SYSTEM_PROMPT = "You are a tool-using assistant. Use the provided tool schema to answer the user."
SELECTORS = ["tail", "front", "random", "hkvd_layer1", "oracle_full_deviation"]
DEFAULT_RECOMPUTE_PCTS = [0, 2, 5, 10, 15, 20, 30, 100]
DEFAULT_P_STARTS = [256, 1024, 4096]

_FILLER_TURNS = [
    "User: I'm building a small data pipeline and I keep running into timezone bugs. Any general advice?",
    "Assistant: Timezone bugs usually come from mixing naive and aware datetimes. Store everything in UTC, convert only at the display boundary, and never rely on the server's local clock for business logic.",
    "User: Makes sense. We also have a flaky integration test that fails about one run in twenty.",
    "Assistant: Flaky tests are often hidden ordering or timing assumptions. Try seeding randomness, pinning clocks, and isolating shared state between cases before reaching for retries.",
    "User: Good point. Separately, our log volume exploded last week and storage costs spiked.",
    "Assistant: Audit log levels first: a lot of teams ship debug logging to production by accident. Sampling high-frequency events and dropping redundant fields usually recovers most of the cost.",
    "User: We're also debating whether to add a cache in front of the database.",
    "Assistant: Caching helps read-heavy workloads but adds invalidation complexity. Measure your actual hit rate and tail latency before committing, and prefer a small TTL over manual invalidation when you can.",
    "User: Last thing, the team wants to standardize how we handle background jobs.",
    "Assistant: Pick one queue, make jobs idempotent, and always record an explicit status. Idempotency is what lets you retry safely when a worker dies mid-task.",
]


def tokenize(llm: llama_cpp.Llama, text: str) -> list[int]:
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


def build_preceding_tokens(llm: llama_cpp.Llama, system_prompt: str, target_len: int) -> list[int]:
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
        "that matches the user's intent. You MUST output ONLY the tool name, with no "
        "other text, punctuation, explanation, or markdown.\n\n"
        f"Query: {user_query}\n"
        "Selected Tool Name:"
    )


def cache_preceding(ctx, preceding: list[int]) -> None:
    nexus_fsm_ext.clear_kv_cache(ctx)
    if preceding:
        nexus_fsm_ext.decode_tokens(ctx, preceding, 0, 0)


def reference_logits(ctx, preceding: list[int], schema_tokens: list[int], query_tokens: list[int]) -> np.ndarray:
    cache_preceding(ctx, preceding)
    tokens = schema_tokens + query_tokens
    start = len(preceding)
    return np.array(nexus_fsm_ext.decode_tokens(ctx, tokens, start, 0), dtype=np.float32)


def token_deviation_l2(true_k: np.ndarray, splice_k: np.ndarray) -> np.ndarray:
    """Per-token L2 over heads*dim. Inputs shape (seq, n_head, d_head)."""
    diff = true_k - splice_k
    return np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1)


def read_kv_slice(ctx, layer: int, p0: int, p1: int, read_v: bool = False) -> np.ndarray:
    flat, seq_len, n_head_kv, d_head = nexus_fsm_ext.read_kv_slice(ctx, layer, p0, p1, read_v)
    return np.array(flat, dtype=np.float32).reshape(seq_len, n_head_kv, d_head)


def compute_deviations_with_preceding(
    ctx,
    cache,
    atb_path: Path,
    preceding: list[int],
    schema_tokens: list[int],
    p_start: int,
    schema_len: int,
    n_layer: int,
) -> tuple[np.ndarray, np.ndarray]:
    cache_preceding(ctx, preceding)
    nexus_fsm_ext.decode_tokens(ctx, schema_tokens, p_start, 0)
    true_by_layer: dict[int, np.ndarray] = {}
    for il in range(n_layer):
        true_by_layer[il] = read_kv_slice(ctx, il, p_start, p_start + schema_len)

    cache_preceding(ctx, preceding)
    handle = cache.get_or_load(str(atb_path))
    nexus_fsm_ext.inject_tool_page(ctx, handle, p_start, 0)

    layer1_dev = token_deviation_l2(
        true_by_layer[1],
        read_kv_slice(ctx, 1, p_start, p_start + schema_len),
    )

    oracle_dev = np.zeros(schema_len, dtype=np.float64)
    for il in range(n_layer):
        splice_k = read_kv_slice(ctx, il, p_start, p_start + schema_len)
        oracle_dev += token_deviation_l2(true_by_layer[il], splice_k)

    return layer1_dev, oracle_dev


def n_select(schema_len: int, recompute_pct: float) -> int:
    if recompute_pct <= 0 or schema_len <= 0:
        return 0
    return max(1, int(math.ceil(schema_len * recompute_pct / 100.0)))


def select_token_indices(
    selector: str,
    schema_len: int,
    recompute_pct: float,
    layer1_dev: np.ndarray,
    oracle_dev: np.ndarray,
    rng: np.random.Generator,
) -> list[int]:
    n = n_select(schema_len, recompute_pct)
    if n == 0:
        return []

    if selector == "tail":
        return list(range(schema_len - n, schema_len))
    if selector == "front":
        return list(range(n))
    if selector == "random":
        return sorted(rng.choice(schema_len, size=n, replace=False).tolist())
    if selector == "hkvd_layer1":
        order = np.argsort(-layer1_dev)
        return sorted(order[:n].tolist())
    if selector == "oracle_full_deviation":
        order = np.argsort(-oracle_dev)
        return sorted(order[:n].tolist())
    raise ValueError(f"unknown selector: {selector}")


def splice_recompute_query(
    ctx,
    cache,
    atb_path: Path,
    preceding: list[int],
    schema_tokens: list[int],
    query_tokens: list[int],
    p_start: int,
    selected: list[int],
) -> tuple[np.ndarray, float]:
    """Timer boundary: splice + recompute + query decode (preceding pre-cached)."""
    cache_preceding(ctx, preceding)
    t0 = time.perf_counter()
    handle = cache.get_or_load(str(atb_path))
    nexus_fsm_ext.inject_tool_page(ctx, handle, p_start, 0)
    if selected:
        recompute_selected(ctx, schema_tokens, p_start, selected)
    logits = np.array(
        nexus_fsm_ext.decode_tokens(ctx, query_tokens, p_start + len(schema_tokens), 0),
        dtype=np.float32,
    )
    elapsed_us = (time.perf_counter() - t0) * 1e6
    return logits, elapsed_us


def anchor_full_prefill_latency(
    ctx,
    preceding: list[int],
    schema_tokens: list[int],
    query_tokens: list[int],
) -> float:
    cache_preceding(ctx, preceding)
    t0 = time.perf_counter()
    start = len(preceding)
    nexus_fsm_ext.decode_tokens(ctx, schema_tokens + query_tokens, start, 0)
    return (time.perf_counter() - t0) * 1e6


def anchor_reprefill_latency(
    ctx,
    preceding: list[int],
    schema_tokens: list[int],
    query_tokens: list[int],
    p_start: int,
    cache,
    atb_path: Path,
) -> float:
    """r=100%: splice then re-decode entire schema + query."""
    selected = list(range(len(schema_tokens)))
    _, elapsed = splice_recompute_query(
        ctx, cache, atb_path, preceding, schema_tokens, query_tokens, p_start, selected)
    return elapsed


def summarize(values: list[float]) -> dict:
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


def parse_args():
    p = argparse.ArgumentParser(description="Phase B-ext selective recompute ablation.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--p-starts", default="256,1024")
    p.add_argument("--selectors", default=",".join(SELECTORS))
    p.add_argument("--recompute-pcts", default="0,2,5,10,20")
    p.add_argument("--case-limit", type=int, default=50)
    p.add_argument("--output", default="results/bench_phaseB_selective_recompute.json")
    p.add_argument("--workdir", default="results/phaseB_recompute_work")
    p.add_argument("--atb-workdir", default="results/phaseA_tool_match_work")
    p.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    p.add_argument("--n-ctx", type=int, default=8192)
    p.add_argument("--n-gpu-layers", type=int, default=999)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--quick", action="store_true", help="Run reduced matrix: P=256, tail+hkvd, r in {0,5,10,100}.")
    p.add_argument("--full", action="store_true", help="Full G2 matrix: all selectors, P and r through 30%% + 100%% anchor.")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    recompute_pcts = [float(x) for x in args.recompute_pcts.split(",") if x.strip()]
    p_starts = [int(x) for x in args.p_starts.split(",") if x.strip()]
    selectors = [x.strip() for x in args.selectors.split(",") if x.strip()]
    if args.quick:
        p_starts = [256]
        selectors = ["tail", "hkvd_layer1"]
        recompute_pcts = [0, 5, 10, 100]
    elif not args.full:
        p_starts = [256, 1024]
        recompute_pcts = [0, 2, 5, 10, 20, 100]

    tools = load_first_10_tools()
    tool_by_name = {t["name"]: t for t in tools}
    cases = [c for c in queries_dataset if c["tool"] in tool_by_name][: args.case_limit]
    print(f"Phase B-ext: {len(cases)} cases, P={p_starts}, selectors={selectors}, r%={recompute_pcts}")

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    atb_src = Path(args.atb_workdir)

    max_p = max(p_starts)
    n_batch = min(args.n_ctx, max(2048, max_p + 640))
    llm = llama_cpp.Llama(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_batch=n_batch,
        n_ubatch=n_batch,
        n_gpu_layers=args.n_gpu_layers,
        flash_attn=True,
        verbose=False,
    )
    ctx = llm._ctx.ctx
    cache = nexus_fsm_ext.NexusBlockCache(8 * 1024 * 1024 * 1024)
    n_layer = int(llm.metadata.get("llama.block_count", llm.metadata.get("qwen2.block_count", 48)))

    atb_by_tool: dict[str, Path] = {}
    schema_tokens_by_tool: dict[str, list[int]] = {}
    schema_len_by_tool: dict[str, int] = {}
    for i, tool in enumerate(tools):
        text = schema_text_for(tool)
        atb_path = atb_src / f"tool_{i}.isolated.atb"
        if not atb_path.exists():
            schema_path = workdir / f"tool_{i}.schema.json"
            schema_path.write_text(text)
            atb_path = workdir / f"tool_{i}.isolated.atb"
            subprocess.run(
                ["./build/nexus_kv_compiler", "--model", args.model, "--schema", str(schema_path), "--output", str(atb_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        toks = tokenize(llm, text)
        atb_by_tool[tool["name"]] = atb_path
        schema_tokens_by_tool[tool["name"]] = toks
        schema_len_by_tool[tool["name"]] = len(toks)

    preceding_by_pos = {p: build_preceding_tokens(llm, args.system_prompt, p) for p in p_starts}

    records = []
    anchor_records = []
    deviation_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}

    for idx, case in enumerate(cases):
        tool_name = case["tool"]
        schema_tokens = schema_tokens_by_tool[tool_name]
        schema_len = schema_len_by_tool[tool_name]
        query_tokens = tokenize(llm, format_query(case["query"]))
        atb_path = atb_by_tool[tool_name]

        for p_start in p_starts:
            preceding = preceding_by_pos[p_start]
            ref = reference_logits(ctx, preceding, schema_tokens, query_tokens)
            top1_ref = int(np.argmax(ref))

            dev_key = (tool_name, p_start)
            if dev_key not in deviation_cache:
                layer1_dev, oracle_dev = compute_deviations_with_preceding(
                    ctx, cache, atb_path, preceding, schema_tokens, p_start, schema_len, n_layer)
                deviation_cache[dev_key] = (layer1_dev, oracle_dev)
            layer1_dev, oracle_dev = deviation_cache[dev_key]

            full_prefill_us = anchor_full_prefill_latency(ctx, preceding, schema_tokens, query_tokens)
            reprefill_us = anchor_reprefill_latency(
                ctx, preceding, schema_tokens, query_tokens, p_start, cache, atb_path)
            anchor_records.append({
                "case_id": f"{tool_name}-{idx}",
                "tool_name": tool_name,
                "p_start": p_start,
                "anchor_full_prefill_us": full_prefill_us,
                "anchor_reprefill_100pct_us": reprefill_us,
            })

            for selector in selectors:
                for r_pct in recompute_pcts:
                    selected = select_token_indices(
                        selector, schema_len, r_pct, layer1_dev, oracle_dev, rng)
                    splice, latency_us = splice_recompute_query(
                        ctx, cache, atb_path, preceding, schema_tokens, query_tokens,
                        p_start, selected)
                    kl = kl_ref_to_splice(ref, splice)
                    top1_splice = int(np.argmax(splice))
                    records.append({
                        "case_id": f"{tool_name}-{idx}",
                        "tool_name": tool_name,
                        "p_start": p_start,
                        "selector": selector,
                        "recompute_pct": r_pct,
                        "recompute_tokens": len(selected),
                        "schema_tokens": schema_len,
                        "kl_ref_to_splice": kl,
                        "top1_reference": top1_ref,
                        "top1_splice": top1_splice,
                        "top1_agreement": top1_ref == top1_splice,
                        "splice_recompute_us": latency_us,
                    })

        if (idx + 1) % 10 == 0:
            print(f"  {idx + 1}/{len(cases)} cases")

    by_config = {}
    for p_start in p_starts:
        for selector in selectors:
            for r_pct in recompute_pcts:
                key = f"P{p_start}_{selector}_r{r_pct}"
                rows = [
                    r for r in records
                    if r["p_start"] == p_start and r["selector"] == selector and r["recompute_pct"] == r_pct
                ]
                kls = [r["kl_ref_to_splice"] for r in rows]
                lats = [r["splice_recompute_us"] for r in rows]
                agree = [1.0 if r["top1_agreement"] else 0.0 for r in rows]
                by_config[key] = {
                    "p_start": p_start,
                    "selector": selector,
                    "recompute_pct": r_pct,
                    "n": len(rows),
                    "kl_ref_to_splice": summarize(kls),
                    "top1_agreement_rate": float(np.mean(agree)) if agree else 0.0,
                    "splice_recompute_us": summarize(lats),
                }

    anchor_summary = {
        "full_prefill_us": summarize([a["anchor_full_prefill_us"] for a in anchor_records]),
        "reprefill_100pct_us": summarize([a["anchor_reprefill_100pct_us"] for a in anchor_records]),
    }

    r100_rows = [r for r in records if r.get("recompute_pct") == 100.0]
    r100_kls = [r["kl_ref_to_splice"] for r in r100_rows]
    r100_agree = [1.0 if r["top1_agreement"] else 0.0 for r in r100_rows]
    correctness_canary = {
        "r100_mean_kl": float(np.mean(r100_kls)) if r100_kls else None,
        "r100_top1_agreement_rate": float(np.mean(r100_agree)) if r100_agree else None,
        "pass": (
            bool(r100_kls)
            and float(np.mean(r100_kls)) < 0.01
            and float(np.mean(r100_agree)) >= 0.99
        ),
        "note": "Oracle/tail r=100% must match reference prefill logits (canary for recompute path).",
    }

    metrics = {
        "kl_ref_to_splice": {"unit": "nats", "raw_samples": [r["kl_ref_to_splice"] for r in records]},
        "top1_agreement": {"unit": "ratio", "raw_samples": [1.0 if r["top1_agreement"] else 0.0 for r in records]},
        "splice_recompute_us": {"unit": "us", "raw_samples": [r["splice_recompute_us"] for r in records]},
    }

    write_artifact(
        args.output,
        "bench_phaseB_selective_recompute",
        model_hash=file_sha256(args.model) if Path(args.model).exists() else "",
        config={
            "model_path": args.model,
            "p_starts": p_starts,
            "selectors": selectors,
            "recompute_pcts": recompute_pcts,
            "n_cases": len(cases),
            "recompute_strategy": "HKVD/oracle deviation + position heuristics; contiguous-run invalidate+decode",
            "latency_note": "Timer excludes preceding-context prefill; anchors: full prefill vs r=100% reprefill.",
            "hkvd_caveat": "HKVD/oracle require offline true-prefill deviation; measures selection ceiling.",
        },
        metrics=metrics,
        records=records,
        extra={"by_config": by_config, "anchors": anchor_summary, "anchor_records": anchor_records,
               "correctness_canary": correctness_canary},
    )

    print(f"\nWrote artifact: {args.output}")
    print(f"Correctness canary r=100%: KL={correctness_canary['r100_mean_kl']} "
          f"top1={correctness_canary['r100_top1_agreement_rate']} pass={correctness_canary['pass']}")
    print("\n=== Phase B-ext Pareto summary (mean KL / latency ms / top-1) ===")
    for p_start in p_starts:
        print(f"\nP_start={p_start}  anchors: full_prefill={anchor_summary['full_prefill_us'].get('mean', 0)/1000:.1f}ms  "
              f"reprefill100={anchor_summary['reprefill_100pct_us'].get('mean', 0)/1000:.1f}ms")
        print(f"{'selector':>22} {'r%':>4} {'KL':>8} {'top1':>6} {'lat ms':>8}")
        for selector in selectors:
            for r_pct in recompute_pcts:
                b = by_config[f"P{p_start}_{selector}_r{r_pct}"]
                klm = b["kl_ref_to_splice"]
                lat = b["splice_recompute_us"]
                print(f"{selector:>22} {r_pct:>4.0f} {klm.get('mean', 0):>8.4f} "
                      f"{b['top1_agreement_rate']:>6.3f} {lat.get('mean', 0)/1000:>8.2f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(p_starts), figsize=(6 * len(p_starts), 5), squeeze=False)
        for ax_i, p_start in enumerate(p_starts):
            ax = axes[0][ax_i]
            for selector in selectors:
                xs, ys = [], []
                for r_pct in recompute_pcts:
                    b = by_config[f"P{p_start}_{selector}_r{r_pct}"]
                    xs.append(b["splice_recompute_us"]["mean"] / 1000)
                    ys.append(b["kl_ref_to_splice"]["mean"])
                ax.plot(xs, ys, marker="o", label=selector)
            ax.axhline(0.1, color="gray", ls=":", alpha=0.5)
            fp = anchor_summary["full_prefill_us"].get("mean", 0) / 1000
            rp = anchor_summary["reprefill_100pct_us"].get("mean", 0) / 1000
            ax.scatter([fp], [0], c="red", marker="*", s=120, label="full prefill", zorder=5)
            ax.scatter([rp], [0], c="orange", marker="s", s=80, label="r=100% reprefill", zorder=5)
            ax.set_xlabel("Splice + recompute latency (mean ms)")
            ax.set_ylabel("KL(ref ‖ splice) [nats]")
            ax.set_title(f"P_start={p_start}")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        plt.tight_layout()
        out_png = Path(args.output).with_suffix(".pareto.png")
        plt.savefig(out_png, dpi=130)
        print(f"Saved Pareto plot: {out_png}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
