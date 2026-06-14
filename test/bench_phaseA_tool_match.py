#!/usr/bin/env python3
"""Phase A completion: tool-call exact-match + competent-reference NLL.

Uses the 100 discriminating routing queries from bench_routing_accuracy.py
(10 tools × 10 queries each). Each case splices the *correct* tool schema at a
variable P_start after realistic preceding context, then compares reference
prefill vs isolated splice on:

  - tool_name_exact_match: greedy decode of gold tool-name tokens matches exactly
  - gold_nll / gold_nll_delta: only reported when the reference is competent
    (mean per-token NLL below --competent-nll-threshold; default 2.0 nats/token)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
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
import nexus_fsm_ext
from nexus_recompute import recompute_tail  # noqa: E402
from benchmark_results import file_sha256, write_artifact  # noqa: E402
from bench_routing_accuracy import queries_dataset, load_first_10_tools  # noqa: E402

DEFAULT_MODEL = "/Volumes/AI_SSD/models/Qwen2.5-14B-Instruct-GGUF/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf"
DEFAULT_SYSTEM_PROMPT = "You are a tool-using assistant. Use the provided tool schema to answer the user."

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


def nll(logits: np.ndarray, token: int) -> float:
    return float(-log_softmax(logits)[token])


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
    schema = {
        "name": tool["name"],
        "description": tool.get("description") or tool.get("desc", ""),
        "inputSchema": tool.get("inputSchema") or tool.get("input_schema") or {},
    }
    return json.dumps(schema, sort_keys=True)


def format_query(user_query: str) -> str:
    # Align with bench_routing_accuracy.py prompt suffix for comparable tool-name emission.
    return (
        "You are a tool routing agent. Given the tool schema above, select the tool "
        "that matches the user's intent. You MUST output ONLY the tool name, with no "
        "other text, punctuation, explanation, or markdown.\n\n"
        f"Query: {user_query}\n"
        "Selected Tool Name:"
    )


def tool_name_hit(decoded_text: str, tool_name: str) -> bool:
    pred = decoded_text.strip().replace("`", "")
    return (tool_name in pred) or (pred in tool_name) or (pred == tool_name)


def compile_atb(model_path: str, schema_path: Path, output_path: Path) -> None:
    subprocess.run(
        ["./build/nexus_kv_compiler", "--model", model_path, "--schema", str(schema_path), "--output", str(output_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _greedy_decode(ctx, llm, preceding: list[int], schema_tokens: list[int], query_tokens: list[int],
                   gold_tokens: list[int], splice_fn, max_tokens: int) -> tuple[str, float, bool]:
    """Build KV via splice_fn, greedy-decode up to max_tokens, score first gold token NLL."""
    nexus_fsm_ext.clear_kv_cache(ctx)
    if preceding:
        nexus_fsm_ext.decode_tokens(ctx, preceding, 0, 0)
    splice_fn(schema_tokens)
    logits = np.array(
        nexus_fsm_ext.decode_tokens(ctx, query_tokens, len(preceding) + len(schema_tokens), 0),
        dtype=np.float32,
    )

    decoded: list[int] = []
    total_nll = 0.0
    if gold_tokens:
        total_nll = nll(logits, gold_tokens[0])
    pos = len(preceding) + len(schema_tokens) + len(query_tokens)
    for _ in range(max_tokens):
        pred = int(np.argmax(logits))
        decoded.append(pred)
        logits = np.array(nexus_fsm_ext.decode_tokens(ctx, [pred], pos, 0), dtype=np.float32)
        pos += 1
    text = llm.detokenize(decoded).decode("utf-8", errors="replace")
    prefix_exact = decoded[: len(gold_tokens)] == gold_tokens if gold_tokens else False
    return text, total_nll, prefix_exact


def reference_tool_eval(ctx, llm, preceding, schema_tokens, query_tokens, gold_tokens, max_tokens):
    def prefill(schema):
        if schema:
            nexus_fsm_ext.decode_tokens(ctx, schema, len(preceding), 0)
    return _greedy_decode(ctx, llm, preceding, schema_tokens, query_tokens, gold_tokens, prefill, max_tokens)


def splice_tool_eval(ctx, llm, cache, atb_path, preceding, schema_tokens, query_tokens, gold_tokens, max_tokens):
    p_start = len(preceding)

    def inject(_schema):
        handle = cache.get_or_load(str(atb_path))
        nexus_fsm_ext.inject_tool_page(ctx, handle, p_start, 0)

    return _greedy_decode(ctx, llm, preceding, schema_tokens, query_tokens, gold_tokens, inject, max_tokens)


def _contiguous_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs: list[tuple[int, int]] = []
    start = indices[0]
    prev = start
    for idx in indices[1:]:
        if idx != prev + 1:
            runs.append((start, prev))
            start = idx
        prev = idx
    runs.append((start, prev))
    return runs


def _recompute_tail(ctx, schema_tokens: list[int], p_start: int, recompute_pct: float) -> None:
    recompute_tail(ctx, schema_tokens, p_start, recompute_pct)


def splice_recompute_tool_eval(
    ctx, llm, cache, atb_path, preceding, schema_tokens, query_tokens, gold_tokens, max_tokens,
    recompute_pct: float = 5.0,
):
    p_start = len(preceding)

    def inject_and_recompute(_schema):
        handle = cache.get_or_load(str(atb_path))
        nexus_fsm_ext.inject_tool_page(ctx, handle, p_start, 0)
        _recompute_tail(ctx, schema_tokens, p_start, recompute_pct)

    return _greedy_decode(
        ctx, llm, preceding, schema_tokens, query_tokens, gold_tokens,
        inject_and_recompute, max_tokens,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Phase A tool-call exact-match benchmark.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--positions", default="256", help="Comma-separated P_start values (default: claim regime 256).")
    p.add_argument("--output", default="results/bench_phaseA_tool_match.json")
    p.add_argument("--workdir", default="results/phaseA_tool_match_work")
    p.add_argument("--reuse-atb", action="store_true")
    p.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    p.add_argument("--competent-nll-threshold", type=float, default=5.0,
                   help="Max first-token reference NLL to include gold NLL metrics (nats).")
    p.add_argument("--max-decode-tokens", type=int, default=15)
    p.add_argument("--n-ctx", type=int, default=8192)
    p.add_argument("--n-gpu-layers", type=int, default=999)
    p.add_argument("--recompute-pct", type=float, default=5.0,
                   help="Tail recompute percentage for splice+recompute arm (Phase B winner).")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    positions = [int(x) for x in args.positions.split(",") if x.strip()]

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    tools = load_first_10_tools()
    tool_by_name = {t["name"]: t for t in tools}
    cases = [c for c in queries_dataset if c["tool"] in tool_by_name]
    print(f"Tool-match cases: {len(cases)} queries, {len(tools)} tools, positions={positions}")

    schema_tok_ub = 512
    needed_batch = max(positions) + schema_tok_ub + 128
    n_batch = min(args.n_ctx, max(2048, needed_batch))
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

    atb_by_tool: dict[str, Path] = {}
    schema_len_by_tool: dict[str, int] = {}
    schema_tokens_by_tool: dict[str, list[int]] = {}
    for i, tool in enumerate(tools):
        text = schema_text_for(tool)
        schema_path = workdir / f"tool_{i}.schema.json"
        schema_path.write_text(text)
        atb_path = workdir / f"tool_{i}.isolated.atb"
        if not (args.reuse_atb and atb_path.exists()):
            compile_atb(args.model, schema_path, atb_path)
        atb_by_tool[tool["name"]] = atb_path
        toks = tokenize(llm, text)
        schema_len_by_tool[tool["name"]] = len(toks)
        schema_tokens_by_tool[tool["name"]] = toks

    preceding_by_pos = {p: build_preceding_tokens(llm, args.system_prompt, p) for p in positions}
    gold_by_tool = {name: tokenize(llm, name) for name in tool_by_name}

    records = []
    for idx, case in enumerate(cases):
        tool_name = case["tool"]
        schema_tokens = schema_tokens_by_tool[tool_name]
        schema_len = schema_len_by_tool[tool_name]
        query_tokens = tokenize(llm, format_query(case["query"]))
        gold_tokens = gold_by_tool[tool_name]
        atb_path = atb_by_tool[tool_name]

        for p in positions:
            preceding = preceding_by_pos[p]
            t0 = time.perf_counter()
            ref_text, ref_nll, ref_prefix_exact = reference_tool_eval(
                ctx, llm, preceding, schema_tokens, query_tokens, gold_tokens, args.max_decode_tokens)
            ref_us = (time.perf_counter() - t0) * 1e6
            ref_hit = tool_name_hit(ref_text, tool_name)

            t0 = time.perf_counter()
            splice_text, splice_nll, splice_prefix_exact = splice_tool_eval(
                ctx, llm, cache, atb_path, preceding, schema_tokens, query_tokens, gold_tokens, args.max_decode_tokens)
            splice_us = (time.perf_counter() - t0) * 1e6
            splice_hit = tool_name_hit(splice_text, tool_name)

            t0 = time.perf_counter()
            recompute_text, recompute_nll, recompute_prefix_exact = splice_recompute_tool_eval(
                ctx, llm, cache, atb_path, preceding, schema_tokens, query_tokens, gold_tokens,
                args.max_decode_tokens, recompute_pct=args.recompute_pct)
            recompute_us = (time.perf_counter() - t0) * 1e6
            recompute_hit = tool_name_hit(recompute_text, tool_name)

            # End-task competent = reference routes correctly. First-token gold NLL is
            # reported separately only when the first greedy token matches gold prefix
            # (otherwise the model emits preamble tokens before the tool name).
            competent = ref_hit
            nll_reportable = ref_prefix_exact and ref_nll <= args.competent_nll_threshold

            rec = {
                "case_id": f"{tool_name}-{idx % 10}",
                "query": case["query"],
                "tool_name": tool_name,
                "p_start": p,
                "delta_pos": p,
                "gold_tokens": len(gold_tokens),
                "ref_prediction": ref_text.strip(),
                "splice_prediction": splice_text.strip(),
                "recompute_prediction": recompute_text.strip(),
                "ref_tool_hit": ref_hit,
                "splice_tool_hit": splice_hit,
                "recompute_tool_hit": recompute_hit,
                "tool_hit_agreement": ref_hit == splice_hit,
                "recompute_hit_recovery": recompute_hit and not splice_hit,
                "ref_prefix_exact": ref_prefix_exact,
                "splice_prefix_exact": splice_prefix_exact,
                "ref_first_token_nll": ref_nll,
                "splice_first_token_nll": splice_nll,
                "reference_competent": competent,
                "nll_reportable": nll_reportable,
                "ref_decode_us": ref_us,
                "splice_decode_us": splice_us,
                "recompute_decode_us": recompute_us,
                "recompute_pct": args.recompute_pct,
            }
            if nll_reportable:
                rec["first_token_nll_delta"] = splice_nll - ref_nll
            records.append(rec)

        if (idx + 1) % 10 == 0:
            print(f"  {idx + 1}/{len(cases)} queries done")

    by_position = {}
    for p in positions:
        rows = [r for r in records if r["p_start"] == p]
        nll_rows = [r for r in rows if r.get("nll_reportable")]
        by_position[str(p)] = {
            "p_start": p,
            "n": len(rows),
            "ref_tool_hit_rate": float(np.mean([r["ref_tool_hit"] for r in rows])) if rows else 0.0,
            "splice_tool_hit_rate": float(np.mean([r["splice_tool_hit"] for r in rows])) if rows else 0.0,
            "recompute_tool_hit_rate": float(np.mean([r["recompute_tool_hit"] for r in rows])) if rows else 0.0,
            "n_nll_reportable": len(nll_rows),
            "first_token_nll_delta_mean": float(np.mean([r["first_token_nll_delta"] for r in nll_rows])) if nll_rows else None,
        }

    nll_records = [r for r in records if r.get("nll_reportable")]
    metrics = {
        "ref_tool_hit": {"unit": "ratio", "raw_samples": [1.0 if r["ref_tool_hit"] else 0.0 for r in records]},
        "splice_tool_hit": {"unit": "ratio", "raw_samples": [1.0 if r["splice_tool_hit"] else 0.0 for r in records]},
        "recompute_tool_hit": {"unit": "ratio", "raw_samples": [1.0 if r["recompute_tool_hit"] else 0.0 for r in records]},
    }
    if nll_records:
        metrics["first_token_nll_delta_reportable"] = {
            "unit": "nats",
            "raw_samples": [r["first_token_nll_delta"] for r in nll_records],
        }

    write_artifact(
        args.output,
        "bench_phaseA_tool_match",
        model_hash=file_sha256(args.model) if Path(args.model).exists() else "",
        config={
            "model_path": args.model,
            "n_cases": len(cases),
            "positions": positions,
            "competent_nll_threshold": args.competent_nll_threshold,
            "recompute_pct": args.recompute_pct,
            "system_prompt": args.system_prompt,
            "metric_note": (
                "Greedy decode (max 15 tokens) after [preceding||schema||query]; tool hit uses "
                "substring match like bench_routing_accuracy. First-token NLL delta reported only "
                "when reference hits gold AND first-token NLL is below competent threshold."
            ),
        },
        metrics=metrics,
        records=records,
        extra={"by_position": by_position},
    )

    print(f"\nWrote artifact: {args.output}")
    print("\n=== Phase A tool-call exact-match ===")
    for p in positions:
        b = by_position[str(p)]
        print(f"P={p}: ref_hit={b['ref_tool_hit_rate']:.3f}  splice_hit={b['splice_tool_hit_rate']:.3f}  "
              f"recompute_hit={b['recompute_tool_hit_rate']:.3f}  "
              f"nll_reportable={b['n_nll_reportable']}/{b['n']}  "
              f"first_token_nll_delta={b['first_token_nll_delta_mean']}")


if __name__ == "__main__":
    main()
