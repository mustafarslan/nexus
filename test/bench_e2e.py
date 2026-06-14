#!/usr/bin/env python3
"""Phase D: honest end-to-end baselines vs Nexus at 14B.

Arms:
  B1   Full-schema bloat prefill (all 10 tool schemas as text)
  B2   RAG retrieval: top-k schemas prefilled as text (k in {1,3,5})
  B3   Retrieve-then-reprefill: top-1 retrieve, prefill that one schema (no splice)
  B2r   Hybrid top-5 → isolated 14B rerank → single schema decode
  B2r_ce Hybrid top-5 → cross-encoder rerank → single schema decode
  B3pc Prefix-cache hit: preceding + cached schema KV (seq_cp proxy)
  B3pc_miss Prefix-cache miss: first access full prefill
  N1   Nexus: retrieve -> splice -> r%% recompute -> greedy decode
  N1r  Margin-gated 14B rerank router -> N1 KV splice (offline ceiling)
  N1x  Calibrated cross-encoder gate + fused recompute splice (headline arm)
  N1r_ce Cross-encoder margin-gated router -> N1 KV splice (alias of N1x)
  N1x  Cross-encoder margin-gated router → N1 splice (primary Nexus path)
  N1r_ce Cross-encoder margin-gated router → N1 splice (alias of N1x)

Metrics: tool accuracy, true TTFT (prefill + first token), decode time, McNemar stats.
"""
from __future__ import annotations

import argparse
import json
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
from benchmark_stats import paired_mcnemar, summarize_arm_stats  # noqa: E402
from bench_routing_accuracy import queries_dataset, load_first_10_tools  # noqa: E402
from nexus_recompute import plan_suffix_recompute, recompute_fused_with_query, recompute_tail  # noqa: E402
from nexus_retrieval import (  # noqa: E402
    DEFAULT_EMBED_MODEL,
    DEFAULT_MARGIN_THRESHOLD,
    CrossEncoderReranker,
    calibrate_margin_threshold,
    dense_margin,
    embed_query,
    embed_tools,
    rerank_with_llm_isolated,
    retrieve_hybrid,
    route_tool,
    schema_text,
    tool_document_text,
)

DEFAULT_MODEL = "/Volumes/AI_SSD/models/Qwen2.5-14B-Instruct-GGUF/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf"
DEFAULT_SYSTEM = "You are a tool-using assistant. Use the provided tool schema to answer the user."
DEFAULT_P_START = 256
PREFIX_CACHE_SEQ = 1


def tokenize(llm, text: str) -> list[int]:
    return [int(t) for t in llm.tokenize(text.encode("utf-8"), add_bos=False, special=False)]


def format_query(q: str) -> str:
    return (
        "You are a tool routing agent. Given the tool schema above, select the tool "
        "that matches the user's intent. You MUST output ONLY the tool name, with no "
        "other text, punctuation, explanation, or markdown.\n\n"
        f"Query: {q}\nSelected Tool Name:"
    )


def build_preceding(llm, target_len: int) -> list[int]:
    if target_len <= 0:
        return []
    turns = [
        "User: I'm building a small data pipeline and I keep running into timezone bugs.",
        "Assistant: Store everything in UTC and convert only at display boundaries.",
        "User: We also have a flaky integration test that fails about one run in twenty.",
        "Assistant: Flaky tests are often hidden ordering or timing assumptions.",
    ]
    text = DEFAULT_SYSTEM + "\n"
    i = 0
    toks = tokenize(llm, text)
    while len(toks) < target_len:
        text += turns[i % len(turns)] + "\n"
        i += 1
        toks = tokenize(llm, text)
    return toks[:target_len]


def tool_hit(pred: str, gold: str) -> bool:
    p = pred.strip().replace("`", "")
    return gold in p or p in gold or p == gold


def cache_preceding(ctx, preceding: list[int], seq_id: int = 0) -> None:
    """Reset one sequence and prefill preceding context (preserves other seqs)."""
    nexus_fsm_ext.invalidate_sequence(ctx, seq_id, 0, -1)
    if preceding:
        nexus_fsm_ext.decode_tokens(ctx, preceding, 0, seq_id)


class SchemaPrefixCache:
    """Prefix-cache proxy using seq 1 as warm storage + kv_cache_seq_cp to seq 0."""

    def __init__(self):
        self._warmed: set[str] = set()
        self._resident_tool: str | None = None

    def warm(self, ctx, preceding: list[int], schema_tokens: list[int], tool_name: str, p_start: int) -> None:
        if tool_name == self._resident_tool:
            return
        nexus_fsm_ext.invalidate_sequence(ctx, PREFIX_CACHE_SEQ, 0, -1)
        if preceding:
            nexus_fsm_ext.decode_tokens(ctx, preceding, 0, PREFIX_CACHE_SEQ)
        if schema_tokens:
            nexus_fsm_ext.decode_tokens(ctx, schema_tokens, p_start, PREFIX_CACHE_SEQ)
        self._warmed.add(tool_name)
        self._resident_tool = tool_name

    def apply_hit(self, ctx, preceding: list[int], schema_tokens: list[int], tool_name: str, p_start: int) -> None:
        """Refresh warm seq 1, copy schema KV onto a fresh preceding prefill on seq 0."""
        # Re-warm every hit: intervening N1/N1m decodes on seq 0 can evict seq 1 cells.
        nexus_fsm_ext.invalidate_sequence(ctx, PREFIX_CACHE_SEQ, 0, -1)
        if preceding:
            nexus_fsm_ext.decode_tokens(ctx, preceding, 0, PREFIX_CACHE_SEQ)
        if schema_tokens:
            nexus_fsm_ext.decode_tokens(ctx, schema_tokens, p_start, PREFIX_CACHE_SEQ)
        self._warmed.add(tool_name)
        self._resident_tool = tool_name
        cache_preceding(ctx, preceding, seq_id=0)
        end = p_start + len(schema_tokens)
        nexus_fsm_ext.kv_cache_seq_cp(ctx, PREFIX_CACHE_SEQ, 0, p_start, end)


def greedy_tool_name(
    ctx, llm, preceding, schema_tokens, query_tokens, max_tokens=15,
) -> tuple[str, float, float]:
    """Return (text, ttft_us, decode_us). TTFT = schema prefill + query prefill + first token."""
    p_start = len(preceding)
    t0 = time.perf_counter()
    ttft_us = 0.0
    if schema_tokens:
        nexus_fsm_ext.decode_tokens(ctx, schema_tokens, p_start, 0)
    logits = np.array(
        nexus_fsm_ext.decode_tokens(ctx, query_tokens, p_start + len(schema_tokens), 0),
        dtype=np.float32,
    )
    ttft_us = (time.perf_counter() - t0) * 1e6

    decoded = []
    pos = p_start + len(schema_tokens) + len(query_tokens)
    t_dec = time.perf_counter()
    for _ in range(max_tokens):
        pred = int(np.argmax(logits))
        decoded.append(pred)
        logits = np.array(nexus_fsm_ext.decode_tokens(ctx, [pred], pos, 0), dtype=np.float32)
        pos += 1
    decode_us = (time.perf_counter() - t_dec) * 1e6
    text = llm.detokenize(decoded).decode("utf-8", errors="replace")
    return text, ttft_us, decode_us


def nexus_splice_recompute(
    ctx, llm, cache, atb_path, preceding, schema_tokens, query_tokens,
    recompute_pct: float, max_tokens=15,
) -> tuple[str, float, float]:
    p_start = len(preceding)
    cache_preceding(ctx, preceding)
    t0 = time.perf_counter()
    handle = cache.get_or_load(str(atb_path))
    nexus_fsm_ext.inject_tool_page(ctx, handle, p_start, 0)
    logits = recompute_fused_with_query(
        ctx, schema_tokens, p_start, query_tokens, recompute_pct,
    )
    ttft_us = (time.perf_counter() - t0) * 1e6
    decoded = []
    pos = p_start + len(schema_tokens) + len(query_tokens)
    t_dec = time.perf_counter()
    for _ in range(max_tokens):
        pred = int(np.argmax(logits))
        decoded.append(pred)
        logits = np.array(nexus_fsm_ext.decode_tokens(ctx, [pred], pos, 0), dtype=np.float32)
        pos += 1
    decode_us = (time.perf_counter() - t_dec) * 1e6
    text = llm.detokenize(decoded).decode("utf-8", errors="replace")
    return text, ttft_us, decode_us


def rerank_tool(llm_rerank, query: str, candidates: list[str], tool_by_name: dict) -> tuple[str, float]:
    return rerank_with_llm_isolated(llm_rerank, query, candidates, tool_by_name)


def retrieve_top1_hybrid(q: str, q_emb, tool_embs, tool_names, tools) -> str:
    return retrieve_hybrid(q, q_emb, tool_embs, tool_names, tools, 1)[0]


def compile_atbs(tools, model, workdir: Path) -> dict[str, Path]:
    workdir.mkdir(parents=True, exist_ok=True)
    out = {}
    for i, tool in enumerate(tools):
        text = schema_text(tool)
        schema_path = workdir / f"tool_{i}.schema.json"
        atb_path = workdir / f"tool_{i}.isolated.atb"
        schema_path.write_text(text)
        if not atb_path.exists():
            subprocess.run(
                ["./build/nexus_kv_compiler", "--model", model, "--schema", str(schema_path), "--output", str(atb_path)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        out[tool["name"]] = atb_path
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Phase D end-to-end benchmark.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--p-start", type=int, default=DEFAULT_P_START)
    p.add_argument("--recompute-pct", type=float, default=5.0)
    p.add_argument("--output", default="results/bench_e2e.json")
    p.add_argument("--atb-workdir", default="results/phaseA_tool_match_work")
    p.add_argument("--query-limit", type=int, default=100)
    p.add_argument("--n-ctx", type=int, default=8192)
    p.add_argument("--smoke", action="store_true", help="Write to results/smoke/ (refuses to overwrite full runs).")
    p.add_argument("--tag", default="", help="Suffix tag for output artifact filename.")
    p.add_argument("--rerank-margin", type=float, default=None, help="Dense margin below which rerank fires (default: calibrated).")
    p.add_argument("--disable-cross-encoder", action="store_true", help="Skip cross-encoder / N1x arms.")
    p.add_argument("--margin-calibration", default="results/margin_calibration.json", help="Precomputed margin gate.")
    p.add_argument("--llama8-model", default="", help="Optional Llama-3.1-8B GGUF for second-model arm.")
    p.add_argument("--force", action="store_true", help="Allow overwriting larger-n artifacts.")
    return p.parse_args()


def tool_attention_tokens(llm, tool_names: list[str], tool_by_name: dict) -> list[int]:
    """Tool-Attention-style lightweight loader: names + 80-char descriptions only."""
    lines = []
    for name in tool_names:
        t = tool_by_name[name]
        desc = (t.get("description") or t.get("desc") or "")[:80]
        lines.append(f"- {name}: {desc}")
    return tokenize(llm, "Available tools:\n" + "\n".join(lines) + "\n")


def _attach_verb_prefix(tools: list[dict]) -> None:
    for t in tools:
        name = t["name"]
        i = name.find("_")
        t["verb_prefix"] = name[: i + 1] if i >= 0 else name


def main():
    args = parse_args()
    if os.environ.get("NEXUS_BENCH_TOOLS_JSON"):
        tools = json.loads(os.environ["NEXUS_BENCH_TOOLS_JSON"])
    else:
        tools = load_first_10_tools()
    _attach_verb_prefix(tools)
    tool_by_name = {t["name"]: t for t in tools}
    if os.environ.get("NEXUS_BENCH_QUERIES_JSON"):
        cases = json.loads(os.environ["NEXUS_BENCH_QUERIES_JSON"])[: args.query_limit]
    else:
        cases = [c for c in queries_dataset if c["tool"] in tool_by_name][: args.query_limit]

    llm_emb = llama_cpp.Llama(model_path=args.embed_model, embedding=True, verbose=False)
    tool_embs, tool_names = embed_tools(llm_emb, tools)

    n_batch = min(args.n_ctx, max(2048, args.p_start + 800))
    llm = llama_cpp.Llama(
        model_path=args.model, n_ctx=args.n_ctx, n_batch=n_batch, n_ubatch=n_batch,
        n_gpu_layers=999, flash_attn=True, verbose=False,
    )
    llm_rerank = llama_cpp.Llama(
        model_path=args.model, n_ctx=4096, n_batch=512, n_ubatch=512,
        n_gpu_layers=999, flash_attn=True, verbose=False,
    )
    use_ce = not args.disable_cross_encoder
    cross_encoder = None
    if use_ce:
        _ce = CrossEncoderReranker()
        cross_encoder = _ce if _ce.available else None
        if cross_encoder is None:
            print("Warning: cross-encoder unavailable (pip install sentence-transformers); N1x/B2r_ce skipped")
        else:
            cross_encoder.warmup()

    cal_path = Path(args.margin_calibration)
    cal = {}
    if cal_path.exists():
        cal = json.loads(cal_path.read_text()).get("calibration", {})
    if not cal:
        margins = []
        embed_hits = []
        for case in cases:
            q_emb = embed_query(llm_emb, case["query"])
            margins.append(dense_margin(q_emb, tool_embs))
            top1 = retrieve_hybrid(case["query"], q_emb, tool_embs, tool_names, tools, 1)[0]
            embed_hits.append(top1 == case["tool"])
        cal = calibrate_margin_threshold(margins, embed_hits)
    margin_threshold = args.rerank_margin if args.rerank_margin is not None else cal.get("threshold", DEFAULT_MARGIN_THRESHOLD)
    print(f"Margin gate: threshold={margin_threshold:.4f} fire_rate={cal['fire_rate']:.2f} "
          f"miss_coverage={cal['miss_coverage']:.2f}")
    ctx = llm._ctx.ctx
    cache = nexus_fsm_ext.NexusBlockCache(8 * 1024 * 1024 * 1024)
    prefix_cache = SchemaPrefixCache()
    preceding = build_preceding(llm, args.p_start)
    atb_by_tool = compile_atbs(tools, args.model, Path(args.atb_workdir))
    schema_tok = {t["name"]: tokenize(llm, schema_text(t)) for t in tools}
    bloat_tokens = []
    for t in tools:
        bloat_tokens.extend(schema_tok[t["name"]])

    records = []
    for idx, case in enumerate(cases):
        q = case["query"]
        gold = case["tool"]
        query_tokens = tokenize(llm, format_query(q))
        q_emb = embed_query(llm_emb, q)
        top1 = retrieve_top1_hybrid(q, q_emb, tool_embs, tool_names, tools)
        top3 = retrieve_hybrid(q, q_emb, tool_embs, tool_names, tools, 3)
        top5 = retrieve_hybrid(q, q_emb, tool_embs, tool_names, tools, 5)
        retrieval_hit = top1 == gold

        def rec(arm, pred, ttft_us, decode_us, *, retrieval_hit_override=None, **extra):
            records.append({
                "arm": arm, "case_id": idx, "gold": gold, "pred": pred.strip(),
                "tool_hit": tool_hit(pred, gold),
                "retrieval_hit": retrieval_hit if retrieval_hit_override is None else retrieval_hit_override,
                "ttft_us": ttft_us, "decode_us": decode_us, **extra,
            })

        cache_preceding(ctx, preceding)
        pred, ttft, dec = greedy_tool_name(ctx, llm, preceding, bloat_tokens, query_tokens)
        rec("B1", pred, ttft, dec, tokens_prefilled=len(bloat_tokens))

        for k, retrieved in [(1, [top1]), (3, top3), (5, top5)]:
            arm = f"B2_k{k}"
            toks = []
            for name in retrieved:
                toks.extend(schema_tok[name])
            cache_preceding(ctx, preceding)
            pred, ttft, dec = greedy_tool_name(ctx, llm, preceding, toks, query_tokens)
            rec(arm, pred, ttft, dec, tokens_prefilled=len(toks), k=k)

        cache_preceding(ctx, preceding)
        pred, ttft, dec = greedy_tool_name(ctx, llm, preceding, schema_tok[top1], query_tokens)
        rec("B3", pred, ttft, dec, tokens_prefilled=len(schema_tok[top1]), retrieved_tool=top1)
        rec("B_rag_mcp", pred, ttft, dec, tokens_prefilled=len(schema_tok[top1]), retrieved_tool=top1)

        ta_toks = tool_attention_tokens(llm, top5, tool_by_name)
        cache_preceding(ctx, preceding)
        pred, ttft, dec = greedy_tool_name(ctx, llm, preceding, ta_toks, query_tokens)
        rec("B_tool_attention", pred, ttft, dec, tokens_prefilled=len(ta_toks), k=5)

        reranked, rerank_us = rerank_tool(llm_rerank, q, top5, tool_by_name)
        cache_preceding(ctx, preceding)
        pred, ttft, dec = greedy_tool_name(ctx, llm, preceding, schema_tok[reranked], query_tokens)
        rec("B2r", pred, ttft + rerank_us, dec, tokens_prefilled=len(schema_tok[reranked]),
            retrieved_tool=reranked, rerank_us=rerank_us, retrieval_hit_rerank=reranked == gold,
            retrieval_hit_override=reranked == gold)

        routed_llm, rerank_used_llm, route_us_llm, dense_margin_llm, _ = route_tool(
            q, q_emb, tool_embs, tool_names, tools, tool_by_name,
            margin_threshold=margin_threshold, llm_rerank=llm_rerank, rerank_mode="llm",
        )
        cache_preceding(ctx, preceding)
        pred, ttft, dec = nexus_splice_recompute(
            ctx, llm, cache, atb_by_tool[routed_llm], preceding, schema_tok[routed_llm], query_tokens, args.recompute_pct)
        rec("N1r", pred, ttft + (route_us_llm if rerank_used_llm else 0.0), dec,
            tokens_prefilled=len(schema_tok[routed_llm]), retrieved_tool=routed_llm,
            rerank_used=rerank_used_llm, rerank_us=route_us_llm if rerank_used_llm else 0.0,
            dense_margin=dense_margin_llm, retrieval_hit_override=routed_llm == gold)

        if cross_encoder is not None:
            routed_ce, rerank_used_ce, route_us_ce, dense_margin_ce, _ = route_tool(
                q, q_emb, tool_embs, tool_names, tools, tool_by_name,
                margin_threshold=margin_threshold, cross_encoder=cross_encoder, rerank_mode="cross_encoder",
            )
            cache_preceding(ctx, preceding)
            pred, ttft, dec = greedy_tool_name(ctx, llm, preceding, schema_tok[routed_ce], query_tokens)
            rec("B2r_ce", pred, ttft + (route_us_ce if rerank_used_ce else 0.0), dec,
                tokens_prefilled=len(schema_tok[routed_ce]), retrieved_tool=routed_ce,
                rerank_used=rerank_used_ce, rerank_us=route_us_ce if rerank_used_ce else 0.0,
                dense_margin=dense_margin_ce, retrieval_hit_override=routed_ce == gold)

            cache_preceding(ctx, preceding)
            pred, ttft, dec = nexus_splice_recompute(
                ctx, llm, cache, atb_by_tool[routed_ce], preceding, schema_tok[routed_ce], query_tokens, args.recompute_pct)
            rec("N1x", pred, ttft + (route_us_ce if rerank_used_ce else 0.0), dec,
                tokens_prefilled=len(schema_tok[routed_ce]), retrieved_tool=routed_ce,
                rerank_used=rerank_used_ce, rerank_us=route_us_ce if rerank_used_ce else 0.0,
                dense_margin=dense_margin_ce, retrieval_hit_override=routed_ce == gold)
            rec("N1r_ce", pred, ttft + (route_us_ce if rerank_used_ce else 0.0), dec,
                tokens_prefilled=len(schema_tok[routed_ce]), retrieved_tool=routed_ce,
                rerank_used=rerank_used_ce, rerank_us=route_us_ce if rerank_used_ce else 0.0,
                dense_margin=dense_margin_ce, retrieval_hit_override=routed_ce == gold)

        was_warm = top1 in prefix_cache._warmed and prefix_cache._resident_tool == top1
        if was_warm:
            prefix_cache.apply_hit(ctx, preceding, schema_tok[top1], top1, args.p_start)
            t0 = time.perf_counter()
            logits = np.array(
                nexus_fsm_ext.decode_tokens(ctx, query_tokens, args.p_start + len(schema_tok[top1]), 0),
                dtype=np.float32,
            )
            ttft = (time.perf_counter() - t0) * 1e6
            decoded = []
            pos = args.p_start + len(schema_tok[top1]) + len(query_tokens)
            t_dec = time.perf_counter()
            for _ in range(15):
                pred_tok = int(np.argmax(logits))
                decoded.append(pred_tok)
                logits = np.array(nexus_fsm_ext.decode_tokens(ctx, [pred_tok], pos, 0), dtype=np.float32)
                pos += 1
            dec = (time.perf_counter() - t_dec) * 1e6
            pred = llm.detokenize(decoded).decode("utf-8", errors="replace")
            rec("B3pc_hit", pred, ttft, dec, cache_hit=True, retrieved_tool=top1)
        else:
            prefix_cache.warm(ctx, preceding, schema_tok[top1], top1, args.p_start)
            cache_preceding(ctx, preceding)
            pred, ttft, dec = greedy_tool_name(ctx, llm, preceding, schema_tok[top1], query_tokens)
            rec("B3pc_miss", pred, ttft, dec, cache_hit=False, retrieved_tool=top1)

        atb = atb_by_tool[top1]
        cache_preceding(ctx, preceding)
        pred, ttft, dec = nexus_splice_recompute(
            ctx, llm, cache, atb, preceding, schema_tok[top1], query_tokens, args.recompute_pct)
        plan = plan_suffix_recompute("tail", len(schema_tok[top1]), args.recompute_pct)
        rec("N1", pred, ttft, dec, tokens_prefilled=len(schema_tok[top1]),
            retrieved_tool=top1, recompute_pct=args.recompute_pct,
            actual_recompute_tokens=plan.actual_recompute_tokens,
            actual_recompute_pct=plan.actual_recompute_pct)

        if (idx + 1) % 20 == 0:
            print(f"  {idx + 1}/{len(cases)} queries")

    arms = sorted(set(r["arm"] for r in records))
    by_arm = {arm: summarize_arm_stats(records, arm) for arm in arms}

    paired_stats = {}
    if "B3" in by_arm and "N1" in by_arm:
        b3_rows = sorted([r for r in records if r["arm"] == "B3"], key=lambda x: x["case_id"])
        n1_rows = sorted([r for r in records if r["arm"] == "N1"], key=lambda x: x["case_id"])
        paired_stats["B3_vs_N1"] = paired_mcnemar(
            [r["tool_hit"] for r in b3_rows],
            [r["tool_hit"] for r in n1_rows],
        )
        paired_stats["B3_vs_N1"]["ttft_delta_ms_b3_minus_n1"] = (
            by_arm["B3"]["ttft_us_mean"] - by_arm["N1"]["ttft_us_mean"]
        ) / 1000.0
    if "B3" in by_arm and "N1r" in by_arm:
        b3_rows = sorted([r for r in records if r["arm"] == "B3"], key=lambda x: x["case_id"])
        n1r_rows = sorted([r for r in records if r["arm"] == "N1r"], key=lambda x: x["case_id"])
        paired_stats["B3_vs_N1r"] = paired_mcnemar(
            [r["tool_hit"] for r in b3_rows],
            [r["tool_hit"] for r in n1r_rows],
        )
        paired_stats["B3_vs_N1r"]["ttft_delta_ms_b3_minus_n1r"] = (
            by_arm["B3"]["ttft_us_mean"] - by_arm["N1r"]["ttft_us_mean"]
        ) / 1000.0

    b3 = by_arm.get("B3", {})
    n1 = by_arm.get("N1", {})
    mcn = paired_stats.get("B3_vs_N1", {})
    acc_delta = abs(b3.get("tool_accuracy", 0) - n1.get("tool_accuracy", 0))
    ttft_delta_ms = mcn.get("ttft_delta_ms_b3_minus_n1", 0.0)
    significant = mcn.get("significant_005", False)
    verdict = {
        "b3_accuracy": b3.get("tool_accuracy"),
        "n1_accuracy": n1.get("tool_accuracy"),
        "accuracy_delta": acc_delta,
        "mcnemar_p_value": mcn.get("p_value"),
        "mcnemar_significant_005": significant,
        "ttft_delta_ms_b3_minus_n1": ttft_delta_ms,
        "decision": (
            "contribution_is_retrieval_not_splicing"
            if acc_delta <= 0.01 and not significant and abs(ttft_delta_ms) < 500
            else "splice_may_add_value_on_latency_or_accuracy"
        ),
        "rule": "McNemar-gated: if B3~N1 accuracy (not significant) and small TTFT delta, rewrite around retrieval",
    }

    if args.llama8_model and Path(args.llama8_model).exists():
        print("\n=== Second model arm (Llama-8B) ===")
        llm8 = llama_cpp.Llama(
            model_path=args.llama8_model, n_ctx=args.n_ctx, n_batch=2048, n_ubatch=2048,
            n_gpu_layers=999, flash_attn=True, verbose=False,
        )
        ctx8 = llm8._ctx.ctx
        atb8 = compile_atbs(tools, args.llama8_model, Path(args.atb_workdir) / "llama8")
        schema_tok8 = {t["name"]: tokenize(llm8, schema_text(t)) for t in tools}
        preceding8 = build_preceding(llm8, args.p_start)
        for idx, case in enumerate(cases[: min(20, len(cases))]):
            q_emb8 = embed_query(llm_emb, case["query"])
            top1_8 = retrieve_hybrid(case["query"], q_emb8, tool_embs, tool_names, tools, 1)[0]
            if top1_8 not in atb8:
                continue
            query_tokens8 = tokenize(llm8, format_query(case["query"]))
            cache_preceding(ctx8, preceding8)
            pred, ttft, dec = nexus_splice_recompute(
                ctx8, llm8, cache, atb8[top1_8], preceding8, schema_tok8[top1_8], query_tokens8, args.recompute_pct,
            )
            records.append({
                "arm": "N1_llama8", "case_id": idx, "gold": case["tool"], "pred": pred.strip(),
                "tool_hit": tool_hit(pred, case["tool"]), "retrieval_hit": top1_8 == case["tool"],
                "ttft_us": ttft, "decode_us": dec, "retrieved_tool": top1_8,
            })
        by_arm["N1_llama8"] = summarize_arm_stats(records, "N1_llama8")
        arms = sorted(set(r["arm"] for r in records))

    out_path = write_artifact(
        args.output, "bench_e2e",
        model_hash=file_sha256(args.model) if Path(args.model).exists() else "",
        config={
            "model_path": args.model,
            "embed_model": args.embed_model,
            "p_start": args.p_start,
            "recompute_pct": args.recompute_pct,
            "n_queries": len(cases),
            "arms": arms,
            "rerank_margin": margin_threshold,
            "margin_calibration": cal,
            "enable_cross_encoder": use_ce and cross_encoder is not None,
            "retrieval_protocol": "hybrid dense+BM25 RRF (nomic prefixes)",
        },
        metrics={"tool_hit": {"unit": "ratio", "raw_samples": [1.0 if r["tool_hit"] else 0.0 for r in records]}},
        records=records,
        extra={"by_arm": by_arm, "paired_stats": paired_stats, "verdict": verdict},
        smoke=args.smoke,
        tag=args.tag or None,
        force=args.force,
    )

    print(f"\nWrote {out_path}")
    print("\n=== Phase D by arm ===")
    for arm in arms:
        b = by_arm[arm]
        print(f"{arm:10s} acc={b['tool_accuracy']:.3f}  ttft={b['ttft_us_mean']/1000:.1f}ms  "
              f"decode={b['decode_us_mean']/1000:.1f}ms  retrieval={b['retrieval_hit_rate']:.3f}")
    print(f"\nVerdict: {verdict['decision']}  (acc_delta={acc_delta:.3f}, "
          f"mcnemar_p={mcn.get('p_value', 'n/a')}, ttft_delta={ttft_delta_ms:.1f}ms)")


if __name__ == "__main__":
    main()
