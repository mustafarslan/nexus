#!/usr/bin/env python3
"""Experiment 4: ML routing ablation with dynamic intent-signature mocking for v0."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

from _common import DATA_DIR, REPO_ROOT, ensure_dirs, percentiles_us, resolve_embed_model, resolve_model, setup_paths, write_csv

setup_paths()

import llama_cpp  # noqa: E402
from bench_routing_accuracy import queries_dataset, load_first_10_tools  # noqa: E402
from nexus_agent import NexusAgent  # noqa: E402
from nexus_retrieval import (  # noqa: E402
    DEFAULT_RERANK_MARGIN,
    embed_query,
    embed_tool_document,
    tool_document_text,
)


def tool_hit(pred: str | None, gold: str) -> bool:
    if not pred:
        return False
    p = pred.strip().replace("`", "")
    return gold in p or p in gold or p == gold


def p80_margin_threshold() -> float:
    import json
    cal_path = REPO_ROOT / "results" / "margin_calibration_p20.json"
    if cal_path.exists():
        cal = json.loads(cal_path.read_text())
        margins = [float(r["margin"]) for r in cal.get("per_query", []) if "margin" in r]
        if margins:
            return float(np.percentile(margins, 80))
    return 0.15


def build_base_agent(model: str, embed_model: str) -> NexusAgent:
    llm_emb = llama_cpp.Llama(model_path=embed_model, embedding=True, verbose=False)
    llm = llama_cpp.Llama(
        model_path=model, n_ctx=8192, n_batch=2048, n_gpu_layers=999, flash_attn=True, verbose=False,
    )
    llm_rerank = llama_cpp.Llama(
        model_path=model, n_ctx=4096, n_batch=512, n_gpu_layers=999, flash_attn=True, verbose=False,
    )
    agent = NexusAgent(llm, llm_emb, base_pos=256, max_splice_pos=256)
    agent.configure_reranker(llm_rerank=llm_rerank)
    return agent


def register_production_tools(agent: NexusAgent, tools: list, atb_dir: Path, *, use_intent_signatures: bool = True) -> None:
    for i, tool in enumerate(tools):
        if use_intent_signatures:
            emb = embed_tool_document(agent.embedding_llm, tool)
        else:
            raw = agent.embedding_llm.embed(
                "search_document: " + tool_document_text(tool)
            )
            emb = np.array(raw[0] if isinstance(raw[0], list) else raw, dtype=np.float32)
        digest = f"Tool Name: {tool['name']}. Description: {tool.get('description', '')}."
        atb = atb_dir / f"tool_{i}.isolated.atb"
        agent.register_tool(i + 1, tool["name"], emb, digest, str(atb), schema=tool.get("inputSchema"))
    agent.pin_hot_tools(list(range(1, 6)))


def run_config(
    config_name: str,
    agent: NexusAgent,
    cases: list[dict],
    *,
    rerank_margin: float,
    disable_ce: bool = False,
    patch_intent: bool = False,
) -> dict:
    agent.rerank_margin = rerank_margin
    if disable_ce:
        agent.configure_reranker(llm_rerank=None)
        agent.cross_encoder = None
    else:
        agent.configure_reranker(llm_rerank=agent.llm_rerank)

    records: list[dict] = []
    ctx = patch("nexus_retrieval._intent_signature_prefix", return_value="") if patch_intent else None
    try:
        if ctx:
            ctx.start()
        for idx, case in enumerate(cases):
            q = case["query"]
            gold = case["tool"]
            t0 = time.perf_counter()
            _, tool_name, meta = agent.route_with_retrieval(q)
            ttft_us = (time.perf_counter() - t0) * 1e6
            records.append({
                "tool_hit": tool_hit(tool_name, gold),
                "ttft_us": ttft_us,
                "rerank_used": meta.get("rerank_used", False),
            })
            if (idx + 1) % 25 == 0:
                print(f"  {config_name}: {idx + 1}/{len(cases)}")
    finally:
        if ctx:
            ctx.stop()

    ttfts = [r["ttft_us"] for r in records]
    stats = percentiles_us(ttfts)
    ce_rate = sum(1 for r in records if r["rerank_used"]) / len(records) if records else 0.0
    acc = sum(r["tool_hit"] for r in records) / len(records) if records else 0.0

    return {
        "config": config_name,
        "ce_invocation_rate": round(ce_rate, 4),
        "ttft_p50_ms": round(stats["p50"] / 1000.0, 3),
        "ttft_p90_ms": round(stats["p90"] / 1000.0, 3),
        "ttft_p99_ms": round(stats["p99"] / 1000.0, 3),
        "tool_hit_accuracy": round(acc, 4),
        "n_queries": len(records),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-limit", type=int, default=100)
    args = parser.parse_args()

    ensure_dirs()
    model = resolve_model()
    embed_model = resolve_embed_model()
    tools = load_first_10_tools()
    tool_by_name = {t["name"]: t for t in tools}
    cases = [c for c in queries_dataset if c["tool"] in tool_by_name][: args.query_limit]
    atb_dir = REPO_ROOT / "results" / "phaseA_tool_match_work"

    rows: list[dict] = []

    # Dense only
    agent = build_base_agent(model, embed_model)
    register_production_tools(agent, tools, atb_dir)
    rows.append(run_config(
        "Nexus_Dense_Only", agent, cases, rerank_margin=1.0, disable_ce=True,
    ))

    # v0 uncalibrated — no intent signatures, ~80% CE fire
    agent = build_base_agent(model, embed_model)
    register_production_tools(agent, tools, atb_dir, use_intent_signatures=False)
    rows.append(run_config(
        "Nexus_v0_Uncalibrated", agent, cases,
        rerank_margin=p80_margin_threshold(), patch_intent=True,
    ))

    # v1 production
    agent = build_base_agent(model, embed_model)
    register_production_tools(agent, tools, atb_dir, use_intent_signatures=True)
    rows.append(run_config(
        "Nexus_v1_Production", agent, cases, rerank_margin=DEFAULT_RERANK_MARGIN,
    ))

    write_csv(
        DATA_DIR / "ml_ablation_table.csv",
        [
            "config", "ce_invocation_rate", "ttft_p50_ms", "ttft_p90_ms",
            "ttft_p99_ms", "tool_hit_accuracy", "n_queries",
        ],
        rows,
    )
    print(f"Wrote {DATA_DIR / 'ml_ablation_table.csv'}")
    for r in rows:
        print(f"  {r['config']}: acc={r['tool_hit_accuracy']} ce={r['ce_invocation_rate']} ttft_p50={r['ttft_p50_ms']}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
