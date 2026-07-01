#!/usr/bin/env python3
"""Experiment 3: Scalability sweep — SLB scan, memory, E2E TTFT vs registry size N."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from _common import DATA_DIR, REPO_ROOT, ensure_dirs, percentiles_us, resolve_embed_model, resolve_model, setup_paths, write_csv
from _corpus_pad import pad_tools_to_n

setup_paths()

import llama_cpp  # noqa: E402
import nexus_fsm_ext  # noqa: E402
from bench_routing_accuracy import queries_dataset, load_first_10_tools  # noqa: E402
from nexus_agent import NexusAgent  # noqa: E402
from nexus_retrieval import DEFAULT_RERANK_MARGIN, embed_query, embed_tool_document  # noqa: E402

SCALES = [10, 50, 100, 500, 1000]
E2E_QUERIES_AT_SCALE = {10: 10, 50: 10, 100: 10, 500: 10, 1000: 10}


def slb_scan_us(slb, query: np.ndarray, top_k: int = 10, repeats: int = 50) -> tuple[float, float]:
    latencies = []
    q = query.astype(np.float32)
    for _ in range(repeats):
        t0 = time.perf_counter()
        slb.search(q, top_k)
        latencies.append((time.perf_counter() - t0) * 1e6)
    return float(np.percentile(latencies, 50)), float(np.percentile(latencies, 99))


def memory_footprint_mb(n: int, dim: int, padded_dim: int) -> float:
    emb_bytes = n * dim * 4
    slb_bytes = n * padded_dim  # INT8 quantized rows
    digest_overhead = n * 128
    return (emb_bytes + slb_bytes + digest_overhead) / (1024 * 1024)


def ensure_dummy_atb(atb_dir: Path, model: str) -> Path:
    atb_dir.mkdir(parents=True, exist_ok=True)
    dummy = atb_dir / "dummy_shared.atb"
    if dummy.exists():
        return dummy
    # Reuse first compiled ATB if present
    work = REPO_ROOT / "results" / "phaseA_tool_match_work"
    for i in range(10):
        src = work / f"tool_{i}.isolated.atb"
        if src.exists():
            dummy.write_bytes(src.read_bytes())
            return dummy
    raise FileNotFoundError("No ATB available for dummy registration; compile phaseA first.")
    return dummy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", default=",".join(str(s) for s in SCALES))
    args = parser.parse_args()
    scales = [int(s.strip()) for s in args.scales.split(",") if s.strip()]

    ensure_dirs()
    setup_paths()
    model = resolve_model()
    embed_model = resolve_embed_model()
    llm_emb = llama_cpp.Llama(model_path=embed_model, embedding=True, verbose=False)
    dim = llm_emb.n_embd()

    tool_by_name = {t["name"]: t for t in load_first_10_tools()}
    eval_cases = [c for c in queries_dataset if c["tool"] in tool_by_name][:10]

    dummy_atb = ensure_dummy_atb(DATA_DIR / "dummy_atb", model)
    rows: list[dict] = []

    for n in scales:
        tools = pad_tools_to_n(n)
        embs = np.stack([embed_tool_document(llm_emb, t) for t in tools]).astype(np.float32)

        slb = nexus_fsm_ext.NexusSemanticSLB(dim)
        for i in range(n):
            digest = [100 + (i % 50), 101, 102, 103]
            slb.register_tool(1000 + i, embs[i], digest)

        scan_p50, scan_p99 = slb_scan_us(slb, embs[0], top_k=10)
        mem_mb = memory_footprint_mb(n, dim, slb.padded_dim)

        # E2E routing TTFT with shared dummy ATB (no per-tool disk I/O)
        llm = llama_cpp.Llama(
            model_path=model, n_ctx=8192, n_batch=2048, n_gpu_layers=999, flash_attn=True, verbose=False,
        )
        agent = NexusAgent(llm, llm_emb, base_pos=256, max_splice_pos=256, rerank_margin=DEFAULT_RERANK_MARGIN)
        for i, tool in enumerate(tools):
            agent.register_tool(
                i + 1, tool["name"], embs[i],
                f"Tool Name: {tool['name']}.",
                str(dummy_atb),
            )

        n_q = E2E_QUERIES_AT_SCALE.get(n, 10)
        ttfts: list[float] = []
        for case in eval_cases[:n_q]:
            t0 = time.perf_counter()
            agent.route_with_retrieval(case["query"])
            ttfts.append((time.perf_counter() - t0) * 1e6)
        del llm, agent

        e2e_stats = percentiles_us(ttfts)
        rows.append({
            "n_tools": n,
            "slb_scan_p50_us": round(scan_p50, 3),
            "slb_scan_p99_us": round(scan_p99, 3),
            "memory_mb": round(mem_mb, 2),
            "e2e_ttft_p50_ms": round(e2e_stats["p50"] / 1000.0, 3),
            "e2e_ttft_p99_ms": round(e2e_stats["p99"] / 1000.0, 3),
            "n_e2e_queries": n_q,
            "dummy_atb": str(dummy_atb.name),
        })
        print(f"  N={n}: slb_p50={scan_p50:.2f}us mem={mem_mb:.1f}MB e2e_p50={e2e_stats['p50']/1000:.1f}ms")

    write_csv(
        DATA_DIR / "scalability_curve.csv",
        [
            "n_tools", "slb_scan_p50_us", "slb_scan_p99_us", "memory_mb",
            "e2e_ttft_p50_ms", "e2e_ttft_p99_ms", "n_e2e_queries", "dummy_atb",
        ],
        rows,
    )
    print(f"Wrote {DATA_DIR / 'scalability_curve.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
