#!/usr/bin/env python3
"""Phase E: SLB scan latency, recall@k INT8 vs FP32, storage footprint vs N."""
from __future__ import annotations

import argparse
import json
import os
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
from benchmark_results import write_artifact  # noqa: E402
from nexus_retrieval import (  # noqa: E402
    DEFAULT_EMBED_MODEL,
    embed_query,
    fp32_scan,
    int8_quantize_rows,
    int8_scan,
    recall_at_k,
)


def slb_scan_us(slb, query: np.ndarray, top_k: int, repeats: int = 50) -> tuple[float, float]:
    latencies = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        slb.search(query.astype(np.float32), top_k)
        latencies.append((time.perf_counter() - t0) * 1e6)
    return float(np.percentile(latencies, 50)), float(np.percentile(latencies, 99))


def parse_args():
    p = argparse.ArgumentParser(description="Phase E scale curves.")
    p.add_argument("--corpus-dir", default="results/phaseE_corpora")
    p.add_argument("--tiers", default="N100,N1000,N10000,N100000_embeddings_only")
    p.add_argument("--output", default="results/bench_phaseE_scale.json")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--embed-dim", type=int, default=768)
    p.add_argument("--query-limit", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    records = []

    llm_emb = llama_cpp.Llama(model_path=args.embed_model, embedding=True, verbose=False)

    for tier in tiers:
        tier_path = Path(args.corpus_dir) / tier
        corpus_file = tier_path / "corpus.json"
        emb_file = tier_path / "embeddings.npz"
        if not corpus_file.exists() or not emb_file.exists():
            print(f"Skip missing tier {tier}")
            continue
        corpus = json.loads(corpus_file.read_text())
        meta = corpus["meta"]
        npz = np.load(emb_file, allow_pickle=True)
        embs = npz["embeddings"].astype(np.float32)
        names = list(npz["names"]) if "names" in npz else [t["name"] for t in corpus.get("tools", [])]
        n = embs.shape[0]
        name_to_idx = {names[i]: i for i in range(n)}
        emb_norms = np.linalg.norm(embs, axis=1)
        q8, scales = int8_quantize_rows(embs)

        slb = nexus_fsm_ext.NexusSemanticSLB(args.embed_dim)
        for i in range(n):
            digest = [100 + (i % 50), 101, 102, 103]
            slb.register_tool(1000 + i, embs[i], digest)

        queries = meta.get("gold_queries", [])[: args.query_limit]
        scan_p50, scan_p99 = slb_scan_us(slb, embs[0], 10)

        recall_fp32 = {1: [], 5: [], 10: []}
        recall_int8 = {1: [], 5: [], 10: []}
        for qrow in queries:
            gold = qrow["tool"]
            if gold not in name_to_idx:
                continue
            gidx = name_to_idx[gold]
            qtext = qrow["query"]
            qvec = embed_query(llm_emb, qtext)
            s_fp = fp32_scan(qvec, embs)
            s_i8 = int8_scan(qvec, q8, scales, emb_norms)
            for k in (1, 5, 10):
                recall_fp32[k].append(recall_at_k(s_fp, gidx, k))
                recall_int8[k].append(recall_at_k(s_i8, gidx, k))

        atb_bytes = 0
        for t in corpus.get("tools", []):
            p = t.get("atb_path")
            if p and Path(p).exists():
                atb_bytes += Path(p).stat().st_size

        row = {
            "tier": tier,
            "n_tools": n,
            "embeddings_only": meta.get("embeddings_only", False),
            "slb_scan_p50_us": scan_p50,
            "slb_scan_p99_us": scan_p99,
            "recall_fp32_at_1": float(np.mean(recall_fp32[1])) if recall_fp32[1] else None,
            "recall_fp32_at_5": float(np.mean(recall_fp32[5])) if recall_fp32[5] else None,
            "recall_fp32_at_10": float(np.mean(recall_fp32[10])) if recall_fp32[10] else None,
            "recall_int8_at_1": float(np.mean(recall_int8[1])) if recall_int8[1] else None,
            "recall_int8_at_5": float(np.mean(recall_int8[5])) if recall_int8[5] else None,
            "recall_int8_at_10": float(np.mean(recall_int8[10])) if recall_int8[10] else None,
            "measured_atb_bytes": atb_bytes,
            "extrapolated_atb_gb": (n * 40e6) / 1e9,
            "eval_protocol": "real query text via nomic search_query prefix",
            "n_eval_queries": len(recall_fp32[1]),
        }
        records.append(row)
        print(
            f"{tier}: n={n} SLB p50={scan_p50:.0f}us "
            f"recall@1 fp32={row['recall_fp32_at_1']:.3f} int8={row['recall_int8_at_1']:.3f}"
        )

    write_artifact(
        args.output,
        "bench_phaseE_scale",
        model_hash="",
        config={
            "corpus_dir": args.corpus_dir,
            "tiers": tiers,
            "embed_model": args.embed_model,
            "eval_protocol": "real query embeddings with nomic prefixes",
        },
        metrics={},
        records=records,
        extra={"footnote": "10^5 tier embeddings-only; storage extrapolated at ~40MB/schema observed."},
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
