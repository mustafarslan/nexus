#!/usr/bin/env python3
"""N1: Re-embed Phase E distractors with nomic prefixed protocol; report corrected recall."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "test"))

lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../build/external/llama.cpp/src"))
os.environ.setdefault("LLAMA_CPP_LIB_PATH", lib_dir)
os.environ.setdefault("LLAMA_CPP_LIB", os.path.join(lib_dir, "libllama.dylib"))

import llama_cpp  # noqa: E402
from nexus_retrieval import (  # noqa: E402
    DEFAULT_EMBED_MODEL,
    embed_query,
    embed_tool_document,
    fp32_scan,
    recall_at_k,
)


def parse_args():
    p = argparse.ArgumentParser(description="Re-embed Phase E distractors with honest nomic protocol.")
    p.add_argument("--corpus-dir", default="results/phaseE_corpora")
    p.add_argument("--tiers", default="N100,N1000")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--output", default="results/bench_phaseE_scale_corrected.json")
    p.add_argument("--query-limit", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    llm_emb = llama_cpp.Llama(model_path=args.embed_model, embedding=True, verbose=False)
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    records = []

    for tier in tiers:
        tier_path = Path(args.corpus_dir) / tier
        corpus = json.loads((tier_path / "corpus.json").read_text())
        npz_path = tier_path / "embeddings.npz"
        npz = np.load(npz_path, allow_pickle=True)
        names = list(npz["names"])
        old_embs = npz["embeddings"].astype(np.float32)
        meta = corpus.get("meta", {})
        tools = corpus.get("tools", [])

        new_embs = old_embs.copy()
        reembedded = 0
        for i, name in enumerate(names):
            tool = next((t for t in tools if t.get("name") == name), None)
            if tool is None:
                continue
            if tool.get("is_gold"):
                continue
            new_embs[i] = embed_tool_document(llm_emb, tool)
            reembedded += 1

        name_to_idx = {names[i]: i for i in range(len(names))}
        queries = meta.get("gold_queries", [])[: args.query_limit]
        recall = {1: [], 5: [], 10: []}
        recall_old = {1: [], 5: [], 10: []}
        for qrow in queries:
            gold = qrow["tool"]
            if gold not in name_to_idx:
                continue
            gidx = name_to_idx[gold]
            qvec = embed_query(llm_emb, qrow["query"])
            s_new = fp32_scan(qvec, new_embs)
            s_old = fp32_scan(qvec, old_embs)
            for k in (1, 5, 10):
                recall[k].append(recall_at_k(s_new, gidx, k))
                recall_old[k].append(recall_at_k(s_old, gidx, k))

        row = {
            "tier": tier,
            "n_tools": len(names),
            "distractors_reembedded": reembedded,
            "recall_at_1_old": float(np.mean(recall_old[1])) if recall_old[1] else None,
            "recall_at_1_corrected": float(np.mean(recall[1])) if recall[1] else None,
            "recall_at_5_corrected": float(np.mean(recall[5])) if recall[5] else None,
            "recall_at_10_corrected": float(np.mean(recall[10])) if recall[10] else None,
            "caveat": (
                "N10000/N100000 tiers with random-vector distractors remain upper-bound only; "
                "this script corrects real-manifest tiers where distractor text exists."
            ),
        }
        records.append(row)
        corrected_path = tier_path / "embeddings_corrected.npz"
        np.savez_compressed(corrected_path, embeddings=new_embs, names=np.array(names, dtype=object))
        print(f"{tier}: reembedded={reembedded} recall@1 {row['recall_at_1_old']:.3f} -> {row['recall_at_1_corrected']:.3f}")

    out = {
        "benchmark": "bench_phaseE_scale_corrected",
        "config": vars(args),
        "records": records,
        "distractor_caveat": "Synthetic random-vector distractors in large tiers make tier recall an upper bound.",
    }
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
