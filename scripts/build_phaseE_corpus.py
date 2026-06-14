#!/usr/bin/env python3
"""Build Phase E corpora tiers N={100,1k,10k} + 100k embeddings-only from slb_manifest."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "test"))
from bench_routing_accuracy import load_first_10_tools, queries_dataset  # noqa: E402

lib_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../build/external/llama.cpp/src")
)
os.environ.setdefault("LLAMA_CPP_LIB_PATH", lib_dir)
os.environ.setdefault("LLAMA_CPP_LIB", os.path.join(lib_dir, "libllama.dylib"))


def _load_embedder(model_path: str):
    import llama_cpp  # lazy import

    return llama_cpp.Llama(model_path=model_path, embedding=True, verbose=False)


def _embed_gold_tools(gold_tools: list[dict], embed_model: str) -> list[list[float]]:
    from nexus_retrieval import embed_tool_document  # noqa: WPS433

    llm_emb = _load_embedder(embed_model)
    return [embed_tool_document(llm_emb, t).tolist() for t in gold_tools]


def _lookup_manifest_embedding(all_tools: list[dict], name: str, emb_dim: int) -> list[float] | None:
    for t in all_tools:
        if t.get("name") == name and t.get("embedding"):
            return t["embedding"]
    return None


def main():
    p = argparse.ArgumentParser(description="Build Phase E scale corpora.")
    p.add_argument("--manifest", default="test/schemas/slb_manifest.json")
    p.add_argument("--output-dir", default="results/phaseE_corpora")
    p.add_argument("--embed-model", default="test/nexus_retrieval.py:DEFAULT_EMBED_MODEL")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    embed_model = args.embed_model
    if embed_model.startswith("test/"):
        from nexus_retrieval import DEFAULT_EMBED_MODEL  # noqa: WPS433

        embed_model = DEFAULT_EMBED_MODEL

    rng = random.Random(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    gold_tools = load_first_10_tools()
    gold_names = {t["name"] for t in gold_tools}
    gold_queries = queries_dataset[:100]
    gold_embeddings = _embed_gold_tools(gold_tools, embed_model)

    with open(args.manifest) as f:
        manifest = json.load(f)
    all_tools = manifest.get("tools", manifest if isinstance(manifest, list) else [])
    emb_dim = len(all_tools[0].get("embedding", [])) if all_tools else 768

    def tool_row(i: int, t: dict, embedding: list[float], is_gold: bool, atb_path: str | None = None) -> dict:
        row = {
            "id": i + 1,
            "name": t.get("name", f"tool_{i}"),
            "description": t.get("description", t.get("desc", "")),
            "embedding": embedding,
            "is_gold": is_gold,
        }
        if atb_path is not None:
            row["atb_path"] = atb_path
        return row

    def write_tier(n: int, name: str, embeddings_only: bool = False):
        tier_dir = out / name
        tier_dir.mkdir(parents=True, exist_ok=True)
        pool = [t for t in all_tools if t.get("name") not in gold_names]
        rng.shuffle(pool)
        selected = pool[: max(0, n - len(gold_tools))]
        corpus_tools = gold_tools + selected[: n - len(gold_tools)]
        corpus_tools = corpus_tools[:n]

        tools_out = []
        for i, t in enumerate(corpus_tools):
            if t["name"] in gold_names:
                gi = next(j for j, g in enumerate(gold_tools) if g["name"] == t["name"])
                emb = gold_embeddings[gi]
            else:
                emb = t.get("embedding") or [0.0] * emb_dim
            atb = None if embeddings_only else t.get("atb_path")
            tools_out.append(tool_row(i, t, emb, t["name"] in gold_names, atb))

        meta = {
            "tier": name,
            "n_tools": len(tools_out),
            "embeddings_only": embeddings_only,
            "gold_tool_names": list(gold_names),
            "gold_queries": gold_queries,
            "embed_model": embed_model,
            "embedding_protocol": "nomic search_query/search_document prefixes via nexus_retrieval",
        }
        (tier_dir / "corpus.json").write_text(json.dumps({"meta": meta, "tools": tools_out}, indent=2))
        emb = np.array([np.array(t["embedding"], dtype=np.float32) for t in tools_out], dtype=np.float32)
        np.savez_compressed(
            tier_dir / "embeddings.npz",
            embeddings=emb,
            names=np.array([t["name"] for t in tools_out], dtype=object),
        )
        print(f"Wrote {name}: n={len(tools_out)} embeddings_only={embeddings_only}")

    write_tier(100, "N100")
    write_tier(1000, "N1000")
    write_tier(10000, "N10000")

    tier_dir = out / "N100000_embeddings_only"
    tier_dir.mkdir(parents=True, exist_ok=True)
    synth = []
    for i in range(100000 - len(gold_tools)):
        vec = np.random.default_rng(args.seed + i).standard_normal(emb_dim).astype(np.float32)
        vec /= max(float(np.linalg.norm(vec)), 1e-8)
        synth.append({
            "name": f"synth_tool_{i}",
            "description": f"Synthetic distractor tool {i}",
            "embedding": vec.tolist(),
            "is_gold": False,
        })

    gold_rows = []
    for i, (t, emb) in enumerate(zip(gold_tools, gold_embeddings)):
        gold_rows.append({
            "id": i + 1,
            "name": t["name"],
            "description": t.get("description") or t.get("desc", ""),
            "embedding": emb,
            "is_gold": True,
        })

    corpus = gold_rows + synth
    meta = {
        "tier": "N100000_embeddings_only",
        "n_tools": len(corpus),
        "embeddings_only": True,
        "gold_tool_names": list(gold_names),
        "gold_queries": gold_queries,
        "embed_model": embed_model,
        "embedding_protocol": "nomic search_query/search_document prefixes via nexus_retrieval",
    }
    (tier_dir / "corpus.json").write_text(
        json.dumps({"meta": meta, "tools": corpus[:1000]}, indent=2)
    )
    emb = np.array([np.array(t["embedding"], dtype=np.float32) for t in corpus], dtype=np.float32)
    np.savez_compressed(
        tier_dir / "embeddings.npz",
        embeddings=emb,
        names=np.array([t["name"] for t in corpus], dtype=object),
    )
    print(f"Wrote N100000_embeddings_only: n={len(corpus)}")

    (out / "README.md").write_text(
        "# Phase E corpora\n\n"
        "Tiers N100/N1000/N10000 with gold GitHub tools as needles.\n"
        "Gold tool embeddings use nomic-embed with search_document prefix.\n"
        "N100000 is embeddings-only (SLB scan + recall; no .atb compile).\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
