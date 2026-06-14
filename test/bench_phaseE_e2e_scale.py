#!/usr/bin/env python3
"""I2/N2: End-to-end TTFT vs N and meaningful Zipfian cold-load over 52 real schemas."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
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
from nexus_recompute import recompute_tail  # noqa: E402
from nexus_retrieval import DEFAULT_EMBED_MODEL, embed_query, fp32_scan, schema_text  # noqa: E402

DEFAULT_MODEL = "/Volumes/AI_SSD/models/Qwen2.5-14B-Instruct-GGUF/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf"
DEFAULT_CORPUS = [
    "test/schemas/github_tools.json",
    "test/schemas/massive_20_tools.json",
    "test/schemas/sqlite_tools.json",
]
MAX_SCHEMA_CHARS = 1600
P_START = 256
RECOMPUTE_PCT = 5.0


def tokenize(llm, text: str) -> list[int]:
    return [int(t) for t in llm.tokenize(text.encode("utf-8"), add_bos=False, special=False)]


def load_tier(corpus_dir: Path, tier: str):
    tier_path = corpus_dir / tier
    corpus = json.loads((tier_path / "corpus.json").read_text())
    npz = np.load(tier_path / "embeddings.npz", allow_pickle=True)
    embs = npz["embeddings"].astype(np.float32)
    names = list(npz["names"])
    return corpus, embs, names


def load_corpus_tools() -> list[dict]:
    tools: list[dict] = []
    seen: set[str] = set()
    for path in DEFAULT_CORPUS:
        data = json.loads(Path(path).read_text())
        for tool in data.get("tools", []):
            name = tool.get("name")
            if not name or name in seen:
                continue
            seen.add(name)
            tools.append({
                "name": name,
                "description": tool.get("description", ""),
                "inputSchema": tool.get("inputSchema") or {},
            })
    return tools


def schema_text_for(tool: dict, max_chars: int = MAX_SCHEMA_CHARS) -> str:
    schema = {
        "name": tool["name"],
        "description": tool["description"],
        "inputSchema": tool["inputSchema"],
    }
    text = json.dumps(schema, sort_keys=True)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
    return text


def zipfian_sample(rng: random.Random, n: int, s: float = 1.0) -> int:
    weights = [1.0 / ((i + 1) ** s) for i in range(n)]
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if r <= acc:
            return i
    return n - 1


def parse_args():
    p = argparse.ArgumentParser(description="Phase E E2E TTFT vs N + Zipfian cold load.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--corpus-dir", default="results/phaseE_corpora")
    p.add_argument("--tiers", default="N100,N1000,N10000")
    p.add_argument("--query-limit", type=int, default=20)
    p.add_argument("--zipf-queries", type=int, default=500)
    p.add_argument("--cache-blocks", default="4,8,16,32,64")
    p.add_argument("--pin-top-n", type=int, default=5, help="Pin top-N Zipf head tools in LRU sim.")
    p.add_argument("--zipf-warmup", type=int, default=104, help="Warmup queries before steady-state cold-load measure.")
    p.add_argument("--lfu", action="store_true", help="Use LFU eviction instead of LRU in Zipf sim.")
    p.add_argument("--schemas-dir", default="results/phaseA_fidelity_work")
    p.add_argument("--output", default="results/bench_phaseE_e2e_scale.json")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if not Path(args.model).exists():
        print(f"Model missing: {args.model}")
        return 1

    llm_emb = llama_cpp.Llama(model_path=args.embed_model, embedding=True, verbose=False)
    llm = llama_cpp.Llama(
        model_path=args.model, n_ctx=8192, n_batch=2048, n_ubatch=2048,
        n_gpu_layers=999, flash_attn=True, verbose=False,
    )
    ctx = llm._ctx.ctx
    cache = nexus_fsm_ext.NexusBlockCache(8 * 1024 * 1024 * 1024)
    gold_tools = load_first_10_tools()
    gold_schema_tok = {t["name"]: tokenize(llm, schema_text(t)) for t in gold_tools}
    atb_dir = Path("results/phaseA_tool_match_work")
    atb_by_name = {t["name"]: atb_dir / f"tool_{i}.isolated.atb" for i, t in enumerate(gold_tools)}

    records = []
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]

    for tier in tiers:
        corpus, embs, names = load_tier(Path(args.corpus_dir), tier)
        name_to_idx = {names[i]: i for i in range(len(names))}
        queries = corpus["meta"].get("gold_queries", queries_dataset)[: args.query_limit]
        b3_ttft = []
        n1_ttft = []
        for qrow in queries:
            gold = qrow["tool"]
            if gold not in name_to_idx or gold not in gold_schema_tok:
                continue
            qvec = embed_query(llm_emb, qrow["query"])
            scores = fp32_scan(qvec, embs)
            top1_idx = int(np.argmax(scores))
            top1 = names[top1_idx]
            schema_tokens = gold_schema_tok.get(top1, gold_schema_tok[gold])
            query_tokens = tokenize(llm, f"Query: {qrow['query']}\nSelected Tool Name:")
            nexus_fsm_ext.clear_kv_cache(ctx)
            if P_START:
                nexus_fsm_ext.decode_tokens(ctx, [1] * P_START, 0, 0)
            t0 = time.perf_counter()
            nexus_fsm_ext.decode_tokens(ctx, schema_tokens, P_START, 0)
            nexus_fsm_ext.decode_tokens(ctx, query_tokens, P_START + len(schema_tokens), 0)
            b3_ttft.append((time.perf_counter() - t0) * 1000)

            nexus_fsm_ext.clear_kv_cache(ctx)
            if P_START:
                nexus_fsm_ext.decode_tokens(ctx, [1] * P_START, 0, 0)
            t0 = time.perf_counter()
            atb = atb_by_name.get(top1)
            if atb and atb.exists():
                handle = cache.get_or_load(str(atb))
                nexus_fsm_ext.inject_tool_page(ctx, handle, P_START, 0)
                recompute_tail(ctx, schema_tokens, P_START, RECOMPUTE_PCT)
            else:
                nexus_fsm_ext.decode_tokens(ctx, schema_tokens, P_START, 0)
            nexus_fsm_ext.decode_tokens(ctx, query_tokens, P_START + len(schema_tokens), 0)
            n1_ttft.append((time.perf_counter() - t0) * 1000)

        records.append({
            "tier": tier,
            "n_tools": len(names),
            "b3_ttft_ms_p50": float(np.percentile(b3_ttft, 50)) if b3_ttft else None,
            "n1_ttft_ms_p50": float(np.percentile(n1_ttft, 50)) if n1_ttft else None,
            "n_queries": len(b3_ttft),
        })
        print(f"{tier}: B3 p50={records[-1]['b3_ttft_ms_p50']:.1f}ms N1 p50={records[-1]['n1_ttft_ms_p50']:.1f}ms")

    rng = random.Random(1337)
    schema_dir = Path(args.schemas_dir)
    zipf_tools = load_corpus_tools()
    zipf_atbs = {}
    zipf_schema_tok = {}
    for i, tool in enumerate(zipf_tools):
        name = tool["name"]
        atb = schema_dir / f"tool_{i}.isolated.atb"
        if not atb.exists():
            atb = Path("results/phaseA_fidelity_work") / f"tool_{i}.isolated.atb"
        zipf_atbs[name] = atb
        zipf_schema_tok[name] = tokenize(llm, schema_text_for(tool))

    cache_sizes = [int(x) for x in args.cache_blocks.split(",") if x.strip()]
    if args.smoke:
        args.zipf_queries = min(args.zipf_queries, 50)
        zipf_tools = zipf_tools[:10]
        cache_sizes = cache_sizes[:2]

    zipf_by_cache = {}
    n_universe = len(zipf_tools)
    tool_names = [t.get("name", f"tool_{i}") for i, t in enumerate(zipf_tools)]
    # Tier sweeps pin blocks across the 10-tool gold set; reset before Zipf cold-load sim.
    cache = nexus_fsm_ext.NexusBlockCache(8 * 1024 * 1024 * 1024)

    zipf_head = tool_names[: args.pin_top_n]
    zipf_freq: dict[str, int] = {n: 0 for n in tool_names}
    pinned = set(zipf_head)

    for cache_blocks in cache_sizes:
        lru: list[str] = []
        cold_loads = 0
        latencies = []
        full_residency = cache_blocks >= n_universe
        cap = (n_universe - len(pinned)) if full_residency else max(cache_blocks - len(pinned), 0)

        def touch_lru(tool_name: str) -> None:
            if tool_name in pinned:
                return
            if tool_name in lru:
                lru.remove(tool_name)
            lru.append(tool_name)
            if full_residency:
                return
            if args.lfu:
                while len(lru) > cap:
                    evict = min(lru, key=lambda n: zipf_freq.get(n, 0), default=None)
                    if evict is None:
                        break
                    lru.remove(evict)
            else:
                while len(lru) > cap:
                    lru.pop(0)

        for name in tool_names:
            zipf_freq[name] = zipf_freq.get(name, 0) + 1
            touch_lru(name)
        for _ in range(args.zipf_warmup):
            idx = zipfian_sample(rng, n_universe, s=1.0)
            tool_name = tool_names[idx]
            zipf_freq[tool_name] = zipf_freq.get(tool_name, 0) + 1
            touch_lru(tool_name)

        for _ in range(args.zipf_queries):
            idx = zipfian_sample(rng, n_universe, s=1.0)
            tool_name = tool_names[idx]
            zipf_freq[tool_name] = zipf_freq.get(tool_name, 0) + 1
            schema_tokens = zipf_schema_tok[tool_name]
            atb = zipf_atbs.get(tool_name)
            resident = pinned | set(lru)
            cold = tool_name not in resident
            if cold:
                cold_loads += 1
            touch_lru(tool_name)

            nexus_fsm_ext.clear_kv_cache(ctx)
            if P_START:
                nexus_fsm_ext.decode_tokens(ctx, [1] * P_START, 0, 0)
            t0 = time.perf_counter()
            if atb and atb.exists():
                try:
                    handle = cache.get_or_load(str(atb))
                    nexus_fsm_ext.inject_tool_page(ctx, handle, P_START, 0)
                    recompute_tail(ctx, schema_tokens, P_START, RECOMPUTE_PCT)
                except nexus_fsm_ext.ResourceExhaustedError:
                    nexus_fsm_ext.decode_tokens(ctx, schema_tokens, P_START, 0)
            else:
                nexus_fsm_ext.decode_tokens(ctx, schema_tokens, P_START, 0)
            latencies.append((time.perf_counter() - t0) * 1000)

        zipf_by_cache[str(cache_blocks)] = {
            "cache_blocks": cache_blocks,
            "n_queries": args.zipf_queries,
            "universe_tools": n_universe,
            "cold_load_fraction": cold_loads / max(args.zipf_queries, 1),
            "effective_ttft_ms_p50": float(np.percentile(latencies, 50)),
            "effective_ttft_ms_p99": float(np.percentile(latencies, 99)),
            "pin_top_n": args.pin_top_n,
            "lfu": args.lfu,
            "zipf_warmup": args.zipf_warmup,
        }
        print(f"Zipf cache={cache_blocks}: cold={zipf_by_cache[str(cache_blocks)]['cold_load_fraction']:.2f} "
              f"p50={zipf_by_cache[str(cache_blocks)]['effective_ttft_ms_p50']:.1f}ms")

    zipf_summary = {
        "zipf_s": 1.0,
        "universe_tools": n_universe,
        "by_cache_size": zipf_by_cache,
        "tier_verdict": {
            "10^2_10^4": "measured with KV cache + optional cold splice",
            "10^5": "retrieval-only; per-tool FP16 KV ~4TB extrapolated",
        },
    }

    out_path = write_artifact(
        args.output,
        "bench_phaseE_e2e_scale",
        model_hash=file_sha256(args.model) if Path(args.model).exists() else "",
        config=vars(args),
        metrics={},
        records=records,
        extra={"zipfian": zipf_summary},
        smoke=args.smoke,
        force=args.force,
    )
    print(f"Wrote {out_path}")
    print(json.dumps(zipf_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
