#!/usr/bin/env python3
"""Re-embed the 10-tool GitHub benchmark corpus after schema engineering."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "test"))

lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../build/external/llama.cpp/src"))
os.environ.setdefault("LLAMA_CPP_LIB_PATH", lib_dir)
os.environ.setdefault("LLAMA_CPP_LIB", os.path.join(lib_dir, "libllama.dylib"))

import llama_cpp  # noqa: E402
from bench_routing_accuracy import load_first_10_tools, queries_dataset  # noqa: E402
from nexus_retrieval import (  # noqa: E402
    DEFAULT_EMBED_MODEL,
    embed_query,
    embed_tools,
    retrieve_hybrid,
)


def embed_with_sentence_transformer(model_path: str, tools: list[dict], queries: list[str]):
    from sentence_transformers import SentenceTransformer  # type: ignore
    from nexus_retrieval import NOMIC_DOC_PREFIX, NOMIC_QUERY_PREFIX, tool_document_text

    model = SentenceTransformer(model_path, trust_remote_code=True)
    docs = [NOMIC_DOC_PREFIX + tool_document_text(t) for t in tools]
    qs = [NOMIC_QUERY_PREFIX + q for q in queries]
    doc_embs = model.encode(docs, normalize_embeddings=True)
    q_embs = model.encode(qs, normalize_embeddings=True)
    return np.asarray(doc_embs, dtype=np.float32), np.asarray(q_embs, dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser(description="Re-embed GitHub tools and report recall@1/@5.")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--finetuned-model", default="", help="Optional HF SentenceTransformer path.")
    p.add_argument("--output", default="results/github_tool_embeddings.npz")
    p.add_argument("--recall-gate", type=float, default=0.92)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    tools = load_first_10_tools()
    tool_by_name = {t["name"]: t for t in tools}
    cases = [c for c in queries_dataset if c["tool"] in tool_by_name]

    if args.finetuned_model:
        queries = [c["query"] for c in cases]
        tool_embs, q_embs = embed_with_sentence_transformer(args.finetuned_model, tools, queries)
        tool_names = [t["name"] for t in tools]
        hits1 = hits5 = 0
        for i, case in enumerate(cases):
            scores = tool_embs @ q_embs[i]
            order = np.argsort(-scores)
            gold_idx = tool_names.index(case["tool"])
            hits1 += int(order[0] == gold_idx)
            hits5 += int(gold_idx in order[:5])
    else:
        llm_emb = llama_cpp.Llama(model_path=args.embed_model, embedding=True, verbose=False)
        tool_embs, tool_names = embed_tools(llm_emb, tools)
        hits1 = hits5 = 0
        for case in cases:
            q_emb = embed_query(llm_emb, case["query"])
            top1 = retrieve_hybrid(case["query"], q_emb, tool_embs, tool_names, tools, 1)[0]
            top5 = retrieve_hybrid(case["query"], q_emb, tool_embs, tool_names, tools, 5)
            hits1 += int(top1 == case["tool"])
            hits5 += int(case["tool"] in top5)

    n = max(len(cases), 1)
    recall1 = hits1 / n
    recall5 = hits5 / n
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, embeddings=tool_embs, names=np.array(tool_names))
    print(json.dumps({"recall@1": recall1, "recall@5": recall5, "n": n, "output": str(out)}, indent=2))
    if recall1 < args.recall_gate:
        print(f"FAIL: recall@1 {recall1:.3f} < gate {args.recall_gate}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
