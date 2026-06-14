#!/usr/bin/env python3
"""Scale benchmark on MCP-Zero corpus with Wilson CIs and optional second model."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from bench_e2e import main as e2e_main  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="MCP corpus scale e2e benchmark.")
    p.add_argument("--corpus", default="results/mcp_zero_corpus/corpus.json")
    p.add_argument("--queries", default="results/mcp_zero_corpus/queries.json")
    p.add_argument("--compile-top-n", type=int, default=100, help="Compile ATBs for top-N tools.")
    p.add_argument("--query-limit", type=int, default=1000)
    p.add_argument("--model", default="")
    p.add_argument("--llama8-model", default="", help="Llama-3.1-8B GGUF for second-model arm.")
    p.add_argument("--output", default="results/bench_mcp_corpus.json")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    corpus_path = Path(args.corpus)
    queries_path = Path(args.queries)
    if not corpus_path.exists() or not queries_path.exists():
        print("Run scripts/load_mcp_zero_corpus.py first")
        return 1

    corpus = json.loads(corpus_path.read_text())
    queries = json.loads(queries_path.read_text())["queries"][: args.query_limit]
    tools = corpus["tools"][: max(args.compile_top_n, 10)]

    # Delegate to bench_e2e with injected corpus via env (minimal fork)
    os.environ["NEXUS_BENCH_TOOLS_JSON"] = json.dumps(tools)
    os.environ["NEXUS_BENCH_QUERIES_JSON"] = json.dumps(queries)
    argv = ["bench_e2e.py", "--query-limit", str(len(queries)), "--output", args.output]
    if args.model:
        argv.extend(["--model", args.model])
    if args.llama8_model:
        argv.extend(["--llama8-model", args.llama8_model])
    if args.smoke:
        argv.append("--smoke")
    sys.argv = argv
    e2e_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
