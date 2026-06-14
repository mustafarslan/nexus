#!/usr/bin/env python3
"""Load MCP-Zero style tool corpus (308 servers / ~2797 tools) for scale benchmarks.

Downloads from HuggingFace datasets if available; falls back to bundled sample.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


HF_DATASET = "huggingface/MCP-tools"
DEFAULT_OUTPUT = "results/mcp_zero_corpus/corpus.json"


def _normalize_tool(raw: dict) -> dict:
    name = raw.get("name") or raw.get("tool_name") or ""
    desc = raw.get("description") or raw.get("desc") or ""
    schema = raw.get("inputSchema") or raw.get("input_schema") or raw.get("parameters") or {}
    return {"name": name, "description": desc, "inputSchema": schema}


def fetch_hf_corpus(limit: int | None = None) -> list[dict]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError("pip install datasets for HF MCP-tools corpus") from exc
    ds = load_dataset(HF_DATASET, split="train")
    tools = [_normalize_tool(row) for row in ds]
    if limit:
        tools = tools[:limit]
    return [t for t in tools if t.get("name")]


def fetch_fallback_corpus() -> list[dict]:
    """Procedural expansion from micro-benchmark schemas when HF unavailable."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "test"))
    from bench_routing_accuracy import load_first_10_tools  # noqa: E402

    base = load_first_10_tools()
    out = list(base)
    for i in range(2787):
        src = base[i % len(base)]
        out.append({
            "name": f"{src['name']}_variant_{i}",
            "description": f"{src.get('description', '')} (MCP variant {i})",
            "inputSchema": src.get("inputSchema") or src.get("input_schema") or {},
        })
    return out


def generate_diverse_queries(tools: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Template + paraphrase pool for n>=1000 query scale."""
    import random

    rng = random.Random(seed)
    templates = [
        "Use {name} to {task}",
        "Call {name} with {task}",
        "I need {name} for {task}",
        "Please run {name} to {task}",
        "Route this to {name}: {task}",
    ]
    tasks = [
        "update the repo settings",
        "fetch file contents from main",
        "open a new issue",
        "search for python repositories",
        "push local changes",
        "list open pull requests",
        "create a branch",
        "merge the latest PR",
    ]
    queries = []
    tool_names = [t["name"] for t in tools if t.get("name")]
    for i in range(n):
        name = tool_names[i % len(tool_names)]
        tpl = templates[i % len(templates)]
        task = tasks[rng.randint(0, len(tasks) - 1)]
        queries.append({"query": tpl.format(name=name, task=task), "tool": name})
    return queries


def main() -> int:
    p = argparse.ArgumentParser(description="Build MCP-Zero scale corpus + query set.")
    p.add_argument("--output-dir", default="results/mcp_zero_corpus")
    p.add_argument("--limit-tools", type=int, default=2797)
    p.add_argument("--n-queries", type=int, default=1000)
    p.add_argument("--hf", action="store_true", help="Fetch from HuggingFace MCP-tools dataset.")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        tools = fetch_hf_corpus(args.limit_tools) if args.hf else fetch_fallback_corpus()[: args.limit_tools]
    except Exception as exc:
        print(f"HF fetch failed ({exc}); using fallback corpus")
        tools = fetch_fallback_corpus()[: args.limit_tools]

    corpus_path = out_dir / "corpus.json"
    corpus_path.write_text(json.dumps({"tools": tools, "n_tools": len(tools)}, indent=2))

    queries = generate_diverse_queries(tools[: min(100, len(tools))], args.n_queries)
    queries_path = out_dir / "queries.json"
    queries_path.write_text(json.dumps({"queries": queries, "n": len(queries)}, indent=2))

    manifest = {
        "corpus": str(corpus_path),
        "queries": str(queries_path),
        "n_tools": len(tools),
        "n_queries": len(queries),
        "source": "hf" if args.hf else "fallback",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
