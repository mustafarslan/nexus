#!/usr/bin/env python3
"""Pad tool registry to arbitrary N via MCP-zero / GitHub schema duplication."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from _common import REPO_ROOT, setup_paths

setup_paths()

from bench_routing_accuracy import load_first_10_tools  # noqa: E402


def load_base_tools() -> list[dict]:
    mcp_path = REPO_ROOT / "results" / "mcp_zero_corpus" / "corpus.json"
    if mcp_path.exists():
        corpus = json.loads(mcp_path.read_text())
        tools = corpus.get("tools") or corpus.get("corpus") or []
        if tools:
            return tools
    return load_first_10_tools()


def pad_tools_to_n(n: int) -> list[dict]:
    base = load_base_tools()
    if n <= len(base):
        return base[:n]
    out = list(base)
    i = 0
    while len(out) < n:
        src = base[i % len(base)]
        suffix = len(out)
        out.append({
            "name": f"{src['name']}_pad_{suffix}",
            "description": f"{src.get('description', '')} [registry pad {suffix}]",
            "inputSchema": src.get("inputSchema") or src.get("input_schema") or {},
        })
        i += 1
    return out
