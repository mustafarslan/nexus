#!/usr/bin/env python3
"""H1: Build held-out query sets for scale tiers (N=100/1k/10k)."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

import sys
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "test"))
from bench_routing_accuracy import queries_dataset, load_first_10_tools  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Build scale query sets for retrieval eval.")
    p.add_argument("--output-dir", default="results/scale_query_sets")
    p.add_argument("--queries-per-tier", type=int, default=100)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    gold = load_first_10_tools()
    gold_names = {t["name"] for t in gold}
    base = [q for q in queries_dataset if q["tool"] in gold_names]

    tiers = {
        "N10": {"n_distractors": 0, "queries": base[:100]},
        "N100": {"n_distractors": 90, "queries": base[:100]},
        "N1000": {"n_distractors": 990, "queries": base[:100]},
        "N10000": {"n_distractors": 9990, "queries": base[:100]},
    }

    manifest = []
    for name, spec in tiers.items():
        path = out / f"{name}_queries.json"
        payload = {
            "tier": name,
            "n_distractors": spec["n_distractors"],
            "gold_queries": spec["queries"],
            "note": "Gold GitHub queries; distractor count for corpus sizing only",
        }
        path.write_text(json.dumps(payload, indent=2))
        manifest.append({"tier": name, "path": str(path), "n_queries": len(spec["queries"])})
        print(f"Wrote {path} ({len(spec['queries'])} queries)")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
