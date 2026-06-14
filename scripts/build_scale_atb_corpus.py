#!/usr/bin/env python3
"""Batch-compile ATBs for Phase E scale corpora (N1000/N10000 tiers)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Compile ATBs for scale corpora.")
    p.add_argument("--model", required=True)
    p.add_argument("--corpus-dir", default="results/phaseE_corpora")
    p.add_argument("--tiers", default="N100,N1000,N10000")
    p.add_argument("--compiler", default="./build/nexus_kv_compiler")
    p.add_argument("--pos-bucket", type=int, default=256)
    p.add_argument("--sink-prefix", action="store_true")
    args = p.parse_args()

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    for tier in tiers:
        corpus_path = Path(args.corpus_dir) / tier / "corpus.json"
        if not corpus_path.exists():
            print(f"Skip missing {corpus_path}")
            continue
        corpus = json.loads(corpus_path.read_text())
        tools = corpus.get("tools") or corpus.get("entries") or []
        out_dir = Path(args.corpus_dir) / tier / "atbs"
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = []
        for i, tool in enumerate(tools):
            schema_path = out_dir / f"tool_{i}.schema.json"
            atb_path = out_dir / f"tool_{i}.p{args.pos_bucket}.atb"
            schema_path.write_text(json.dumps(tool, indent=2))
            manifest.append({"schema": str(schema_path), "output": str(atb_path), "base_pos": args.pos_bucket})
            if atb_path.exists():
                continue
            cmd = [
                args.compiler,
                "--model",
                args.model,
                "--schema",
                str(schema_path),
                "--output",
                str(atb_path),
                "--base-pos",
                str(args.pos_bucket),
                "--pos-bucket",
                str(args.pos_bucket),
            ]
            if args.sink_prefix:
                cmd.append("--sink-prefix")
            print(f"Compiling {atb_path.name} ({tier})")
            subprocess.run(cmd, check=True)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Tier {tier}: {len(manifest)} ATBs under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
