#!/usr/bin/env python3
"""Batch-compile MCP schemas to .atb with positional buckets and optional sink prefix."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_BUCKETS = (256, 1024, 4096)


def main():
    p = argparse.ArgumentParser(description="Batch KV compiler wrapper (single model load).")
    p.add_argument("--model", required=True)
    p.add_argument("--manifest", help="JSON list of {schema, output} objects")
    p.add_argument("--schemas-dir", help="Directory of .json schemas")
    p.add_argument("--output-dir", help="Output directory for .atb files")
    p.add_argument("--compiler", default="./build/nexus_kv_compiler")
    p.add_argument("--pos-buckets", default="", help="Comma-separated bucket sizes (e.g. 256,1024,4096)")
    p.add_argument("--sink-prefix", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    buckets = DEFAULT_BUCKETS
    if args.pos_buckets.strip():
        buckets = tuple(int(x.strip()) for x in args.pos_buckets.split(",") if x.strip())

    jobs: list[tuple[str, str, int]] = []
    if args.manifest:
        data = json.loads(Path(args.manifest).read_text())
        for row in data:
            schema = row["schema"]
            out = row["output"]
            base_pos = int(row.get("base_pos", buckets[0]))
            jobs.append((schema, out, base_pos))
    elif args.schemas_dir and args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for schema in sorted(Path(args.schemas_dir).glob("*.json")):
            for bucket in buckets:
                jobs.append((str(schema), str(out_dir / f"{schema.stem}.p{bucket}.atb"), bucket))
    else:
        print("Provide --manifest or --schemas-dir + --output-dir", file=sys.stderr)
        return 1

    batch_file = Path("results/batch_compile_list.txt")
    batch_file.parent.mkdir(parents=True, exist_ok=True)
    meta_rows = []
    with batch_file.open("w") as f:
        for schema, output, base_pos in jobs:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            f.write(f"{schema},{output}\n")
            meta_rows.append({"schema": schema, "output": output, "base_pos": base_pos})

    cmd = [args.compiler, "--model", args.model, "--batch-list", str(batch_file)]
    if args.sink_prefix:
        cmd.append("--sink-prefix")
    if args.verbose:
        cmd.append("--verbose")
    # Pos bucket applied per-job via sidecar meta (compiler uses --base-pos from env per line — extend via wrapper)
    # For multi-bucket dirs, run sequential compiles with per-job base-pos.
    if args.schemas_dir and args.output_dir and not args.manifest:
        for schema, output, base_pos in jobs:
            one = Path("results/batch_compile_one.txt")
            one.write_text(f"{schema},{output}\n")
            job_cmd = [
                args.compiler,
                "--model",
                args.model,
                "--batch-list",
                str(one),
                "--base-pos",
                str(base_pos),
                "--pos-bucket",
                str(base_pos),
            ]
            if args.sink_prefix:
                job_cmd.append("--sink-prefix")
            if args.verbose:
                job_cmd.append("--verbose")
            print(f"Compiling {output} @ base_pos={base_pos}")
            subprocess.run(job_cmd, check=True)
    else:
        print(f"Batch compiling {len(jobs)} schemas via {batch_file}")
        subprocess.run(cmd, check=True)

    meta_path = Path("results/batch_compile_meta.json")
    meta_path.write_text(json.dumps({"jobs": meta_rows, "buckets": list(buckets)}, indent=2) + "\n")
    print(f"Wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
