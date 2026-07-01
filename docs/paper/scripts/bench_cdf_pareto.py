#!/usr/bin/env python3
"""Experiment 1: E2E latency CDF and Pareto frontier (B1, B3, Nexus Production)."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _common import (
    DATA_DIR,
    RAW_DIR,
    REPO_ROOT,
    ensure_dirs,
    load_json,
    percentiles_us,
    resolve_model,
    setup_paths,
    write_csv,
)

ARM_MAP = {
    "B1_Bloat": "B1",
    "B3_Baseline": "B3",
    "Nexus_Production": "GW_route",
}


def run_frozen_benches(model: str, force: bool) -> tuple[Path, Path]:
    e2e_out = RAW_DIR / "e2e_full.json"
    gw_out = RAW_DIR / "gateway_full.json"
    force_flag = ["--force"] if force else []

    if not e2e_out.exists() or force:
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "test" / "bench_e2e.py"),
                "--model", model,
                "--query-limit", "100",
                "--output", str(e2e_out),
                *force_flag,
            ],
            cwd=REPO_ROOT,
            check=True,
        )

    ce_dir = REPO_ROOT / "results" / "tool_cross_encoder_finetuned_v3"
    if ce_dir.exists():
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "test" / "bench_gateway_e2e.py"),
                "--model", model,
                "--query-limit", "100",
                "--output", str(gw_out),
                *force_flag,
            ],
            cwd=REPO_ROOT,
            check=True,
        )
    else:
        print(f"WARNING: CE v3 missing at {ce_dir}; using N1x from e2e as Nexus_Production proxy")
    return e2e_out, gw_out


def extract_latencies(e2e_path: Path, gw_path: Path) -> dict[str, list[dict]]:
    e2e = load_json(e2e_path)
    gw = load_json(gw_path)
    out: dict[str, list[dict]] = {}
    for paper_arm, bench_arm in ARM_MAP.items():
        if bench_arm == "GW_route":
            recs = [r for r in gw.get("records", []) if r.get("arm") == bench_arm]
            if not recs:
                recs = [
                    {**r, "arm": "GW_route"}
                    for r in e2e.get("records", [])
                    if r.get("arm") == "N1x"
                ]
        else:
            recs = [r for r in e2e.get("records", []) if r.get("arm") == bench_arm]
        recs = sorted(recs, key=lambda r: r.get("case_id", 0))
        out[paper_arm] = recs
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-run", action="store_true", help="Use existing _raw JSON artifacts")
    args = parser.parse_args()

    ensure_dirs()
    setup_paths()
    model = resolve_model()

    if args.skip_run:
        e2e_path = RAW_DIR / "e2e_full.json"
        gw_path = RAW_DIR / "gateway_full.json"
        if not e2e_path.exists() or not gw_path.exists():
            print("Missing _raw artifacts; run without --skip-run")
            return 1
    else:
        e2e_path, gw_path = run_frozen_benches(model, args.force)

    by_arm = extract_latencies(e2e_path, gw_path)

    cdf_rows: list[dict] = []
    pareto_rows: list[dict] = []
    for paper_arm, recs in by_arm.items():
        if not recs:
            print(f"WARNING: no records for {paper_arm}")
            continue
        ttfts = [float(r["ttft_us"]) for r in recs]
        hits = [bool(r.get("tool_hit")) for r in recs]
        stats = percentiles_us(ttfts)
        acc = sum(hits) / len(hits)

        for i, (r, lat) in enumerate(zip(recs, ttfts)):
            cdf_rows.append({
                "arm": paper_arm,
                "query_idx": i,
                "ttft_ms": lat / 1000.0,
                "tool_hit": int(r.get("tool_hit", False)),
            })

        pareto_rows.append({
            "arm": paper_arm,
            "tool_hit_accuracy": round(acc, 4),
            "ttft_p50_ms": round(stats["p50"] / 1000.0, 3),
            "ttft_p90_ms": round(stats["p90"] / 1000.0, 3),
            "ttft_p99_ms": round(stats["p99"] / 1000.0, 3),
            "n_queries": len(recs),
        })

    write_csv(
        DATA_DIR / "e2e_cdf.csv",
        ["arm", "query_idx", "ttft_ms", "tool_hit"],
        cdf_rows,
    )
    write_csv(
        DATA_DIR / "pareto_frontier.csv",
        ["arm", "tool_hit_accuracy", "ttft_p50_ms", "ttft_p90_ms", "ttft_p99_ms", "n_queries"],
        pareto_rows,
    )
    print(f"Wrote {DATA_DIR / 'e2e_cdf.csv'} ({len(cdf_rows)} rows)")
    print(f"Wrote {DATA_DIR / 'pareto_frontier.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
