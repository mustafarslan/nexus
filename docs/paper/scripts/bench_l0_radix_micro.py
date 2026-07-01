#!/usr/bin/env python3
"""Experiment 5: L0 exact-token radix microbenchmark — 10 seeds x 100 requests = 1000."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from _common import DATA_DIR, RAW_DIR, REPO_ROOT, ensure_dirs, load_json, percentiles_us, resolve_model, write_csv


def run_phase28_seeds(model: str, n_seeds: int = 10) -> list[dict]:
    binary = REPO_ROOT / "build" / "bench_phase28_radix_prefix"
    if not binary.exists():
        subprocess.run(["cmake", "--build", str(REPO_ROOT / "build"), "--target", "bench_phase28_radix_prefix"], check=True)
    artifacts = []
    for seed in range(n_seeds):
        out = RAW_DIR / f"l0_seed_{seed}.json"
        proc = subprocess.run(
            [str(binary), "--model", model, "--seed", str(seed), "--output", str(out)],
            cwd=REPO_ROOT,
            check=False,
        )
        if not out.exists():
            raise RuntimeError(f"phase28 seed {seed} failed (exit {proc.returncode}) without output")
        artifacts.append(load_json(out))
        if proc.returncode not in (0, 2):
            raise RuntimeError(f"phase28 seed {seed} failed with exit {proc.returncode}")
    return artifacts


def aggregate_l0(artifacts: list[dict]) -> dict:
    hit_rates: list[float] = []
    copy_samples: list[float] = []
    for art in artifacts:
        metrics = art.get("metrics", {})
        hit_rates.extend(metrics.get("hit_rate", {}).get("raw_samples", []))
        for key in ("copy_p50_us",):
            copy_samples.extend(metrics.get(key, {}).get("raw_samples", []))
        # Also collect per-run p99 if present
    # Re-run aggregation: each artifact has summary p50/p99 — expand via raw or summary
    all_p50: list[float] = []
    all_p99: list[float] = []
    for art in artifacts:
        m = art.get("metrics", {})
        all_p50.append(m.get("copy_p50_us", {}).get("summary", {}).get("p50", 0.0))
        all_p99.append(m.get("copy_p99_us", {}).get("summary", {}).get("p50", 0.0))
        hit_rates.append(m.get("hit_rate", {}).get("summary", {}).get("mean", 0.0))

    return {
        "hit_rate_pct": float(np.mean(hit_rates)) * 100.0 if hit_rates else 0.0,
        "copy_p50_us": float(np.percentile(all_p50, 50)) if all_p50 else 0.0,
        "copy_p99_us": float(np.percentile(all_p99, 99)) if all_p99 else 0.0,
        "n_requests": len(artifacts) * 100,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    model = resolve_model()

    if args.skip_run:
        artifacts = [load_json(RAW_DIR / f"l0_seed_{s}.json") for s in range(args.seeds)]
    else:
        artifacts = run_phase28_seeds(model, args.seeds)

    agg = aggregate_l0(artifacts)
    rows = [
        {
            "mode": "exact_token",
            "hit_rate_pct": round(agg["hit_rate_pct"], 2),
            "fragmentation_pct": round(100.0 - agg["hit_rate_pct"], 2),
            "copy_p50_us": round(agg["copy_p50_us"], 3),
            "copy_p99_us": round(agg["copy_p99_us"], 3),
            "n_requests": agg["n_requests"],
            "provenance": "bench_phase28_radix_prefix C++ microbench",
        },
        {
            "mode": "chunk32_simulated",
            "hit_rate_pct": 0.0,
            "fragmentation_pct": 100.0,
            "copy_p50_us": "",
            "copy_p99_us": "",
            "n_requests": 0,
            "provenance": "RELEASE_V1 graveyard: FNV-1a 32-token chunking 0% L0 hit",
        },
    ]

    write_csv(
        DATA_DIR / "l0_radix_metrics.csv",
        ["mode", "hit_rate_pct", "fragmentation_pct", "copy_p50_us", "copy_p99_us", "n_requests", "provenance"],
        rows,
        comment_lines=["L0 radix exact-token walk — C++ hardware timing only"],
    )
    print(f"Wrote {DATA_DIR / 'l0_radix_metrics.csv'}")
    print(f"  exact_token: hit={agg['hit_rate_pct']:.1f}% copy_p50={agg['copy_p50_us']:.2f}us")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
