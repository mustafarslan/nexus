#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate figure-ready summaries from Nexus benchmark artifacts")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default="results/summary.csv")
    return parser.parse_args()


def plot_e2e_pareto(results_dir: Path) -> Path | None:
    """TTFT P50 vs tool-hit Pareto figure from bench_e2e.json."""
    e2e_path = results_dir / "bench_e2e.json"
    if not e2e_path.exists():
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    artifact = json.loads(e2e_path.read_text())
    by_arm = artifact.get("by_arm") or (artifact.get("extra") or {}).get("by_arm", {})
    if not by_arm:
        return None
    xs, ys, labels = [], [], []
    for arm, stats in sorted(by_arm.items()):
        if arm in ("B_rag_mcp", "N1r_ce"):
            continue
        ttft = stats.get("ttft_us_p50", 0) / 1000.0
        acc = stats.get("tool_accuracy", 0)
        if ttft <= 0:
            continue
        xs.append(ttft)
        ys.append(acc)
        labels.append(arm)
    if not xs:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs, ys, s=60)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("TTFT P50 (ms)")
    ax.set_ylabel("Tool-hit rate")
    ax.set_title("Nexus e2e: latency vs accuracy Pareto")
    ax.grid(True, alpha=0.3)
    out = results_dir / "bench_e2e_pareto.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    pareto = plot_e2e_pareto(results_dir)
    if pareto:
        print(f"Wrote Pareto figure {pareto}")
    rows = []
    for path in sorted(Path(args.results_dir).glob("*.json")):
        if path.name.endswith(".invalid.json"):
            continue
        artifact = json.loads(path.read_text())
        for metric_name, metric in artifact.get("metrics", {}).items():
            summary = metric.get("summary", {})
            rows.append({
                "artifact": path.name,
                "benchmark": artifact.get("benchmark", ""),
                "metric": metric_name,
                "unit": metric.get("unit", ""),
                "n": summary.get("n", 0),
                "mean": summary.get("mean", 0.0),
                "std": summary.get("std", 0.0),
                "p50": summary.get("p50", 0.0),
                "p90": summary.get("p90", 0.0),
                "p99": summary.get("p99", 0.0),
                "git_sha": artifact.get("git_sha", ""),
                "llama_cpp_sha": artifact.get("llama_cpp_sha", ""),
                "model_hash": artifact.get("model_hash", ""),
                "hardware": artifact.get("hardware", ""),
            })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        fieldnames = [
            "artifact",
            "benchmark",
            "metric",
            "unit",
            "n",
            "mean",
            "std",
            "p50",
            "p90",
            "p99",
            "git_sha",
            "llama_cpp_sha",
            "model_hash",
            "hardware",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} metric summaries to {output}")


if __name__ == "__main__":
    main()
