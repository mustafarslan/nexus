#!/usr/bin/env python3
"""Generate publication-ready vector PDF figures from paper evaluation CSVs."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _common import DATA_DIR, FIGURES_DIR, ensure_dirs

# Academic style — colorblind-friendly palette
COLORS = {
    "B1_Bloat": "#d62728",
    "B3_Baseline": "#ff7f0e",
    "Nexus_Production": "#2ca02c",
    "PathA_Splice": "#1f77b4",
    "PathB_L0Hit": "#9467bd",
}


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]
    if len(lines) < 2:
        return []
    return list(csv.DictReader(lines))


def setup_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "figure.figsize": (6, 4),
    })


ARM_LABELS = {
    "B1_Bloat": "$B_{FC}$",
    "B3_Baseline": "$B_{RP}$",
    "Nexus_Production": "Nexus Production",
}

PATH_LABELS = {
    "PathA_Splice": "Path A (Splice)",
    "PathB_L0Hit": "Path B (Fallback)",
}


def plot_cdf() -> None:
    rows = read_csv(DATA_DIR / "e2e_cdf.csv")
    if not rows:
        return
    by_arm: dict[str, list[float]] = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(float(r["ttft_ms"]))
    fig, ax = plt.subplots()
    for arm, lats in sorted(by_arm.items()):
        sorted_lats = np.sort(lats)
        y = np.arange(1, len(sorted_lats) + 1) / len(sorted_lats)
        mapped_label = ARM_LABELS.get(arm, arm)
        ax.plot(sorted_lats, y, label=mapped_label, color=COLORS.get(arm, None), linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("TTFT (ms, log scale)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("End-to-End Latency CDF")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_e2e_cdf.pdf")
    plt.close(fig)


def plot_pareto() -> None:
    rows = read_csv(DATA_DIR / "pareto_frontier.csv")
    if not rows:
        return
    fig, ax = plt.subplots()
    for r in rows:
        arm = r["arm"]
        mapped_label = ARM_LABELS.get(arm, arm)
        ax.scatter(
            float(r["ttft_p50_ms"]), float(r["tool_hit_accuracy"]),
            s=120, label=mapped_label, color=COLORS.get(arm, None), zorder=3,
        )
        ax.annotate(mapped_label, (float(r["ttft_p50_ms"]), float(r["tool_hit_accuracy"])),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("TTFT P50 (ms)")
    ax.set_ylabel("E2E Tool-Hit Accuracy")
    ax.set_title("Accuracy--Latency Pareto Frontier")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_pareto_frontier.pdf")
    plt.close(fig)


def plot_waterfall() -> None:
    rows = read_csv(DATA_DIR / "ttft_waterfall.csv")
    if not rows:
        return
    paths = sorted({r["path"] for r in rows if r["stage"] != "total_stacked"})
    fig, axes = plt.subplots(len(paths), 1, figsize=(8, 2.5 * len(paths)), squeeze=False)
    stage_labels = {
        "tokenization_intent": "Tokenize+Intent",
        "dense_embed": "Dense Embed",
        "l1_slb_fsm": "L1 SLB+FSM",
        "ce_rerank_invoked": "CE Rerank",
        "ce_rerank_amortized": "CE (amort.)",
        "l0_radix_copy": "L0 Radix Copy",
        "atb_zero_copy_splice": "ATB Splice",
        "suffix_recompute_first_token": "Recompute+Tok1",
    }
    for i, path in enumerate(paths):
        ax = axes[i, 0]
        stages = [r for r in rows if r["path"] == path and r["stage"] != "total_stacked"]
        stages = sorted(stages, key=lambda r: int(r["stage_order"]))
        labels = [stage_labels.get(s["stage"], s["stage"]) for s in stages]
        vals = [float(s["latency_ms"]) for s in stages if float(s["latency_ms"]) > 0]
        labels = [l for l, v in zip(labels, [float(s["latency_ms"]) for s in stages]) if v > 0]
        if not vals:
            continue
        left = 0.0
        for j, (lab, val) in enumerate(zip(labels, vals)):
            ax.barh(0, val, left=left, height=0.5, label=lab)
            left += val
        mapped_path = PATH_LABELS.get(path, path)
        ax.set_yticks([0])
        ax.set_yticklabels([mapped_path])
        ax.set_xlabel("Latency (ms)")
        ax.set_title(f"TTFT Decomposition: {mapped_path}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_ttft_waterfall.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_scalability() -> None:
    rows = read_csv(DATA_DIR / "scalability_curve.csv")
    if not rows:
        return
    ns = [int(r["n_tools"]) for r in rows]
    slb = [float(r["slb_scan_p50_us"]) for r in rows]
    mem = [float(r["memory_mb"]) for r in rows]

    fig, ax1 = plt.subplots()
    color1 = "#1f77b4"
    ax1.plot(ns, slb, "o-", color=color1, linewidth=2, markersize=7, label="SLB P50 ($\\mu$s)")
    ax1.set_xlabel("Tool Count $N$")
    ax1.set_ylabel("SLB Scan P50 ($\\mu$s)", color=color1)
    ax1.set_xscale("log")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "#ff7f0e"
    ax2.plot(ns, mem, "s--", color=color2, linewidth=2, markersize=7, label="Memory (MB)")
    ax2.set_ylabel("Memory Footprint (MB)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title("Registry Scalability")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_scalability.pdf")
    plt.close(fig)


def main() -> int:
    ensure_dirs()
    setup_style()
    plot_cdf()
    plot_pareto()
    plot_waterfall()
    plot_scalability()
    print(f"Wrote figures to {FIGURES_DIR}/")
    for f in sorted(FIGURES_DIR.glob("fig_*.pdf")):
        print(f"  {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
