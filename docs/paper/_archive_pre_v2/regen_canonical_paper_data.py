#!/usr/bin/env python3
"""Regenerate canonical paper data + macro figures from results/v1.1_canonical/ ONLY.

Artifact-driven: every number traces to a committed canonical JSON. No hardcoded
metrics. Outputs land in docs/paper/data/v1_1/ and docs/paper/figures/ (suffixed
_v1_1). Run: .venv/bin/python docs/paper/scripts/regen_canonical_paper_data.py
"""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
CANON = REPO / "results" / "v1.1_canonical" / "raw"
OUT = REPO / "docs" / "paper" / "data" / "v1_1"
FIG = REPO / "docs" / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Paper arm -> canonical e2e arm (TRUE TTFT incl. first token).
MACRO = [("B_FC (full-context)", "B1"), ("B_RP (retrieve+prefill)", "B3"), ("Nexus (N1x)", "N1x")]


def load(name):
    return json.load(open(CANON / name))


def e2e_ttft_ms(by_records, arm):
    return sorted(float(r["ttft_us"]) / 1000.0 for r in by_records if r.get("arm") == arm)


def pct(xs, p):
    return float(np.percentile(xs, p)) if xs else 0.0


def main():
    e2e = load("e2e_clean_n30.json")
    recs = e2e.get("records", [])
    ba = e2e.get("by_arm", {})

    # --- macro table + pareto (TRUE TTFT) ---
    macro_rows, pareto_rows, cdf_rows = [], [], []
    for label, arm in MACRO:
        t = e2e_ttft_ms(recs, arm)
        acc = ba.get(arm, {}).get("tool_accuracy", 0.0)
        macro_rows.append({
            "arm": label, "ttft_p50_ms": round(pct(t, 50), 1),
            "ttft_p90_ms": round(pct(t, 90), 1), "ttft_p99_ms": round(pct(t, 99), 1),
            "accuracy": acc, "n": len(t),
        })
        pareto_rows.append({"arm": label, "accuracy": acc, "ttft_p50_ms": round(pct(t, 50), 1), "n": len(t)})
        for i, lat in enumerate(t):
            cdf_rows.append({"arm": label, "idx": i, "ttft_ms": round(lat, 3)})

    # --- gateway routing-only (n=500 pooled) ---
    gw = []
    for f in sorted(glob.glob(str(CANON / "gateway_run*.json"))):
        d = json.load(open(f))
        gw += [float(r["ttft_us"]) / 1000.0 for r in d["records"] if r.get("arm") == "GW_route"]
    gw_acc = json.load(open(sorted(glob.glob(str(CANON / "gateway_run*.json")))[0]))["by_arm"]["GW_route"]["tool_accuracy"]

    # --- L0 (10 seeds) ---
    hit, p50 = [], []
    for s in range(10):
        m = load(f"l0_seed_{s}.json")["metrics"]
        hit.append(m["hit_rate"]["summary"]["mean"])
        p50.append(m["copy_p50_us"]["summary"]["p50"])

    # --- recall ---
    rc = load("recall_miss_analysis.json")

    def write(name, fieldnames, rows, header_comment):
        with (OUT / name).open("w", newline="") as f:
            f.write(f"# {header_comment}\n")
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    prov = "GENERATED from results/v1.1_canonical/ by regen_canonical_paper_data.py. TRUE TTFT (incl. first token) unless noted."
    write("macro_ttft.csv", ["arm", "ttft_p50_ms", "ttft_p90_ms", "ttft_p99_ms", "accuracy", "n"], macro_rows, prov)
    write("pareto_frontier.csv", ["arm", "accuracy", "ttft_p50_ms", "n"], pareto_rows, prov)
    write("e2e_cdf.csv", ["arm", "idx", "ttft_ms"], cdf_rows, prov)
    write("routing_micro.csv", ["metric", "value", "unit", "n", "source"], [
        {"metric": "gateway_route_splice_p50", "value": round(pct(gw, 50), 1), "unit": "ms", "n": len(gw), "source": "gateway_run[1-5] (route-only, decode_us=0)"},
        {"metric": "gateway_accuracy", "value": gw_acc, "unit": "frac", "n": len(gw), "source": "gateway_run[1-5]"},
        {"metric": "l0_copy_p50", "value": round(float(np.median(p50)), 2), "unit": "us", "n": 10, "source": "l0_seed_[0-9]"},
        {"metric": "l0_hit_rate", "value": round(float(np.mean(hit)) * 100, 1), "unit": "pct", "n": 10, "source": "l0_seed_[0-9]"},
    ], prov)
    write("recall.csv", ["metric", "value"], [
        {"metric": "dense_recall_at_1", "value": rc["recall_at_1"]},
        {"metric": "dense_recall_at_5", "value": rc["recall_at_5"]},
        {"metric": "ce_recall_at_1", "value": rc["ce_recall_at_1"]},
        {"metric": "ce_fire_fraction", "value": rc["ce_rerank_fraction"]},
        {"metric": "margin_threshold", "value": rc["margin_threshold"]},
        {"metric": "meets_target_0.90", "value": rc["meets_target"]},
    ], prov)

    # --- figures ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # CDF (true TTFT, log-x)
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        styles = {"B_FC (full-context)": "dotted", "B_RP (retrieve+prefill)": "dashed", "Nexus (N1x)": "solid"}
        for label, arm in MACRO:
            t = sorted(e2e_ttft_ms(recs, arm))
            if not t:
                continue
            y = np.arange(1, len(t) + 1) / len(t)
            ax.plot(t, y, label=label, linestyle=styles.get(label, "solid"))
        ax.set_xscale("log")
        ax.set_xlabel("End-to-end TTFT (ms, incl. first token)")
        ax.set_ylabel("CDF")
        ax.set_title("True-TTFT CDF (n=30, clean serial)", fontsize=8)
        ax.legend(fontsize=6, loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / "fig_e2e_cdf_v1_1.pdf")
        plt.close(fig)

        # Pareto (true TTFT)
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        for row in pareto_rows:
            ax.scatter(row["ttft_p50_ms"], row["accuracy"], s=40)
            ax.annotate(row["arm"], (row["ttft_p50_ms"], row["accuracy"]), fontsize=6,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("True TTFT P50 (ms)")
        ax.set_ylabel("Tool-hit accuracy")
        ax.set_title("Accuracy vs. true TTFT (n=30)", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / "fig_pareto_frontier_v1_1.pdf")
        plt.close(fig)
        print("figures: fig_e2e_cdf_v1_1.pdf, fig_pareto_frontier_v1_1.pdf")
    except Exception as e:
        print(f"WARN figures skipped: {e}")

    print(f"macro: {macro_rows}")
    print(f"gateway route-only p50={pct(gw,50):.1f}ms n={len(gw)}; L0 hit={np.mean(hit)*100:.1f}% copy={np.median(p50):.2f}us")
    print(f"wrote canonical CSVs to {OUT}")


if __name__ == "__main__":
    main()
