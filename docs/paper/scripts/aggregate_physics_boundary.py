#!/usr/bin/env python3
"""Experiment 6: Physics boundary — static historical data from architecture validation."""
from __future__ import annotations

from _common import DATA_DIR, ensure_dirs, write_csv

# HISTORICAL PHYSICS DATA (Sourced from docs/architecture.md prior to V1.0 freeze)
PHYSICS_BOUNDARY_DATA = [
    {"P": 256, "suffix_pct": 5.0, "strategy": "Fused Tail Suffix (Path A)", "kl_mean": 0.0076, "top1": 1.00},
    {"P": 1024, "suffix_pct": 0.0, "strategy": "Bias Calibration (No Query)", "kl_mean": 0.0000, "top1": 1.00},
    {"P": 1024, "suffix_pct": 1.5, "strategy": "LegoLink Sink (k=4)", "kl_mean": 5.7200, "top1": 0.20},
    {"P": 1024, "suffix_pct": 5.0, "strategy": "Scattered Suffix", "kl_mean": 2.7900, "top1": 0.40},
    {"P": 1024, "suffix_pct": 5.9, "strategy": "LegoLink Sink (k=16)", "kl_mean": 0.8300, "top1": 0.70},
    {"P": 1024, "suffix_pct": 10.0, "strategy": "Scattered Suffix", "kl_mean": 2.9900, "top1": 0.10},
    {"P": 1024, "suffix_pct": 20.0, "strategy": "Scattered Suffix", "kl_mean": 5.4900, "top1": 0.00},
    {"P": 1024, "suffix_pct": 100.0, "strategy": "Full Reprefill (1.43s TTFT)", "kl_mean": 0.0000, "top1": 1.00},
]

MULTI_TOOL_PARADOX_DATA = [
    {
        "topology": "Single Tool Splice (N1)",
        "tensor_kl": 0.0076,
        "e2e_routing_accuracy": 0.91,
        "diagnosis": "Causally sound",
    },
    {
        "topology": "Multi-Tool Blockmask (N1m)",
        "tensor_kl": 0.0000,
        "e2e_routing_accuracy": 0.00,
        "diagnosis": "Cross-Attention Blinding (Causal Isolation)",
    },
]

SOURCE_NOTE = "Data sourced from architectural validation records prior to V1.0 codebase freeze."


def main() -> int:
    ensure_dirs()
    physics_rows = [
        {
            "P": r["P"],
            "suffix_pct": r["suffix_pct"],
            "strategy": r["strategy"],
            "kl_mean_nats": r["kl_mean"],
            "top1_agreement": r["top1"],
            "source_note": SOURCE_NOTE,
        }
        for r in PHYSICS_BOUNDARY_DATA
    ]
    paradox_rows = [
        {**r, "source_note": SOURCE_NOTE}
        for r in MULTI_TOOL_PARADOX_DATA
    ]

    write_csv(
        DATA_DIR / "physics_boundary_rope.csv",
        ["P", "suffix_pct", "strategy", "kl_mean_nats", "top1_agreement", "source_note"],
        physics_rows,
        comment_lines=[SOURCE_NOTE],
    )
    write_csv(
        DATA_DIR / "multi_tool_paradox.csv",
        ["topology", "tensor_kl", "e2e_routing_accuracy", "diagnosis", "source_note"],
        paradox_rows,
        comment_lines=[SOURCE_NOTE],
    )
    print(f"Wrote {DATA_DIR / 'physics_boundary_rope.csv'} ({len(physics_rows)} rows)")
    print(f"Wrote {DATA_DIR / 'multi_tool_paradox.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
