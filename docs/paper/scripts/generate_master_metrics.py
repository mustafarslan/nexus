#!/usr/bin/env python3
"""Aggregate all paper CSVs into PAPER_MASTER_METRICS.json."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from _common import DATA_DIR, PAPER_ROOT, ensure_dirs, git_sha, hardware_str, write_json


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            break
        f.seek(0)
        # Skip comment lines
        lines = [ln for ln in f if not ln.startswith("#")]
    if not lines:
        return []
    reader = csv.DictReader(lines)
    return list(reader)


def summarize_cdf(rows: list[dict]) -> dict:
    by_arm: dict[str, list[float]] = {}
    hits: dict[str, list[int]] = {}
    for r in rows:
        arm = r["arm"]
        by_arm.setdefault(arm, []).append(float(r["ttft_ms"]))
        hits.setdefault(arm, []).append(int(r.get("tool_hit", 0)))
    import numpy as np
    out = {}
    for arm, lats in by_arm.items():
        arr = np.array(lats)
        h = hits.get(arm, [])
        out[arm] = {
            "ttft_p50_ms": float(np.percentile(arr, 50)),
            "ttft_p90_ms": float(np.percentile(arr, 90)),
            "ttft_p99_ms": float(np.percentile(arr, 99)),
            "tool_hit_accuracy": sum(h) / len(h) if h else 0.0,
            "n": len(lats),
        }
    return out


def main() -> int:
    ensure_dirs()
    master = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha(),
        "hardware": hardware_str(),
        "experiments": {},
    }

    cdf = read_csv(DATA_DIR / "e2e_cdf.csv")
    if cdf:
        master["experiments"]["e2e_cdf"] = summarize_cdf(cdf)

    for name, fname in [
        ("pareto", "pareto_frontier.csv"),
        ("waterfall", "ttft_waterfall.csv"),
        ("scalability", "scalability_curve.csv"),
        ("ml_ablation", "ml_ablation_table.csv"),
        ("l0_radix", "l0_radix_metrics.csv"),
        ("physics_boundary", "physics_boundary_rope.csv"),
        ("multi_tool_paradox", "multi_tool_paradox.csv"),
    ]:
        rows = read_csv(DATA_DIR / fname)
        if rows:
            master["experiments"][name] = rows

    out = DATA_DIR / "PAPER_MASTER_METRICS.json"
    write_json(out, master)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
