#!/usr/bin/env python3
"""Tombstone superseded benchmark artifacts (.invalid.json pattern)."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_TOMBSTONES = [
    ("results/bench_phaseB_selective_recompute_full.json", "pre-fix recompute path; G4 verdict void"),
    ("results/g4_gate_verdict.json", "computed on pre-fix ablation artifact"),
]


def tombstone(path: Path, reason: str, dry_run: bool = False) -> None:
    if not path.exists():
        print(f"Skip missing {path}")
        return
    invalid = path.with_suffix(".invalid.json")
    if invalid.exists():
        print(f"Already tombstoned {path}")
        return
    data = json.loads(path.read_text())
    data["_tombstone"] = {
        "reason": reason,
        "original_path": str(path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if dry_run:
        print(f"Would tombstone {path} -> {invalid}")
        return
    invalid.write_text(json.dumps(data, indent=2) + "\n")
    path.unlink()
    print(f"Tombstoned {path} -> {invalid}")


def parse_args():
    p = argparse.ArgumentParser(description="Tombstone superseded benchmark artifacts.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--path", action="append", default=[])
    p.add_argument("--reason", default="superseded by Phases K-O remediation")
    return p.parse_args()


def main():
    args = parse_args()
    if args.path:
        for p in args.path:
            tombstone(Path(p), args.reason, args.dry_run)
    else:
        for p, reason in DEFAULT_TOMBSTONES:
            tombstone(Path(p), reason, args.dry_run)
    # Smoke e2e with n=5
    e2e = Path("results/bench_e2e.json")
    if e2e.exists():
        data = json.loads(e2e.read_text())
        n = (data.get("config") or {}).get("n_queries", 0)
        if n < 50:
            tombstone(e2e, f"smoke run n={n}; destroyed full Phase D artifact", args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
