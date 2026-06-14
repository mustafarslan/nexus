#!/usr/bin/env python3
"""Re-run Phases K-O + G4 and regenerate BENCHMARK_VERDICT with B3 baseline + bootstrap CIs."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Phase 5 full benchmark rerun.")
    p.add_argument("--skip-ko", action="store_true", help="Only regenerate verdict from existing artifacts")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    if not args.skip_ko:
        scripts = [
            "scripts/run_phases_ko.sh",
        ]
        if args.smoke:
            scripts = ["scripts/run_phases_ko_remaining.sh"]
        for script in scripts:
            path = root / script
            if path.exists():
                print(f"Running {path}")
                subprocess.run(["bash", str(path)], cwd=root, check=False)

        g4 = root / "scripts/apply_g4_gate.py"
        if g4.exists():
            subprocess.run([sys.executable, str(g4)], cwd=root, check=False)

    plot = root / "scripts/plot_from_results.py"
    if plot.exists():
        subprocess.run([sys.executable, str(plot)], cwd=root, check=False)

    summary = root / "scripts/generate_executive_summary.py"
    subprocess.run([sys.executable, str(summary), "--b3-baseline"], cwd=root, check=False)
    print("Phase 5 complete: see results/BENCHMARK_VERDICT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
