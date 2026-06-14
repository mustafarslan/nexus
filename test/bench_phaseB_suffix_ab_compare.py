#!/usr/bin/env python3
"""Phase 2: A/B equivalence — legacy full-prefill vs kv-reuse harness."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

KL_TOL = 1e-4


def run(mode: str, out: Path, checkpoint: Path) -> None:
    cmd = [
        sys.executable,
        "test/bench_phaseB_suffix_recompute.py",
        "--case-limit", "2",
        "--p-starts", "256",
        "--force",
        "--output", str(out),
        "--checkpoint-jsonl", str(checkpoint),
    ]
    if mode == "legacy":
        cmd.append("--legacy-cache")
    subprocess.run(cmd, check=True)


def load_records(path: Path) -> list[dict]:
    return json.loads(path.read_text()).get("records", [])


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    legacy_out = root / "results/smoke/bench_phaseB_suffix_legacy_ab.json"
    reuse_out = root / "results/smoke/bench_phaseB_suffix_reuse_ab.json"
    legacy_ckpt = root / "results/smoke/bench_phaseB_suffix_legacy_ab.jsonl"
    reuse_ckpt = root / "results/smoke/bench_phaseB_suffix_reuse_ab.jsonl"
    legacy_out.parent.mkdir(parents=True, exist_ok=True)

    print("=== A/B: legacy-cache ===")
    run("legacy", legacy_out, legacy_ckpt)
    print("=== A/B: kv-reuse ===")
    run("reuse", reuse_out, reuse_ckpt)

    legacy = {(r["case_id"], r["p_start"], r["selector"], r["nominal_suffix_pct"]): r for r in load_records(legacy_out)}
    reuse = {(r["case_id"], r["p_start"], r["selector"], r["nominal_suffix_pct"]): r for r in load_records(reuse_out)}

    mismatches = []
    for key, lrow in legacy.items():
        rrow = reuse.get(key)
        if rrow is None:
            mismatches.append({"key": key, "error": "missing in reuse"})
            continue
        if lrow["actual_recompute_tokens"] != rrow["actual_recompute_tokens"]:
            mismatches.append({"key": key, "field": "actual_recompute_tokens", "legacy": lrow["actual_recompute_tokens"], "reuse": rrow["actual_recompute_tokens"]})
        if lrow["top1_agreement"] != rrow["top1_agreement"]:
            mismatches.append({"key": key, "field": "top1_agreement", "legacy": lrow["top1_agreement"], "reuse": rrow["top1_agreement"]})
        if abs(lrow["kl_ref_to_splice"] - rrow["kl_ref_to_splice"]) > KL_TOL:
            mismatches.append({"key": key, "field": "kl_ref_to_splice", "legacy": lrow["kl_ref_to_splice"], "reuse": rrow["kl_ref_to_splice"]})

    report = {
        "n_legacy": len(legacy),
        "n_reuse": len(reuse),
        "n_mismatches": len(mismatches),
        "mismatches": mismatches[:20],
        "pass": len(mismatches) == 0,
    }
    report_path = root / "results/smoke/bench_phaseB_suffix_ab_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
