#!/usr/bin/env python3
"""G4/L2: Apply pre-registered viability gate from suffix ablation results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="G4 splice viability gate (suffix framing).")
    p.add_argument("--input", default="results/bench_phaseB_suffix_recompute_full.json")
    p.add_argument("--p-start", type=int, default=1024)
    p.add_argument("--max-suffix-pct", type=float, default=20.0)
    p.add_argument("--kl-threshold", type=float, default=0.1)
    p.add_argument("--top1-threshold", type=float, default=0.99)
    p.add_argument("--output", default="results/g4_gate_verdict.json")
    return p.parse_args()


def _suffix_pct(row: dict) -> float:
    return row.get("nominal_suffix_pct", row.get("recompute_pct", 0))


def main():
    args = parse_args()
    path = Path(args.input)
    if not path.exists():
        print(f"Missing ablation artifact: {path}")
        return 1

    data = json.loads(path.read_text())
    by_config = data.get("by_config", {})
    record_rows = data.get("records") or []
    if not record_rows:
        record_rows = (data.get("benchmark") or {}).get("records") or []
    if not by_config and record_rows:
        by_config = {}
        for i, row in enumerate(record_rows):
            key = row.get("arm") or row.get("config") or f"record_{i}"
            by_config[key] = row
    candidates = []
    for key, row in by_config.items():
        if row.get("p_start") != args.p_start:
            continue
        s_pct = _suffix_pct(row)
        if s_pct <= 0 or s_pct > args.max_suffix_pct:
            continue
        kl = row.get("kl_ref_to_splice", {}).get("mean", 999)
        top1 = row.get("top1_agreement_rate", 0)
        lat = row.get("splice_recompute_us", {}).get("p50", 0)
        actual_tok = row.get("actual_recompute_tokens_mean", 0)
        candidates.append({
            "key": key,
            "selector": row.get("selector"),
            "nominal_suffix_pct": s_pct,
            "actual_recompute_tokens_mean": actual_tok,
            "kl_mean": kl,
            "top1_rate": top1,
            "latency_p50_ms": lat / 1000.0,
        })

    # reprefill anchor from s=100 row if present
    reprefill_p50 = 1e18
    for row in by_config.values():
        if row.get("p_start") == args.p_start and _suffix_pct(row) == 100.0:
            reprefill_p50 = row.get("splice_recompute_us", {}).get("p50", reprefill_p50)
            break
    anchors = data.get("anchors", {})
    if reprefill_p50 >= 1e17:
        reprefill_p50 = anchors.get("reprefill_100pct_us", {}).get("p50", 1e18)

    best = None
    for c in candidates:
        if c["kl_mean"] <= args.kl_threshold and c["top1_rate"] >= args.top1_threshold:
            if c["latency_p50_ms"] * 1000 <= reprefill_p50:
                if best is None or c["kl_mean"] < best["kl_mean"]:
                    best = c

    viable = best is not None
    verdict = {
        "input_artifact": str(path),
        "artifact_timestamp": data.get("timestamp"),
        "p_start": args.p_start,
        "max_suffix_pct": args.max_suffix_pct,
        "kl_threshold": args.kl_threshold,
        "top1_threshold": args.top1_threshold,
        "viable": viable,
        "best_config": best,
        "reprefill_anchor_p50_ms": reprefill_p50 / 1000.0 if reprefill_p50 < 1e17 else None,
        "recommendation": (
            "continue_splice_engineering"
            if viable
            else "re_scope_to_retrieval_prefix_cache_fsm"
        ),
        "candidates_evaluated": len(candidates),
        "framing": "suffix_fraction_with_honest_actual_recompute_tokens",
    }

    Path(args.output).write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2))
    return 0 if viable else 2


if __name__ == "__main__":
    raise SystemExit(main())
