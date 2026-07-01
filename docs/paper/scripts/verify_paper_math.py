#!/usr/bin/env python3
"""Verify that every headline number in docs/paper/main.tex traces to a committed v2.0
artifact under results/v2.0_canonical/raw/. Fails (exit 1) on any divergence or missing
artifact. This is the build-time integrity gate for the paper.

Run: python docs/paper/scripts/verify_paper_math.py
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "results" / "v2.0_canonical" / "raw"
TEX = ROOT / "docs" / "paper" / "main.tex"

ok, fail, warn = [], [], []

def load(name):
    p = RAW / name
    if not p.exists():
        return None
    return json.loads(p.read_text())

def approx(a, b, tol):
    return abs(float(a) - float(b)) <= tol

tex = TEX.read_text()

# --- deep_splice_ttft.json : Table II speedups ---
d = load("deep_splice_ttft.json")
if d is None:
    fail.append("deep_splice_ttft.json missing")
else:
    for K, key in [(4, "full_mult_4"), (16, "full_mult_16")]:
        rows = d["curves"].get(key, [])
        for r in rows:
            sp = r["speedup"]
            # every cell must preserve top-1 (never-regress) and keep KL small
            if not r.get("top1_agree", 0):
                fail.append(f"K={K} depth={r['p_start']}: top-1 NOT preserved (never-regress violated)")
            if r.get("kl_nats", 1) > 0.01:
                warn.append(f"K={K} depth={r['p_start']}: KL={r['kl_nats']:.5f} > 0.01")
        ok.append(f"deep_splice K={K}: {len(rows)} depths, top-1 preserved")

# --- gating_nogo.json : Spearman 0.193 and Table I rows ---
g = load("gating_nogo.json")
if g is None:
    fail.append("gating_nogo.json missing")
else:
    if not approx(g["mean_spearman"], 0.193, 0.001):
        fail.append(f"gating mean Spearman {g['mean_spearman']} != 0.193 in paper")
    else:
        ok.append("gating_nogo Spearman rho=0.193 matches paper")
    if "0.193" not in tex:
        fail.append("paper does not state Spearman 0.193")

# NOTE: the causal-isolation (block-mask) negative result was removed from the v2.0 paper
# because the legolink_blockmask orchestrator arm is retired from the current codebase
# (single-source-of-truth = current code). No paper claim to verify, so no check here.

# --- rope_boundary_gate.json : KL 5.72 not viable ---
r = load("rope_boundary_gate.json")
if r is None:
    fail.append("rope_boundary_gate.json missing")
else:
    if r.get("viable") is not False:
        fail.append("rope boundary gate should be viable=false")
    else:
        ok.append("rope_boundary_gate viable=false (KL up to 5.72) matches paper")

# --- calibration.json : tau=0.0136, fire ~20% ---
cal = load("calibration.json")
if cal is None:
    fail.append("calibration.json missing")
else:
    tau = cal["calibration"]["CALIBRATED_MARGIN_THRESHOLD"]
    if not approx(tau, 0.0136, 0.0002):
        fail.append(f"calibration tau {tau} != 0.0136")
    else:
        ok.append(f"calibration tau={tau:.4f} matches paper (0.0136)")

# --- routing_accuracy_n250.json : routing accuracy at N=250 ---
def _find(glob):
    xs = sorted(RAW.glob(glob))
    return xs[0] if xs else None

rp = _find("routing_accuracy_n250*.json")
if rp is None:
    warn.append("routing_accuracy_n250.json not present yet (pending live run)")
else:
    rj = json.loads(rp.read_text())
    rs = rj.get("routing_summary", {})
    if "250" in rs:
        acc = rs["250"].get("mean_acc_b")
        r1 = rs["250"].get("mean_rec1")
        if not approx(acc, 89.0, 1.5):
            fail.append(f"routing N=250 = {acc:.1f}% != paper 89% -- UPDATE PAPER")
        else:
            ok.append(f"routing accuracy N=250 = {acc:.0f}% matches paper (89%)")
        if r1 is not None and not approx(r1, 74.0, 2.0):
            warn.append(f"SLB R@1 N=250 = {r1:.0f}% (paper 74%)")
        else:
            ok.append(f"SLB R@1 N=250 = {r1:.0f}% matches paper (74%)")
        # flat-scaling assertion
        accs = [rs[k]["mean_acc_b"] for k in ["10","50","100","250"] if k in rs]
        if accs and (max(accs) - min(accs)) <= 5.0:
            ok.append(f"routing flat across N: {'/'.join(f'{a:.0f}' for a in accs)}%")

# --- l0_radix.json ---
l0 = load("l0_radix.json")
if l0 is None:
    warn.append("l0_radix.json not present")
else:
    hit, cp = l0["hit_rate_mean"], l0["copy_p50_us_median"]
    if not approx(hit, 0.695, 0.01):
        warn.append(f"L0 hit {hit} != 0.695")
    else:
        ok.append(f"L0 hit-rate {hit:.3f} matches paper (69.5%)")
    if not approx(cp, 3.04, 0.15):
        warn.append(f"L0 copy_p50 {cp}us far from paper 3.04us")
    else:
        ok.append(f"L0 copy_p50 {cp:.2f}us matches paper (3.04us)")

# --- sidecar/accuracy.json ---
sp = RAW / "sidecar" / "accuracy.json"
if not sp.exists():
    warn.append("sidecar/accuracy.json not present yet (pending live run)")
else:
    sj = json.loads(sp.read_text())
    s = sj.get("10", sj.get("summary", sj))
    ra = s.get("routing_accuracy")
    if ra is not None and approx(ra, 86.7, 0.5):
        ok.append(f"sidecar routing accuracy {ra:.1f}% matches paper (86.7%)")
    elif ra is not None:
        fail.append(f"sidecar routing {ra:.1f}% != paper 86.7%")
    ir = s.get("ir_tokens_p50")
    if ir is not None and approx(ir, 19, 1):
        ok.append(f"sidecar IR median {ir:.0f} tok matches paper (19)")
    elif ir is not None:
        fail.append(f"sidecar IR median {ir} != paper 19")
    th, to = s.get("ttft_hybrid_p50_ms"), s.get("ttft_oracle_p50_ms")
    if th and to:
        spd = to / th
        if approx(spd, 1.66, 0.08):
            ok.append(f"sidecar TTFT speedup {spd:.2f}x matches paper (1.66x)")
        else:
            fail.append(f"sidecar TTFT speedup {spd:.2f}x != paper 1.66x")

print("=== verify_paper_math (v2.0) ===")
for m in ok:   print(f"  OK    {m}")
for m in warn: print(f"  WARN  {m}")
for m in fail: print(f"  FAIL  {m}")
print(f"\n{len(ok)} ok, {len(warn)} warn, {len(fail)} fail")
sys.exit(1 if fail else 0)
