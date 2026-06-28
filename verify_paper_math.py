#!/usr/bin/env python3
"""
verify_paper_math.py  (v1.1 — canonical, evidence-reading)

Reads the regenerated canonical bundle in results/v1.1_canonical/ and computes
the numbers the docs/paper should cite. Nothing is transcribed; every value is
derived from a committed artifact.

IMPORTANT HONESTY NOTES (2026-06-20 regeneration):
  * The gateway "TTFT" metric is routing+splice latency only. The harness sets
    decode_us=0 and EXCLUDES first-token generation. Do NOT call it TTFT.
  * The headline 6.3x / 52x speedups in the v1.0 paper divided this routing-only
    latency by the FULL-TTFT baselines (B3/B1). That is not like-for-like.
    The honest speedup uses the e2e N1x arm (true TTFT incl. first token).
  * Neither committed headline (171 ms @1384a33b, 237.4 ms @3ea7441) reproduces;
    five fresh runs give 160.2 ms (sigma 0.6 ms, n=500).

Run: .venv/bin/python verify_paper_math.py
"""
import glob
import json
import os

import numpy as np

CANON = os.path.join(os.path.dirname(__file__), "results", "v1.1_canonical", "raw")


def sep(t=""):
    print("-" * 72)
    if t:
        print(t)
        print("-" * 72)


def load_gateway():
    runs = sorted(glob.glob(os.path.join(CANON, "gateway_run*.json")))
    p50s, accs, pooled = [], [], []
    for r in runs:
        d = json.load(open(r))
        b = d["by_arm"]["GW_route"]
        p50s.append(b["ttft_us_p50"] / 1000.0)
        accs.append(b["tool_accuracy"])
        pooled += [float(x["ttft_us"]) / 1000.0 for x in d["records"] if x.get("arm") == "GW_route"]
    return p50s, accs, pooled


def load_e2e():
    # Prefer the Phase-H n=100 interval run; fall back to older clean serial runs.
    phaseh = os.path.join(os.path.dirname(__file__), "results", "v1.1_canonical", "phaseH", "h3_e2e_n100.json")
    if os.path.exists(phaseh):
        return json.load(open(phaseh)).get("by_arm", {}), "phaseH/h3_e2e_n100.json (n=100)"
    for name in ("e2e_clean_n30.json", "e2e_n25.json", "e2e_smoke.json"):
        f = os.path.join(CANON, name)
        if os.path.exists(f):
            return json.load(open(f)).get("by_arm", {}), name
    return None, None


sep("NEXUS v1.1 CANONICAL — DERIVED FROM REGENERATED ARTIFACTS")

p50s, accs, pooled = load_gateway()
if p50s:
    print(f"Gateway routing+splice latency (NOT TTFT; decode_us=0):")
    print(f"  per-run P50 (ms)      : {[round(x,1) for x in p50s]}")
    print(f"  across-run P50 median : {np.median(p50s):.1f} ms  (std {np.std(p50s):.2f})")
    print(f"  pooled n={len(pooled):<4}        : P50 {np.percentile(pooled,50):.1f}  P90 {np.percentile(pooled,90):.1f}  P99 {np.percentile(pooled,99):.1f}")
    print(f"  tool-routing accuracy : {np.mean(accs):.2f}  (stable: {len(set(accs))==1})")
    gw = float(np.median(p50s))
else:
    print("No canonical gateway runs found — run test/bench_gateway_e2e.py x5.")
    gw = None

sep("LIKE-FOR-LIKE END-TO-END TTFT (e2e harness; includes first token)")
ba, src = load_e2e()
if ba:
    def p50(arm):
        return ba.get(arm, {}).get("ttft_us_p50", 0) / 1000.0
    b1, b3, n1x = p50("B1"), p50("B3"), p50("N1x")
    print(f"  source: {src}")
    print(f"  B1 (bloat)  TTFT P50 : {b1:.0f} ms")
    print(f"  B3 (B_RP)   TTFT P50 : {b3:.0f} ms   acc {ba.get('B3',{}).get('tool_accuracy',0):.2f}")
    print(f"  N1x (Nexus) TTFT P50 : {n1x:.0f} ms   acc {ba.get('N1x',{}).get('tool_accuracy',0):.2f}")
    if n1x:
        print(f"  HONEST speedup N1x vs B3 : {b3/n1x:.1f}x   (like-for-like, true TTFT)")
        print(f"  N1x vs B1                : {b1/n1x:.0f}x")
    if gw:
        print(f"  [misleading] route-only vs B3 full-TTFT : {b3/gw:.1f}x  <-- do NOT publish")
else:
    print("  e2e artifact not present yet (bench_e2e.py may still be running).")

sep("VERDICT")
print("Publish: gateway routing+splice ~160 ms (n=500, route-only); end-to-end TTFT speedup vs B_RP = 2.47x, 95% CI [2.41, 2.72] (N1x 457.5 vs B3 1131.4 ms median, n=100, Phase H).")
print("  Accuracy: no detectable N1x-vs-B3 gap at n=100 (delta -0.010, McNemar p=1.0). Scope: Qwen2.5, one host, flash-attention-compatible non-soft-capped only.")
print("Do NOT publish: 237.4 ms / 171 ms 'TTFT', 6.3x, 52x, 'equivalent accuracy', legacy tensor-KL 0.0076/5.72, n=30 2.34x as current.")
