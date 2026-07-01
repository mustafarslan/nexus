#!/usr/bin/env python3
"""Re-derive every quantitative claim in docs/paper/main.tex from the canonical
v2.0 artifacts and emit a verification table. Also computes Wilson 95% CIs for all
headline proportions and re-confirms Eq.(2) reproduces the Table II R(%) column.

Read-only over results/v2.0_canonical/. No fabricated numbers: every row traces to a file.
"""
import json
import math
import os

ROOT = os.path.join(os.path.dirname(__file__), "..", "results", "v2.0_canonical", "raw")


def load(name):
    with open(os.path.join(ROOT, name)) as f:
        return json.load(f)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def R_eq2(npast, K, M=256, Rbase=5.0):
    if npast <= M:
        return Rbase
    return min(100.0, Rbase + (npast - M) / (M * (K - 1)) * (100 - Rbase))


rows = []  # (claim, paper, recomputed, verdict)


def check(claim, paper, recomputed, ok=None):
    if ok is None:
        try:
            ok = abs(float(paper) - float(recomputed)) < 0.15
        except (TypeError, ValueError):
            ok = str(paper) == str(recomputed)
    rows.append((claim, str(paper), str(recomputed), "MATCH" if ok else "DISCREPANCY"))


# ---- Routing (Table III + inline §VII-B) ----
r = load("routing_accuracy_n250.json")["routing_summary"]
for N, pv in [("10", 92), ("50", 90), ("100", 89), ("250", 89)]:
    check(f"routing acc_b N={N}", pv, r[N]["mean_acc_b"])
check("oracle acc N=10", 98, r["10"]["mean_acc_a"])
for N in ("50", "100", "250"):
    check(f"oracle acc N={N} (overflow=0)", 0.0, r["250" if N == "250" else N]["mean_acc_a"])
check("SLB R@1 N=250", 74, r["250"]["mean_rec1"])
check("SLB R@3 N=250", 95, r["250"]["mean_rec3"])
check("SLB in-situ us N=250", 17.6, r["250"]["avg_lat_slb_ms"] * 1000, ok=abs(r["250"]["avg_lat_slb_ms"] * 1000 - 17.6) < 0.1)

# ---- SLB pure SIMD ----
check("SLB pure SIMD us", 8.25, load("slb_latency.json")["median_us"])

# ---- Calibration ----
c = load("calibration.json")
check("tau (P20 adversarial)", 0.0136, c["calibration"]["CALIBRATED_MARGIN_THRESHOLD"], ok=abs(c["calibration"]["CALIBRATED_MARGIN_THRESHOLD"] - 0.0136) < 0.001)
check("gate fire rate %", 20.3, c["p20"]["fire_rate"] * 100, ok=abs(c["p20"]["fire_rate"] * 100 - 20.3) < 0.1)

# ---- Gating ----
g = load("gating_nogo.json")
check("gating mean Spearman rho", 0.193, g["mean_spearman"])
drifts = [row["max_drift"] for row in g["per_condition"]]
check("gating max drift range (spread exists)", "175-207", f"{min(drifts):.0f}-{max(drifts):.0f}")

# ---- L0 radix ----
l = load("l0_radix.json")
check("L0 radix copy us P50", 3.04, l["copy_p50_us_median"])
check("L0 warm-hit %", 69.5, l["hit_rate_mean"] * 100)

# ---- Sidecar (real denominators, n=30 consistency set) ----
s = load("sidecar/accuracy.json")["10"]
check("sidecar routing %", 86.7, s["routing_accuracy"], ok=abs(s["routing_accuracy"] - 86.667) < 0.1)
check("sidecar arg routed-only %", 100, s["arg_accuracy_routed_only"])
check("sidecar arg e2e %", 80, s["arg_accuracy_e2e"])
check("sidecar JSON valid %", 100, s["json_valid_rate_hybrid"])
check("IR tokens p50/p99", "19/32", f"{int(s['ir_tokens_p50'])}/{int(s['ir_tokens_p99'])}")
check("ttft hybrid ms", 443.8, s["ttft_hybrid_p50_ms"], ok=abs(s["ttft_hybrid_p50_ms"] - 443.8) < 0.1)
check("ttft oracle ms", 737.3, s["ttft_oracle_p50_ms"], ok=abs(s["ttft_oracle_p50_ms"] - 737.3) < 0.1)
check("hybrid speedup x", 1.66, s["ttft_oracle_p50_ms"] / s["ttft_hybrid_p50_ms"], ok=abs(s["ttft_oracle_p50_ms"] / s["ttft_hybrid_p50_ms"] - 1.66) < 0.02)
check("sidecar token saving %", 80, s["token_savings_pct"], ok=abs(s["token_savings_pct"] - 80.0) < 1.0)

# ---- Deep splice TTFT (Table II) ----
d = load("deep_splice_ttft.json")["curves"]
paper_ttft = {
    "full_mult_4": {256: (5.0, 3278, 2010, 1.63), 512: (36.7, 4672, 3761, 1.24),
                    1024: (100.0, 7302, 7425, 0.98), 2048: (100.0, 13092, 13142, 1.00)},
    "full_mult_16": {256: (5.0, 3327, 1924, 1.73), 512: (11.3, 4747, 3336, 1.42),
                     1024: (24.0, 7625, 6532, 1.17), 2048: (49.3, 13468, 12620, 1.07)},
}
for curve, cells in paper_ttft.items():
    for row in d[curve]:
        p = row["p_start"]
        pe, ppf, psp, psu = cells[p]
        check(f"{curve} n={p} R%", pe, row["eff_pct"])
        check(f"{curve} n={p} prefill ms", ppf, round(row["prefill_ms_median"]), ok=abs(round(row["prefill_ms_median"]) - ppf) <= 1)
        check(f"{curve} n={p} splice ms", psp, round(row["splice_ms_median"]), ok=abs(round(row["splice_ms_median"]) - psp) <= 1)
        check(f"{curve} n={p} speedup", psu, row["speedup"], ok=abs(row["speedup"] - psu) < 0.01)

# ---- Eq.(2) reproduces R(%) ----
eq2_ok = True
for n, K, rep in [(256, 4, 5.0), (512, 4, 36.7), (1024, 4, 100.0), (2048, 4, 100.0),
                  (256, 16, 5.0), (512, 16, 11.3), (1024, 16, 24.0), (2048, 16, 49.3)]:
    if abs(R_eq2(n, K) - rep) >= 0.15:
        eq2_ok = False
check("Eq.(2) reproduces all Table II R(%)", "yes", "yes" if eq2_ok else "NO", ok=eq2_ok)

# ---- Wilson CIs for headline proportions (real k/n) ----
props = [
    ("routing acc N=250", 89, 100),
    ("SLB R@1 N=250", 74, 100),
    ("SLB R@3 N=250", 95, 100),
    ("sidecar routing", 26, 30),
    ("sidecar arg e2e", 24, 30),
    ("sidecar arg routed-only", 26, 26),
]

if __name__ == "__main__":
    print("\n=== VERIFICATION TABLE ===")
    w = max(len(x[0]) for x in rows)
    ndis = 0
    for claim, pv, rv, verdict in rows:
        if verdict == "DISCREPANCY":
            ndis += 1
        print(f"{claim:<{w}}  paper={pv:<10} recomputed={rv:<12} {verdict}")
    print(f"\n{len(rows)} claims checked, {ndis} discrepancies.")

    print("\n=== WILSON 95% CIs (real k/n) ===")
    for name, k, n in props:
        lo, hi = wilson(k, n)
        print(f"{name:<26} {k}/{n} = {100*k/n:5.1f}%   95% CI [{100*lo:.1f}, {100*hi:.1f}]")
