#!/usr/bin/env python3
"""
verify_paper_math.py
Empirical arithmetic verification for docs/paper/nexus_paper.tex.

Anti-hallucination protocol: every number printed by this script is computed,
not transcribed. Run: python3 verify_paper_math.py
"""

# --- Component latencies exactly as written in Section V (milliseconds) ---
delta_prefill_ms = 13.41     # Query Delta-Prefill (250-token query GPU prefill)
splicing_ms      = 1.89      # Physical KV cache splice (P50)
fsm_ms           = 0.00579   # FSM logit masking (5.79 us)
slb_ms           = 0.00225   # SLB similarity scan (2.25 us)

baseline_prefill_ms = 2226.00  # FlashAttention-2 GPU prefill of 12,500-token schema

# Numbers asserted in the manuscript that we are auditing:
claimed_ttft_ms      = 15.31    # stated end-to-end TTFT
claimed_speedup      = 145.4    # stated TTFT speedup multiplier
claimed_splice_only  = 1178     # "1,178x faster" for splice-only vs baseline
claimed_routing_us   = 9        # "< 9 us" combined SLB+FSM
claimed_routing_pct  = 0.06     # "< 0.06%" of total budget
claimed_slb_abstract_us = 1.83  # SLB latency quoted in Abstract / Contribution C1


def sep(t=""):
    print("-" * 68)
    if t:
        print(t)
        print("-" * 68)


sep("STAGE 1: EMPIRICAL ARITHMETIC VERIFICATION")

# 1) TTFT component sum
ttft_sum = delta_prefill_ms + splicing_ms + fsm_ms + slb_ms
print(f"Delta-Prefill          : {delta_prefill_ms:>12.5f} ms")
print(f"Splicing               : {splicing_ms:>12.5f} ms")
print(f"FSM masking            : {fsm_ms:>12.5f} ms")
print(f"SLB scan               : {slb_ms:>12.5f} ms")
print(f"  => EXACT TTFT sum     : {ttft_sum:>12.5f} ms")
print(f"  => rounded (2 dp)     : {round(ttft_sum, 2):>12.2f} ms")
print(f"  manuscript claims     : {claimed_ttft_ms:>12.2f} ms")
print(f"  rounded-sum == claim? : {round(ttft_sum, 2) == claimed_ttft_ms}")

sep()

# 2) End-to-end speedup multiplier
speedup = baseline_prefill_ms / ttft_sum
print(f"Baseline / TTFT sum    : {baseline_prefill_ms} / {ttft_sum:.5f}")
print(f"  => EXACT speedup      : {speedup:.4f}x")
print(f"  => rounded (1 dp)     : {round(speedup, 1)}x")
print(f"  manuscript claims     : {claimed_speedup}x")
print(f"  rounded == claim?     : {round(speedup, 1) == claimed_speedup}")

sep()

# 3) Splice-only speedup ("1,178x faster")
splice_speedup = baseline_prefill_ms / splicing_ms
print(f"Baseline / splice-only : {baseline_prefill_ms} / {splicing_ms}")
print(f"  => EXACT             : {splice_speedup:.2f}x")
print(f"  => rounded (int)     : {round(splice_speedup)}x")
print(f"  manuscript claims    : {claimed_splice_only}x")
print(f"  rounded == claim?    : {round(splice_speedup) == claimed_splice_only}")

sep()

# 4) Routing overhead micro-claims
routing_us = (fsm_ms + slb_ms) * 1000.0
routing_pct = routing_us / (ttft_sum * 1000.0) * 100.0
print(f"SLB+FSM combined       : {routing_us:.3f} us   (< {claimed_routing_us} us? {routing_us < claimed_routing_us})")
print(f"Routing % of TTFT      : {routing_pct:.5f}% (< {claimed_routing_pct}%? {routing_pct < claimed_routing_pct})")

sep()

# 5) Internal SLB latency consistency check
print(f"SLB latency in Abstract/C1 : {claimed_slb_abstract_us} us")
print(f"SLB latency in breakdown   : {slb_ms*1000:.2f} us")
print(f"  consistent?              : {abs(claimed_slb_abstract_us - slb_ms*1000) < 1e-9}")

sep("SUMMARY")
print(f"Use TTFT    = {round(ttft_sum,2)} ms   (exact {ttft_sum:.5f})")
print(f"Use speedup = {round(speedup,1)}x   (exact {speedup:.4f})")
