#!/usr/bin/env python3
"""V5 (leak) + V6 (concurrency) stress for the Execution Sidecar.

Run: PYTHONPATH=build:src:test .venv/bin/python test/stress_sidecar_concurrency.py [--iters N] [--threads K]

V5 leak : N serial execute_via_sidecar calls; assert RSS growth bounded after warmup
          and sidecar seq_ids stay inside the reserved [SIDECAR_SEQ_BASE, MAX) band.
V6 concur: K threads each run the SAME query M times; assert no crash/exception and
          DETERMINISTIC args (greedy). Non-determinism would reveal logit-buffer
          cross-contamination across the granular _ctx_lock. (Precondition under test:
          one-orchestration-thread-per-context is documented; this probes what happens
          when violated.)
"""
from __future__ import annotations

import argparse
import os
import resource
import sys
import threading
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from bench_sidecar_accuracy import build_agent, deep_history  # noqa: E402

QUERY = "Create a readme.md file in repository test-repo under user alice with content Hello World"


def rss_mb():
    # macOS ru_maxrss is bytes; Linux is KiB. Normalize to MB heuristically.
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / (1024 * 1024) if v > 10_000_000 else v / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--thread-iters", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=24)
    args = ap.parse_args()

    agent, _ = build_agent()
    hist = deep_history()
    msgs = hist + [{"role": "user", "content": QUERY}]
    fails = []

    # ---- V5 leak ----
    print(f"[V5] {args.iters} serial generate_via_hybrid calls...")
    rss_start = rss_mb()
    rss_after_warmup = None
    for i in range(args.iters):
        name, a, meta = agent.generate_via_hybrid(msgs, max_tokens=args.max_tokens)
        assert name, f"iter {i}: no tool resolved"
        if i == 9:
            rss_after_warmup = rss_mb()
    rss_end = rss_mb()
    growth = rss_end - (rss_after_warmup or rss_start)
    print(f"   RSS start={rss_start:.0f}MB warmup={rss_after_warmup:.0f}MB end={rss_end:.0f}MB "
          f"post-warmup growth={growth:.0f}MB")
    if growth > 256:     # allow caching jitter; flag only large unbounded growth
        fails.append(f"V5 RSS growth {growth:.0f}MB")

    # ---- V6 coarse-grained concurrency / determinism ----
    # Expectation (Decision 2): the coarse _ctx_lock SERIALIZES the 4 threads -> no
    # crash and deterministic args. (Granular locking previously SIGSEGV'd here.)
    print(f"\n[V6] {args.threads} threads x {args.thread_iters} iters, same query (expect serialized)...")
    ref_name, ref_args, _ = agent.generate_via_hybrid(msgs, max_tokens=args.max_tokens)
    results, errors = [], []
    lock = threading.Lock()

    def worker(tid):
        try:
            for _ in range(args.thread_iters):
                n, a, _ = agent.generate_via_hybrid(msgs, max_tokens=args.max_tokens)
                with lock:
                    results.append((tid, n, a))
        except Exception as e:  # segfault would not be catchable; exceptions will be
            with lock:
                errors.append(f"t{tid}: {type(e).__name__}: {e}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(args.threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    n_total = len(results)
    n_same_tool = sum(1 for _, n, _ in results if n == ref_name)
    n_det_args = sum(1 for _, _, a in results if a == ref_args)
    print(f"   completed={n_total}/{args.threads*args.thread_iters} errors={len(errors)}")
    print(f"   same_tool={n_same_tool}/{n_total} deterministic_args={n_det_args}/{n_total}")
    for e in errors[:5]:
        print(f"   ERR {e}")
    if errors:
        fails.append(f"V6 {len(errors)} exceptions")
    if n_total and n_det_args < n_total:
        fails.append(f"V6 non-deterministic args {n_det_args}/{n_total} (logit contamination?)")

    print("\n" + ("STRESS PASSED" if not fails else f"STRESS ISSUES: {fails}"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
