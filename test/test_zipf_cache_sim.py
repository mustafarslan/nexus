#!/usr/bin/env python3
"""Unit test Zipf cache simulation (LFU + pinning) without LLM."""
from __future__ import annotations

import random


def zipfian_sample(rng: random.Random, n: int, s: float = 1.0) -> int:
    weights = [1.0 / ((i + 1) ** s) for i in range(n)]
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if r <= acc:
            return i
    return n - 1


def simulate_zipf(
    n_tools: int = 52,
    n_queries: int = 500,
    cache_blocks: int = 32,
    pin_top_n: int = 5,
    lfu: bool = True,
    seed: int = 1337,
    warmup: int | None = None,
) -> float:
    rng = random.Random(seed)
    tool_names = [f"tool_{i}" for i in range(n_tools)]
    pinned = set(tool_names[:pin_top_n])
    lru: list[str] = []
    freq: dict[str, int] = {n: 0 for n in tool_names}
    full_residency = cache_blocks >= n_tools
    cap = (n_tools - len(pinned)) if full_residency else max(cache_blocks - len(pinned), 0)

    def touch(name: str) -> None:
        nonlocal lru
        if name in pinned:
            return
        if name in lru:
            lru.remove(name)
        lru.append(name)
        if full_residency:
            return
        if lfu:
            while len(lru) > cap:
                evict = min(lru, key=lambda n: freq.get(n, 0), default=None)
                if evict is None:
                    break
                lru.remove(evict)
        else:
            while len(lru) > cap:
                lru.pop(0)

    warm = warmup if warmup is not None else n_tools * 2
    for name in tool_names:
        freq[name] += 1
        touch(name)
    for _ in range(warm):
        idx = zipfian_sample(rng, n_tools, s=1.0)
        name = tool_names[idx]
        freq[name] += 1
        touch(name)

    cold = 0
    for _ in range(n_queries):
        idx = zipfian_sample(rng, n_tools, s=1.0)
        name = tool_names[idx]
        freq[name] += 1
        resident = pinned | set(lru)
        if name not in resident:
            cold += 1
        touch(name)
    return cold / max(n_queries, 1)


def test_pinning_beats_baseline_at_32():
    with_pin = simulate_zipf(cache_blocks=32, pin_top_n=5, lfu=False)
    no_pin = simulate_zipf(cache_blocks=32, pin_top_n=0, lfu=False)
    assert with_pin <= no_pin, f"pinning should reduce cold-load: {with_pin:.3f} vs {no_pin:.3f}"


def test_64_blocks_steady_cold_under_2pct():
    frac = simulate_zipf(cache_blocks=64, pin_top_n=5, lfu=True)
    assert frac < 0.02, f"expected steady cold-load <2% at 64 blocks, got {frac:.3f}"


if __name__ == "__main__":
    for blocks in (32, 64):
        f = simulate_zipf(cache_blocks=blocks, pin_top_n=5, lfu=True)
        print(f"cache={blocks} cold_load={f:.3f}")
