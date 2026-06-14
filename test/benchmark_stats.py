"""Statistical helpers for Nexus benchmark artifacts."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def mcnemar_exact(b: int, c: int) -> dict[str, float]:
    """McNemar test for paired binary outcomes.

    b = count where arm A correct, arm B wrong
    c = count where arm A wrong, arm B correct
    """
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0, "significant_005": False}
    # Exact binomial two-sided p-value
    k = min(b, c)
    p = 0.0
    for i in range(k + 1):
        p += math.comb(n, i) * (0.5 ** n)
    p_value = min(1.0, 2.0 * p)
    return {
        "b": b,
        "c": c,
        "n_discordant": n,
        "p_value": p_value,
        "significant_005": p_value < 0.05,
    }


def paired_mcnemar(
    hits_a: Iterable[bool],
    hits_b: Iterable[bool],
) -> dict[str, float]:
    a = list(hits_a)
    b = list(hits_b)
    if len(a) != len(b):
        raise ValueError("paired arrays must have equal length")
    b_cnt = sum(1 for x, y in zip(a, b) if x and not y)
    c_cnt = sum(1 for x, y in zip(a, b) if not x and y)
    out = mcnemar_exact(b_cnt, c_cnt)
    out["accuracy_a"] = float(np.mean(a)) if a else 0.0
    out["accuracy_b"] = float(np.mean(b)) if b else 0.0
    out["accuracy_delta_a_minus_b"] = out["accuracy_a"] - out["accuracy_b"]
    return out


def bootstrap_ci(
    samples: list[float],
    stat_fn=None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 1337,
) -> dict[str, float]:
    if not samples:
        return {"low": 0.0, "high": 0.0, "point": 0.0, "n": 0}
    stat_fn = stat_fn or (lambda xs: float(np.percentile(xs, 50)))
    rng = np.random.default_rng(seed)
    arr = np.array(samples, dtype=np.float64)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(arr), len(arr))
        boots.append(stat_fn(arr[idx]))
    low, high = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "low": float(low),
        "high": float(high),
        "point": float(stat_fn(arr)),
        "n": len(samples),
    }


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> dict[str, float]:
    """Wilson score interval for binomial proportion (95% default)."""
    if n <= 0:
        return {"low": 0.0, "high": 0.0, "point": 0.0, "n": 0}
    z = 1.959963984540054 if alpha == 0.05 else 1.96
    p_hat = successes / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n) / denom
    return {
        "low": max(0.0, center - margin),
        "high": min(1.0, center + margin),
        "point": p_hat,
        "n": n,
    }


def summarize_arm_stats(records: list[dict], arm: str) -> dict:
    rows = [r for r in records if r.get("arm") == arm]
    hits = [bool(r.get("tool_hit")) for r in rows]
    ttft = [float(r.get("ttft_us", 0)) for r in rows]
    decode = [float(r.get("decode_us", 0)) for r in rows]
    rets = [bool(r.get("retrieval_hit")) for r in rows]
    decode_ok = [bool(r["tool_hit"]) for r in rows if r.get("retrieval_hit")]
    n_hits = sum(1 for h in hits if h)
    return {
        "n": len(rows),
        "tool_accuracy": float(np.mean(hits)) if hits else 0.0,
        "tool_accuracy_ci": bootstrap_ci([1.0 if h else 0.0 for h in hits], stat_fn=np.mean),
        "tool_accuracy_wilson": wilson_ci(n_hits, len(hits)),
        "retrieval_hit_rate": float(np.mean(rets)) if rets else 0.0,
        "decode_accuracy_given_retrieval_hit": float(np.mean(decode_ok)) if decode_ok else None,
        "ttft_us_mean": float(np.mean(ttft)) if ttft else 0.0,
        "ttft_us_p50": float(np.percentile(ttft, 50)) if ttft else 0.0,
        "ttft_us_p50_ci": bootstrap_ci(ttft, stat_fn=lambda xs: float(np.percentile(xs, 50))),
        "decode_us_mean": float(np.mean(decode)) if decode else 0.0,
    }
