"""Shared selective recompute helpers (causal suffix repair after splice)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import nexus_fsm_ext

SUFFIX_SELECTORS = (
    "tail",
    "hkvd_suffix_start",
    "oracle_suffix_start",
    "boundary_window",
    "fused_hkvd",
)
DEFAULT_SUFFIX_PCTS = (2, 5, 10, 15, 20, 30, 50, 100)


@dataclass(frozen=True)
class RecomputePlan:
    """Causal suffix recompute plan after splice."""

    suffix_start_idx: int
    nominal_suffix_pct: float
    selector: str
    selected_token_count: int
    actual_recompute_tokens: int
    actual_recompute_pct: float

    @property
    def selected(self) -> list[int]:
        if self.actual_recompute_tokens <= 0:
            return []
        start = self.suffix_start_idx
        end = start + self.actual_recompute_tokens
        return list(range(start, end))


def n_select(schema_len: int, suffix_pct: float) -> int:
    if suffix_pct <= 0 or schema_len <= 0:
        return 0
    if suffix_pct >= 100:
        return schema_len
    return max(1, int(math.ceil(schema_len * suffix_pct / 100.0)))


def actual_recompute_tokens(schema_len: int, suffix_start_idx: int) -> int:
    if schema_len <= 0 or suffix_start_idx >= schema_len:
        return 0
    return schema_len - suffix_start_idx


def plan_suffix_recompute(
    selector: str,
    schema_len: int,
    suffix_pct: float,
    layer1_dev=None,
    oracle_dev=None,
) -> RecomputePlan:
    """Build an honest suffix recompute plan.

    Under causal constraints, recompute always decodes schema_tokens[start:].
    Selectors differ only in suffix_start_idx.
    """
    if suffix_pct <= 0 or schema_len <= 0:
        return RecomputePlan(0, suffix_pct, selector, 0, 0, 0.0)

    n = n_select(schema_len, suffix_pct)
    if selector == "tail":
        start = max(0, schema_len - n)
    elif selector == "hkvd_suffix_start":
        if layer1_dev is None or len(layer1_dev) != schema_len:
            raise ValueError("hkvd_suffix_start requires layer1_dev")
        order = np_argsort_desc(layer1_dev)
        top = sorted(order[:n])
        start = top[0] if top else max(0, schema_len - n)
    elif selector == "oracle_suffix_start":
        if oracle_dev is None or len(oracle_dev) != schema_len:
            raise ValueError("oracle_suffix_start requires oracle_dev")
        order = np_argsort_desc(oracle_dev)
        top = sorted(order[:n])
        start = top[0] if top else max(0, schema_len - n)
    elif selector == "boundary_window":
        # Cross-boundary repair: recompute prefix window + tail seam (CacheBlend-style).
        head_n = max(1, n // 2)
        tail_n = max(1, n - head_n)
        start = 0
        n = min(schema_len, head_n + tail_n)
    elif selector == "fused_hkvd":
        if layer1_dev is None or len(layer1_dev) != schema_len:
            raise ValueError("fused_hkvd requires layer1_dev")
        order = np_argsort_desc(layer1_dev)
        top = sorted(order[:n])
        start = top[0] if top else max(0, schema_len - n)
    else:
        raise ValueError(f"unknown suffix selector: {selector}")

    actual = actual_recompute_tokens(schema_len, start)
    actual_pct = 100.0 * actual / schema_len if schema_len else 0.0
    return RecomputePlan(
        suffix_start_idx=start,
        nominal_suffix_pct=suffix_pct,
        selector=selector,
        selected_token_count=n,
        actual_recompute_tokens=actual,
        actual_recompute_pct=actual_pct,
    )


def np_argsort_desc(arr) -> list[int]:
    import numpy as np

    return np.argsort(-np.asarray(arr, dtype=np.float64)).tolist()


def contiguous_runs(indices: Iterable[int]) -> list[tuple[int, int]]:
    idxs = sorted(set(indices))
    if not idxs:
        return []
    runs: list[tuple[int, int]] = []
    start = idxs[0]
    prev = start
    for idx in idxs[1:]:
        if idx != prev + 1:
            runs.append((start, prev))
            start = idx
        prev = idx
    runs.append((start, prev))
    return runs


def schema_invalidate_end(p_start: int, schema_len: int) -> int:
    """Exclusive end position for llama_kv_cache_seq_rm [p0, p1)."""
    return p_start + schema_len


def recompute_selected(
    ctx,
    schema_tokens: list[int],
    p_start: int,
    selected: list[int],
) -> int:
    """Recompute suffix from min(selected). Returns actual_recompute_tokens."""
    if not selected:
        return 0
    min_idx = min(selected)
    abs_start = p_start + min_idx
    # llama_kv_cache_seq_rm is half-open [p0, p1); use exclusive schema end.
    nexus_fsm_ext.invalidate_sequence(ctx, 0, abs_start, schema_invalidate_end(p_start, len(schema_tokens)))
    nexus_fsm_ext.decode_tokens(ctx, schema_tokens[min_idx:], abs_start, 0)
    return len(schema_tokens) - min_idx


def batch_recompute_tail(
    ctx,
    schema_tokens: list[int],
    p_start: int,
    recompute_pct: float,
    *,
    n_ubatch: int = 2048,
) -> int:
    """Suffix recompute using full ubatch where possible."""
    plan = plan_suffix_recompute("tail", len(schema_tokens), recompute_pct)
    if plan.actual_recompute_tokens <= 0:
        return 0
    abs_start = p_start + plan.suffix_start_idx
    nexus_fsm_ext.invalidate_sequence(
        ctx, 0, abs_start, schema_invalidate_end(p_start, len(schema_tokens))
    )
    toks = schema_tokens[plan.suffix_start_idx:]
    # decode_tokens already batches internally; n_ubatch reserved for future native hook.
    _ = n_ubatch
    nexus_fsm_ext.decode_tokens(ctx, toks, abs_start, 0)
    return plan.actual_recompute_tokens


def recompute_plan(ctx, schema_tokens: list[int], p_start: int, plan: RecomputePlan) -> int:
    return recompute_selected(ctx, schema_tokens, p_start, plan.selected)


def recompute_tail(
    ctx,
    schema_tokens: list[int],
    p_start: int,
    recompute_pct: float,
) -> int:
    return batch_recompute_tail(ctx, schema_tokens, p_start, recompute_pct)


def recompute_fused_with_query(
    ctx,
    schema_tokens: list[int],
    p_start: int,
    query_tokens: list[int],
    recompute_pct: float,
    *,
    seq_id: int = 0,
):
    """Fuse suffix recompute + query prefill into one llama_batch (single sync).

    Causal masking ensures logits match the two-step invalidate→suffix→query path.
    """
    import numpy as np

    plan = plan_suffix_recompute("tail", len(schema_tokens), recompute_pct)
    if plan.actual_recompute_tokens <= 0:
        return np.array(
            nexus_fsm_ext.decode_tokens(ctx, query_tokens, p_start + len(schema_tokens), seq_id),
            dtype=np.float32,
        )
    return np.array(
        nexus_fsm_ext.recompute_fused_with_query(
            ctx,
            schema_tokens,
            p_start,
            plan.suffix_start_idx,
            query_tokens,
            seq_id,
        ),
        dtype=np.float32,
    )
