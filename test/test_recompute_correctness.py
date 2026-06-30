#!/usr/bin/env python3
"""G1: Unit test that oracle r=100% recompute matches reference prefill logits."""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../build/external/llama.cpp/src"))
os.environ["LLAMA_CPP_LIB_PATH"] = lib_dir
os.environ["LLAMA_CPP_LIB"] = os.path.join(lib_dir, "libllama.dylib")

import llama_cpp  # noqa: E402
import nexus_fsm_ext  # noqa: E402
from bench_phaseB_selective_recompute import (  # noqa: E402
    DEFAULT_MODEL,
    build_preceding_tokens,
    compute_deviations_with_preceding,
    format_query,
    kl_ref_to_splice,
    reference_logits,
    schema_text_for,
    select_token_indices,
    splice_recompute_query,
    tokenize,
)
from bench_routing_accuracy import load_first_10_tools, queries_dataset  # noqa: E402


def _top1(logits: np.ndarray) -> int:
    return int(np.argmax(logits))


@pytest.fixture(scope="module")
def llm_ctx():
    model = DEFAULT_MODEL
    if not Path(model).exists():
        pytest.skip(f"model not found: {model}")
    llm = llama_cpp.Llama(
        model_path=model, n_ctx=8192, n_batch=2048, n_ubatch=2048,
        n_gpu_layers=999, flash_attn=True, verbose=False,
    )
    ctx = llm._ctx.ctx
    cache = nexus_fsm_ext.NexusBlockCache(2 * 1024 * 1024 * 1024)
    return llm, ctx, cache


def test_oracle_r100_matches_reference(llm_ctx):
    llm, ctx, cache = llm_ctx
    tool = load_first_10_tools()[0]
    atb = Path("results/phaseA_tool_match_work/tool_0.isolated.atb")
    if not atb.exists():
        pytest.skip("missing compiled ATB")
    case = queries_dataset[0]
    schema_tokens = tokenize(llm, schema_text_for(tool))
    query_tokens = tokenize(llm, format_query(case["query"]))
    p_start = 256
    preceding = build_preceding_tokens(llm, "You are a tool-using assistant.", p_start)
    schema_len = len(schema_tokens)
    n_layer = int(llm.metadata.get("llama.block_count", 48))

    ref = reference_logits(ctx, preceding, schema_tokens, query_tokens)
    layer1_dev, oracle_dev = compute_deviations_with_preceding(
        ctx, cache, atb, preceding, schema_tokens, p_start, schema_len, n_layer,
    )
    rng = np.random.default_rng(0)
    selected = select_token_indices("oracle_full_deviation", schema_len, 100.0, layer1_dev, oracle_dev, rng)
    assert len(selected) == schema_len

    splice, _ = splice_recompute_query(
        ctx, cache, atb, preceding, schema_tokens, query_tokens, p_start, selected,
    )
    kl = kl_ref_to_splice(ref, splice)
    assert _top1(ref) == _top1(splice), f"top1 mismatch ref={_top1(ref)} splice={_top1(splice)} kl={kl}"
    assert kl < 0.05, f"oracle r=100% KL too high: {kl}"


def test_schema_invalidate_end_is_exclusive():
    from nexus_recompute import schema_invalidate_end

    assert schema_invalidate_end(256, 174) == 430


def test_r100_recompute_matches_reference_at_p1024(llm_ctx):
    """After exclusive-invalidate fix, full suffix repair should match reference at P=1024."""
    from nexus_recompute import plan_suffix_recompute, recompute_plan, schema_invalidate_end

    llm, ctx, cache = llm_ctx
    tool = load_first_10_tools()[0]
    atb = Path("results/phaseA_tool_match_work/tool_0.isolated.atb")
    if not atb.exists():
        pytest.skip("missing compiled ATB")
    case = queries_dataset[0]
    schema_tokens = tokenize(llm, schema_text_for(tool))
    query_tokens = tokenize(llm, format_query(case["query"]))
    p_start = 1024
    preceding = build_preceding_tokens(llm, "You are a tool-using assistant.", p_start)

    ref = reference_logits(ctx, preceding, schema_tokens, query_tokens)
    nexus_fsm_ext.clear_kv_cache(ctx)
    if preceding:
        nexus_fsm_ext.decode_tokens(ctx, preceding, 0, 0)
    handle = cache.get_or_load(str(atb))
    nexus_fsm_ext.inject_tool_page(ctx, handle, p_start, 0)
    plan = plan_suffix_recompute("tail", len(schema_tokens), 100.0)
    assert schema_invalidate_end(p_start, len(schema_tokens)) == p_start + len(schema_tokens)
    recompute_plan(ctx, schema_tokens, p_start, plan)
    splice_logits = np.array(
        nexus_fsm_ext.decode_tokens(ctx, query_tokens, p_start + len(schema_tokens), 0),
        dtype=np.float32,
    )
    kl = kl_ref_to_splice(ref, splice_logits)
    assert _top1(ref) == _top1(splice_logits), f"P1024 r100 top1 mismatch kl={kl}"
    assert kl < 0.05, f"P1024 r100 KL too high after invalidate fix: {kl}"


def _adaptive_recompute_pct(n_past, max_splice_pos=256, full_mult=4.0, base_pct=5.0):
    """Mirror of NexusOrchestrator's depth-adaptive eff_recompute_pct ramp (Phase 1)."""
    if n_past <= max_splice_pos:
        return base_pct
    lo, hi = float(max_splice_pos), max_splice_pos * full_mult
    frac = min(max((n_past - lo) / (hi - lo), 0.0), 1.0) if hi > lo else 1.0
    return base_pct + frac * (100.0 - base_pct)


def test_adaptive_recompute_never_regress_at_p512(llm_ctx):
    """Phase 1: the depth-adaptive recompute fraction (~37% at P=512) must hold top-1
    against the reference, i.e. the deep splice never regresses below text prefill."""
    from nexus_recompute import recompute_tail

    llm, ctx, cache = llm_ctx
    tool = load_first_10_tools()[0]
    atb = Path("results/phaseA_tool_match_work/tool_0.isolated.atb")
    if not atb.exists():
        pytest.skip("missing compiled ATB")
    case = queries_dataset[0]
    schema_tokens = tokenize(llm, schema_text_for(tool))
    query_tokens = tokenize(llm, format_query(case["query"]))
    p_start = 512
    preceding = build_preceding_tokens(llm, "You are a tool-using assistant.", p_start)

    ref = reference_logits(ctx, preceding, schema_tokens, query_tokens)
    nexus_fsm_ext.clear_kv_cache(ctx)
    if preceding:
        nexus_fsm_ext.decode_tokens(ctx, preceding, 0, 0)
    handle = cache.get_or_load(str(atb))
    nexus_fsm_ext.inject_tool_page(ctx, handle, p_start, 0)
    eff = _adaptive_recompute_pct(p_start)
    assert 30.0 < eff < 45.0, f"unexpected adaptive pct at P=512: {eff}"
    recompute_tail(ctx, schema_tokens, p_start, eff)
    splice = np.array(
        nexus_fsm_ext.decode_tokens(ctx, query_tokens, p_start + len(schema_tokens), 0),
        dtype=np.float32,
    )
    kl = kl_ref_to_splice(ref, splice)
    assert _top1(ref) == _top1(splice), f"P512 adaptive top1 regressed kl={kl}"
    assert kl < 0.05, f"P512 adaptive KL too high: {kl}"


def test_fused_recompute_matches_two_step_at_p256(llm_ctx):
    """Fused suffix+query batch must match invalidate→suffix→query logits."""
    from nexus_recompute import plan_suffix_recompute, recompute_fused_with_query, recompute_tail

    llm, ctx, cache = llm_ctx
    tool = load_first_10_tools()[0]
    atb = Path("results/phaseA_tool_match_work/tool_0.isolated.atb")
    if not atb.exists():
        pytest.skip("missing compiled ATB")
    case = queries_dataset[0]
    schema_tokens = tokenize(llm, schema_text_for(tool))
    query_tokens = tokenize(llm, format_query(case["query"]))
    p_start = 256
    preceding = build_preceding_tokens(llm, "You are a tool-using assistant.", p_start)
    recompute_pct = 5.0

    nexus_fsm_ext.clear_kv_cache(ctx)
    if preceding:
        nexus_fsm_ext.decode_tokens(ctx, preceding, 0, 0)
    handle = cache.get_or_load(str(atb))
    nexus_fsm_ext.inject_tool_page(ctx, handle, p_start, 0)
    recompute_tail(ctx, schema_tokens, p_start, recompute_pct)
    two_step = np.array(
        nexus_fsm_ext.decode_tokens(ctx, query_tokens, p_start + len(schema_tokens), 0),
        dtype=np.float32,
    )

    nexus_fsm_ext.clear_kv_cache(ctx)
    if preceding:
        nexus_fsm_ext.decode_tokens(ctx, preceding, 0, 0)
    handle = cache.get_or_load(str(atb))
    nexus_fsm_ext.inject_tool_page(ctx, handle, p_start, 0)
    fused = recompute_fused_with_query(ctx, schema_tokens, p_start, query_tokens, recompute_pct)

    kl = kl_ref_to_splice(two_step, fused)
    plan = plan_suffix_recompute("tail", len(schema_tokens), recompute_pct)
    assert plan.actual_recompute_tokens > 0
    assert _top1(two_step) == _top1(fused), f"fused top1 mismatch kl={kl}"
    assert kl < 1e-4, f"fused vs two-step KL too high: {kl}"


def test_tail_r5_improves_over_bare_at_p256(llm_ctx):
    llm, ctx, cache = llm_ctx
    tool = load_first_10_tools()[0]
    atb = Path("results/phaseA_tool_match_work/tool_0.isolated.atb")
    if not atb.exists():
        pytest.skip("missing compiled ATB")
    case = queries_dataset[0]
    schema_tokens = tokenize(llm, schema_text_for(tool))
    query_tokens = tokenize(llm, format_query(case["query"]))
    p_start = 256
    preceding = build_preceding_tokens(llm, "You are a tool-using assistant.", p_start)

    ref = reference_logits(ctx, preceding, schema_tokens, query_tokens)
    bare, _ = splice_recompute_query(ctx, cache, atb, preceding, schema_tokens, query_tokens, p_start, [])
    n = max(1, int(len(schema_tokens) * 0.05))
    tail_sel = list(range(len(schema_tokens) - n, len(schema_tokens)))
    tail, _ = splice_recompute_query(
        ctx, cache, atb, preceding, schema_tokens, query_tokens, p_start, tail_sel,
    )
    kl_bare = kl_ref_to_splice(ref, bare)
    kl_tail = kl_ref_to_splice(ref, tail)
    assert kl_tail < kl_bare, f"tail r=5% should improve over bare at P=256: {kl_tail} vs {kl_bare}"
    assert kl_tail < 0.05
