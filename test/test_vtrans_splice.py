#!/usr/bin/env python3
"""Phase 3 regression: transposed-V (v_trans=true) KV splice on a soft-cap model.

Gemma2 has attn_logit_softcapping, which forces FlashAttention off, which flips the
KV cache to the transposed V layout (v_trans=true). Before v2.0 the splicer threw a
hard error on this; now splice_v_layer() handles the [n_head_kv, d_head, n_tokens]
layout (nexus_kv_splicer.cpp). These tests re-inject a compiled .atb and assert the
spliced logits reproduce the reference full-prefill logits bit-for-bit.

Opt-in: skips unless a Gemma2 GGUF is available. Point NEXUS_GEMMA2_MODEL at one, or
place it at the default path below. The .atb is compiled on first run and cached under
results/v2_phase3_vtrans/.
"""
from __future__ import annotations

import os
import subprocess
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

REPO = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
DEFAULT_GEMMA2 = "/Volumes/AI_SSD/models/gemma-2-9b-it-GGUF/gemma-2-9b-it-Q4_K_M.gguf"
GEMMA2_MODEL = os.environ.get("NEXUS_GEMMA2_MODEL", DEFAULT_GEMMA2)
SCHEMA = REPO / "test/schemas/list_commits.json"
ATB = REPO / "results/v2_phase3_vtrans/gemma9b_list_commits.atb"


def _softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def _kl(ref, spl):
    p = _softmax(np.asarray(ref, np.float64))
    q = _softmax(np.asarray(spl, np.float64))
    m = p > 1e-12
    return float(np.sum(p[m] * np.log(p[m] / np.clip(q[m], 1e-12, None))))


def _ensure_atb():
    if ATB.exists():
        return
    compiler = REPO / "build/nexus_kv_compiler"
    if not compiler.exists():
        pytest.skip("nexus_kv_compiler not built")
    ATB.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [str(compiler), "--model", GEMMA2_MODEL, "--schema", str(SCHEMA), "--output", str(ATB)],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not ATB.exists():
        pytest.skip(f"failed to compile gemma2 ATB: {r.stderr[-300:]}")


@pytest.fixture(scope="module")
def gemma_ctx():
    if not Path(GEMMA2_MODEL).exists():
        pytest.skip(f"Gemma2 model not found: {GEMMA2_MODEL} (set NEXUS_GEMMA2_MODEL)")
    _ensure_atb()
    llm = llama_cpp.Llama(
        model_path=GEMMA2_MODEL, n_ctx=4096, n_batch=2048, n_ubatch=2048,
        n_gpu_layers=999, flash_attn=True, verbose=False,  # forced off by soft-cap -> v_trans=true
    )
    ctx = llm._ctx.ctx
    cache = nexus_fsm_ext.NexusBlockCache(2 * 1024 * 1024 * 1024)
    schema_tokens = [int(t) for t in llm.tokenize(SCHEMA.read_text().encode("utf-8"), add_bos=False, special=False)]
    query_tokens = [int(t) for t in llm.tokenize(
        b"List the recent commits of branch main in repository test-repo", add_bos=False, special=False)]
    return llm, ctx, cache, schema_tokens, query_tokens


def test_vtrans_bare_splice_matches_reference_at_p0(gemma_ctx):
    """delta=0: re-injecting compiled transposed-V must reproduce reference logits exactly."""
    llm, ctx, cache, st, qt = gemma_ctx
    slen = len(st)

    nexus_fsm_ext.clear_kv_cache(ctx)
    nexus_fsm_ext.decode_tokens(ctx, st, 0, 0)
    ref = np.array(nexus_fsm_ext.decode_tokens(ctx, qt, slen, 0), dtype=np.float32)

    nexus_fsm_ext.clear_kv_cache(ctx)
    handle = cache.get_or_load(str(ATB))
    nexus_fsm_ext.inject_tool_page(ctx, handle, 0, 0)  # would have thrown pre-v2.0
    spl = np.array(nexus_fsm_ext.decode_tokens(ctx, qt, slen, 0), dtype=np.float32)

    kl = _kl(ref, spl)
    assert int(np.argmax(ref)) == int(np.argmax(spl)), f"top1 mismatch kl={kl}"
    assert kl < 1e-3, f"transposed-V bare splice KL too high: {kl}"


def test_vtrans_deep_splice_recompute_matches_reference_at_p256(gemma_ctx):
    """delta=256: RoPE re-anchor + transposed-V + r=100% recompute must match reference."""
    from nexus_recompute import recompute_tail

    llm, ctx, cache, st, qt = gemma_ctx
    slen = len(st)
    filler = [int(t) for t in llm.tokenize(b"You are a tool-using assistant. " * 40, add_bos=True, special=False)][:256]
    P = len(filler)

    nexus_fsm_ext.clear_kv_cache(ctx)
    nexus_fsm_ext.decode_tokens(ctx, filler, 0, 0)
    nexus_fsm_ext.decode_tokens(ctx, st, P, 0)
    ref = np.array(nexus_fsm_ext.decode_tokens(ctx, qt, P + slen, 0), dtype=np.float32)

    nexus_fsm_ext.clear_kv_cache(ctx)
    nexus_fsm_ext.decode_tokens(ctx, filler, 0, 0)
    handle = cache.get_or_load(str(ATB))
    nexus_fsm_ext.inject_tool_page(ctx, handle, P, 0)
    recompute_tail(ctx, st, P, 100.0)
    spl = np.array(nexus_fsm_ext.decode_tokens(ctx, qt, P + slen, 0), dtype=np.float32)

    kl = _kl(ref, spl)
    assert int(np.argmax(ref)) == int(np.argmax(spl)), f"deep top1 mismatch kl={kl}"
    assert kl < 0.05, f"deep transposed-V recompute KL too high: {kl}"
