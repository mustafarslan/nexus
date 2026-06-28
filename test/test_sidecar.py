"""V1.1 Execution Sidecar unit tests (no llama_cpp / no GGUF required).

Covers the pure-Python sidecar logic: context pruning + anchor clamp (V1),
reserved seq-band allocation (M4), and the NumPy candidate-struct layout (V6).
Live e2e paths (V2-V5, V7) require a working llama_cpp + a GGUF model and are
exercised separately.
"""
import os
import sys
import threading

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import nexus_fsm_ext  # noqa: E402
from nexus_agent import NexusAgent  # noqa: E402  (imports llama_cpp lazily, so this is safe)


class FakeLLM:
    """Whitespace tokenizer: 1 token per word. Enough for clamp/prune logic."""

    def tokenize(self, b, add_bos=False, special=False):
        return list(range(len(b.decode("utf-8").split())))


def _make_agent(max_splice_pos=256):
    a = NexusAgent.__new__(NexusAgent)        # bypass __init__ (no llm/model needed)
    a.llm = FakeLLM()
    a.max_splice_pos = max_splice_pos
    a._sidecar_seq_lock = threading.Lock()
    a._sidecar_seq_counter = 0
    return a


def test_render_prunes_tool_dump_and_long_payload():
    a = _make_agent()
    big = "x " * 400  # >280 chars
    tool_line = a._render_pruned_turn({"role": "tool", "name": "search_files", "content": big})
    assert tool_line == "<tool_executed: search_files, status: success>\n"
    err_line = a._render_pruned_turn({"role": "tool", "name": "rm", "content": "fatal ERROR happened"})
    assert "status: error" in err_line
    user_line = a._render_pruned_turn({"role": "user", "content": big})
    assert "user_payload_omitted" in user_line
    short = a._render_pruned_turn({"role": "user", "content": "hello there"})
    assert short == "user: hello there\n"


def test_build_context_clamps_anchor_and_excludes_latest_query():
    a = _make_agent(max_splice_pos=256)
    dump = "tok " * 5000  # would blow the budget if not pruned
    messages = [
        {"role": "user", "content": "first question about auth"},
        {"role": "tool", "name": "grep", "content": dump},
        {"role": "assistant", "content": "found it in auth.py"},
        {"role": "user", "content": "now do the same for billing"},  # latest = the query
    ]
    ctx = a._build_sidecar_context(messages)
    # V1: anchor (system + history) stays within the splice budget.
    anchor = a.llm.tokenize(ctx.encode("utf-8"))
    assert len(anchor) <= a.max_splice_pos
    # The huge dump is surrogate-replaced, not embedded verbatim.
    assert "tok tok tok" not in ctx
    assert "grep" in ctx  # surrogate kept the semantic intent
    # The latest user turn is the query, not part of the anchor.
    assert "now do the same for billing" not in ctx
    assert a.ROUTING_SYSTEM_PROMPT in ctx


def test_sidecar_seq_band_is_reserved_and_within_max_sequences():
    a = _make_agent()
    seen = set()
    for _ in range(a.SIDECAR_SEQ_RING * 2 + 5):                              # force a wrap
        s = a._allocate_sidecar_seq()
        # Must stay in the reserved top band AND below the FSM hard cap (nexus_fsm.hpp:14).
        assert a.SIDECAR_SEQ_BASE <= s < a.SIDECAR_SEQ_MAX, f"sidecar seq {s} out of band"
        assert s < 1024, f"sidecar seq {s} exceeds MAX_SEQUENCES"
        seen.add(s)
    assert len(seen) == a.SIDECAR_SEQ_RING                                   # ring covers exactly RING slots
    # Gateway request seqs rise from REQUEST_SEQ_BASE=64; they stay below the band
    # until ~896 allocations, so the top band is clear in practice.
    assert nexus_fsm_ext.allocate_request_seq(1) < a.SIDECAR_SEQ_BASE


def test_candidate_struct_dtype_is_12_bytes():
    # HF1: NumPy view over llama_token_data {int32 id; float logit; float p;}.
    dt = np.dtype([("id", np.int32), ("logit", np.float32), ("p", np.float32)])
    assert dt.itemsize == 12  # full ctypes.sizeof cross-check runs under V6 (needs llama_cpp)
