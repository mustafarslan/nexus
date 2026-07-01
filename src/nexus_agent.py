import ctypes
import os
import sys
import threading
from pathlib import Path
import numpy as np

# Add the build directory to the Python path to locate the FFI module
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../build"))
if build_dir not in sys.path:
    sys.path.append(build_dir)

try:
    import nexus_fsm_ext
except ImportError as e:
    raise ImportError(
        f"Could not import C++ extension 'nexus_fsm_ext'. "
        f"Ensure the project is built (e.g. via 'cmake -B build && cmake --build build'). "
        f"Error: {e}"
    )


# =============================================================================
# Semantic IR v1.4 — attention-anchor field ordering + one-shot exemplar
# =============================================================================
# High-entropy identifiers the model must bind from the query. Slot confusion
# between adjacent ids (classically owner<->repo) is the dominant residual arg
# error: the model takes a repo name from the query, mis-slots it into `owner`,
# then hallucinates `repo`. Ordering these FIRST in the IR signature and showing
# a typed one-shot exemplar primes the induction heads to map query entities to
# the correct key.
IR_ID_NAMES = {"owner", "repo", "path", "name", "branch", "ref", "title",
               "issue_number", "pull_number", "sha"}
IR_PAYLOAD_NAMES = {"message", "content", "body", "description"}
# v1.6: the JSON exemplar was removed entirely -- it only ever produced leakage (v1.5
# leaked <owner>/<repo>; v1.4 leaked literal "repo"/"main"), never real inference. The
# model now relies solely on the typed signature + native GBNF grammar. IR_EX_STR /
# IR_EX_DROP deleted with it.


# =============================================================================
# Fix 4: RAII Context Manager — Guarantees VRAM cleanup on any exit path
# =============================================================================

class NexusSpliceContext:
    """RAII Context Manager guaranteeing VRAM cleanup on any exit path.

    Ensures unsplice_tool() executes even under:
    - asyncio.CancelledError (client TCP disconnect)
    - KeyboardInterrupt
    - Any Python exception

    The generated token count is tracked locally per sequence context to prevent 
    data races and KV cache corruption in concurrent batching environments.

    Usage:
        with NexusSpliceContext(agent, query, seq_id) as ctx:
            if ctx.resolved_tool_id == 0:
                ...  # no tool matched
            # ... generate tokens (ctx.local_generated_count grows) ...
        # __exit__ auto-computes delta and unsplices correctly
    """

    def __init__(self, agent, query, seq_id=0):
        self.agent = agent
        self.query = query
        self.seq_id = seq_id
        self.resolved_tool_id = 0
        self.tool_name = None
        self._spliced = False
        self._n_past = 0
        self._schema_len = 0
        self._query_len = 0
        self.local_generated_count = 0  # Thread-safe local generated token tracking
        self.hazard_guard = None

    def __enter__(self):
        # Perform routing and splice
        self.resolved_tool_id, self.tool_name = self.agent.route_and_splice(
            self.query, self.seq_id
        )
        if self.resolved_tool_id != 0:
            self._spliced = True
            self._n_past = self.agent.last_splice_pos
            self._schema_len = self.agent.last_schema_len
            self._query_len = self.agent.last_query_len
            try:
                self.hazard_guard = self.agent.orchestrator.get_cache().hazard_guard(self.resolved_tool_id)
                self.hazard_guard.__enter__()
            except Exception as e:
                try:
                    nexus_fsm_ext.unsplice_tool(
                        self.agent.ctx_addr,
                        self.agent.orchestrator,
                        self.seq_id,
                        self._n_past,
                        self._schema_len,
                        self._query_len,
                    )
                except Exception:
                    pass
                self._spliced = False
                self.hazard_guard = None
                raise e
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # GUARANTEED cleanup regardless of exception type.
        # This runs even on asyncio.CancelledError, KeyboardInterrupt, SystemExit.
        try:
            if self._spliced:
                # Use local_generated_count directly to calculate RoPE shift delta
                generated_count = self.local_generated_count

                nexus_fsm_ext.unsplice_tool(
                    self.agent.ctx_addr,
                    self.agent.orchestrator,
                    self.seq_id,
                    self._n_past,
                    self._schema_len,
                    self._query_len + generated_count,
                )
        except Exception as e:
            print(f"WARNING: unsplice_tool cleanup failed: {e}", file=sys.stderr)
        finally:
            if self.hazard_guard is not None:
                try:
                    self.hazard_guard.__exit__(exc_type, exc_val, exc_tb)
                except Exception:
                    pass
                self.hazard_guard = None
            self._spliced = False

        # Don't suppress the original exception
        return False


# =============================================================================
# Fix 3: NexusRoutingProcessor — FSM mask ONLY (no grammar)
# =============================================================================

class NexusRoutingProcessor:
    """LogitsProcessor that applies ONLY the Nexus Radix FSM routing mask.

    Nexus is an MMU, not a grammar engine. JSON constrained decoding is
    delegated to the native host server (e.g., llama_cpp.LlamaGrammar).

    This processor applies the FSM mask during the NAVIGATING routing phase
    and becomes a no-op once a tool leaf is reached.

    Conforms to llama_cpp.LogitsProcessor protocol:
        __call__(input_ids: NDArray[np.intc], scores: NDArray[np.single]) -> NDArray[np.single]
    """

    def __init__(self, agent, seq_id=0):
        self.agent = agent
        self.seq_id = seq_id

    def __call__(self, input_ids, scores):
        """Called by llm.generate() after each decode step.

        Args:
            input_ids: NDArray[np.intc] — tokens generated so far
            scores: NDArray[np.single] — raw logits from the model (vocab_size,)

        Returns:
            Modified scores array with FSM routing mask applied in-place.
        """
        # Apply FSM mask ONLY if actively routing
        fsm_state = self.agent.fsm.get_state(self.seq_id)
        if fsm_state == nexus_fsm_ext.RoutingState.NAVIGATING:
            self.agent.fsm.apply_logit_mask(scores, self.seq_id)

        return scores


# =============================================================================
# NexusAgent — Core agent with RAII + native grammar delegation
# =============================================================================

class NexusAgent:
    MAX_SPLICE_POS = 256
    _V3_CE_PATH = Path(__file__).resolve().parents[1] / "results/tool_cross_encoder_finetuned_v3"

    # --- V1.1 Depth-Decoupled Execution Sidecar ---
    ROUTING_SYSTEM_PROMPT = (
        "You are a tool router. Given the conversation, pick the single best tool.\n"
    )
    # v1.8: + dense kebab-case rule so spaced repo/owner names are sanitized in-model
    # ("project repo" -> "project-repo"), replacing the v1.7 generous case-6 gold relabel.
    # v1.7 base: explicit owner default anchors owner to 'user' when unspecified, relieving
    # the pressure to mis-slot the query's repo name into owner (the v1.3-v1.6 residual).
    # owner is never gold-scored on underspecified queries, so the default is free.
    ARG_GEN_DIRECTIVE = (
        "CRITICAL: Extract args from query. "
        "Kebab-case spaced repo/owner (e.g., 'a b' -> 'a-b'). "
        "Default owner is 'user' if unspecified. "
        "OMIT unspecified optional fields.\n"
    )
    EMBED_CHAR_LIMIT = 8000          # tail-truncate the query before embedding (HF2)
    # Sidecar seqs occupy a reserved band at the TOP of the FSM seq space. The hard
    # cap is C++ MAX_SEQUENCES = 1024 (nexus_fsm.hpp:14); any seq_id >= 1024 is
    # rejected. allocate_request_seq (base 64) can exceed that, so sidecars use raw
    # ids in [SIDECAR_SEQ_BASE, MAX). The gateway's request seqs rise from 64, so the
    # top band stays clear in practice (full isolation needs gateway coordination).
    SIDECAR_SEQ_MAX = 1024           # must match nexus_fsm.hpp MAX_SEQUENCES
    SIDECAR_SEQ_RING = 64            # distinct sidecar seq slots
    SIDECAR_SEQ_BASE = SIDECAR_SEQ_MAX - SIDECAR_SEQ_RING   # sidecars use [960, 1023]

    def __init__(self, llm, embedding_llm=None, dim=384, base_pos=256, auto_route_margin=0.10,
                 max_splice_pos=256, rerank_margin=None):
        """
        Integrates GGUF inference (llama_cpp.Llama) with Project Nexus C++ extensions.

        Args:
            llm: An instance of llama_cpp.Llama.
            embedding_llm: Optional separate embedding llama_cpp.Llama instance.
            dim: Dimension of embeddings (e.g. 768 for Nomic Embed, 384 for MiniLM).
            base_pos: Starting position offset in RoPE context for tools injection.
            auto_route_margin: Confidence margin for SLB Auto-Routing Fast Path.
        """
        self.llm = llm
        self.embedding_llm = embedding_llm
        self.dim = embedding_llm.n_embd() if embedding_llm is not None else dim
        self.base_pos = base_pos
        self.max_splice_pos = max_splice_pos
        if rerank_margin is None:
            try:
                from nexus_calibration import CALIBRATED_MARGIN_THRESHOLD as rerank_margin
            except ImportError:
                rerank_margin = 0.028
        self.rerank_margin = rerank_margin
        self.tool_id_to_name = {}
        self.tool_name_to_id = {}
        self.tool_verb_prefix = {}
        self.tool_embeddings = {}
        self.tool_records = []
        self.llm_rerank = None
        self.cross_encoder = self._default_cross_encoder()
        self.tool_grammars = {}
        self.tool_schema_lengths = {}
        self.tool_schema_tokens = {}
        self.last_query_len = 0
        self.last_splice_pos = 0
        self.last_schema_len = 0

        # Sidecar concurrency state (M1/M5). RLock so nested raw-ctx helpers re-enter.
        self._ctx_lock = threading.RLock()
        self._sidecar_seq_counter = 0
        self._sidecar_seq_lock = threading.Lock()

        # 1. Safely extract the raw llama_context C pointer address
        self.ctx_addr = self._extract_context_pointer(llm)
        if not self.ctx_addr:
            raise ValueError("Extracted llama_context pointer address is null (0).")

        # 2. Instantiate FFI C++ components
        self.slb = nexus_fsm_ext.NexusSemanticSLB(self.dim)
        self.fsm = nexus_fsm_ext.NexusRadixFSM()
        self.orchestrator = nexus_fsm_ext.NexusOrchestrator(
            self.ctx_addr, self.slb, self.fsm, self.base_pos, 0.88, 0.05, auto_route_margin,
            16 * 1024 * 1024 * 1024, max_splice_pos,
        )

    @staticmethod
    def _compute_verb_prefix(name: str) -> str:
        i = name.find("_")
        return name[: i + 1] if i >= 0 else name

    def _extract_context_pointer(self, llm):
        """Extracts the underlying llama_context void pointer as an integer address."""
        # Try finding in the Llama internal context wrappers
        for attr_name in ("_ctx", "ctx"):
            if hasattr(llm, attr_name):
                ctx_obj = getattr(llm, attr_name)
                # If the attribute matches LlamaContext (which has .ctx field)
                if hasattr(ctx_obj, "ctx") and ctx_obj.ctx is not None:
                    ptr = ctx_obj.ctx
                else:
                    ptr = ctx_obj

                if ptr is not None:
                    # Resolve ctypes address
                    if hasattr(ptr, "value"):
                        return ptr.value
                    try:
                        return ctypes.cast(ptr, ctypes.c_void_p).value
                    except Exception:
                        pass
                    
                    # Try casting directly as integer if it's already an integer address
                    try:
                        return int(ptr)
                    except Exception:
                        pass

        raise TypeError("Could not extract a valid llama_context pointer address from the Llama object.")

    def register_tool(self, tool_id, name, embedding, digest_text, atb_path, schema=None):
        """
        Registers a tool in the Radix FSM, Semantic SLB, and Orchestrator.

        Args:
            tool_id: Unique uint32 ID for the tool.
            name: String name of the tool.
            embedding: Vector embedding (numpy array or list) of shape (dim,).
            digest_text: String representing the semantic digest for the tool.
            atb_path: Filepath to compiled .atb block for the tool.
            schema: Optional JSON schema (dict or JSON string) for the tool arguments.
        """
        if len(embedding) != self.dim:
            raise ValueError(f"Embedding size must match agent dim {self.dim}, got {len(embedding)}")

        # Tokenize the tool name (this forms the Radix FSM route)
        route_tokens = self.llm.tokenize(name.encode("utf-8"), add_bos=False, special=False)
        route_tokens_list = [int(t) for t in route_tokens]

        # FFI registers
        self.fsm.add_route(tool_id, route_tokens_list)
        
        vec_np = np.asarray(embedding, dtype=np.float32)
        digest_tokens = self.llm.tokenize(digest_text.encode("utf-8"), add_bos=False, special=False)
        digest_list = [int(t) for t in digest_tokens]
        from nexus_retrieval import lexical_hashes, tool_document_text

        lex = lexical_hashes(tool_document_text({"name": name, "description": digest_text}))
        self.slb.register_tool(tool_id, vec_np, digest_list, lex)
        
        self.orchestrator.preload_tool(tool_id, atb_path)
        self.tool_id_to_name[tool_id] = name
        self.tool_name_to_id[name] = tool_id
        self.tool_embeddings[tool_id] = vec_np
        verb_prefix = self._compute_verb_prefix(name)
        self.tool_verb_prefix[name] = verb_prefix
        self.tool_records.append({
            "name": name,
            "description": digest_text,
            "inputSchema": schema or {},
            "verb_prefix": verb_prefix,
        })

        # Compile and cache GBNF grammar string for native host delegation
        if schema:
            import json
            from llama_cpp.llama_grammar import json_schema_to_gbnf
            try:
                schema_str = json.dumps(schema) if isinstance(schema, dict) else schema
                gbnf_str = json_schema_to_gbnf(schema_str)
                self.tool_grammars[tool_id] = gbnf_str
            except Exception as e:
                sys.stderr.write(f"Warning: Failed to compile JSON schema to GBNF for tool {name}: {e}\n")

        # Read schema length from ATB header (offset 28 is seq_len)
        with open(atb_path, "rb") as f:
            f.seek(28)
            schema_len = int.from_bytes(f.read(4), byteorder="little")
        self.tool_schema_lengths[tool_id] = schema_len

        if schema:
            import json
            schema_str = json.dumps({
                "name": name,
                "description": digest_text,
                "inputSchema": schema if isinstance(schema, dict) else json.loads(schema),
            }, sort_keys=True)
            schema_tokens = [int(t) for t in self.llm.tokenize(schema_str.encode("utf-8"), add_bos=False, special=False)]
            self.tool_schema_tokens[tool_id] = schema_tokens
            self.orchestrator.register_tool_schema_tokens(tool_id, schema_tokens)

    @staticmethod
    def _default_cross_encoder():
        ce_path = NexusAgent._V3_CE_PATH
        if not ce_path.is_dir():
            raise RuntimeError(
                "FATAL: v3 CrossEncoder not found at results/tool_cross_encoder_finetuned_v3. "
                "Run: python scripts/train_cross_encoder.py --output results/tool_cross_encoder_finetuned_v3"
            )
        import sys
        test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../test"))
        if test_dir not in sys.path:
            sys.path.append(test_dir)
        from nexus_retrieval import CrossEncoderReranker
        ce = CrossEncoderReranker(str(ce_path))
        if not ce.available:
            raise RuntimeError(f"FATAL: v3 CrossEncoder failed to load from {ce_path}")
        return ce

    def configure_reranker(self, llm_rerank=None, cross_encoder=None):
        """Attach isolated LLM (offline ceiling) or cross-encoder reranker for margin-gated routing."""
        self.llm_rerank = llm_rerank
        if cross_encoder is not None:
            self.cross_encoder = cross_encoder

    def pin_hot_tools(self, tool_ids):
        """Pin top-N tools in L0 warm tier and ATB block cache."""
        for tool_id in tool_ids:
            self.orchestrator.pin_warm_tool(int(tool_id))
            self.orchestrator.pin_block_tool(int(tool_id))

    def enable_deep_splice(self, full_mult: float = 4.0):
        """Phase v2.0: allow the .atb splice to run past max_splice_pos (deep multi-turn
        context) instead of declining to a Python text prefill. The orchestrator repairs
        RoPE Δθ drift with a depth-adaptive recompute fraction that ramps to 100% (==
        re-prefill, KL=0) by max_splice_pos*full_mult, so top-1 never regresses below the
        text-prefill baseline. Call AFTER registering tools (schema tokens must exist for
        the drift repair to run)."""
        self.orchestrator.set_deep_splice(True, float(full_mult))

    def deep_path_telemetry(self):
        """F0 deep-path (P > MAX_SPLICE_POS) read-only probe counters.

        l0_hit counts position-safe FULL prefix matches the probe *would* copy;
        it is opportunity, not realized savings — the text fallback still runs this
        phase. Invariants: entered == l0_hit + l0_miss; text_fallback == entered.
        """
        o = self.orchestrator
        return {
            "deep_path_entered": o.get_deep_path_entered(),
            "deep_path_l0_hit": o.get_deep_path_l0_hit(),
            "deep_path_l0_miss": o.get_deep_path_l0_miss(),
            "deep_path_text_fallback": o.get_deep_path_text_fallback(),
            # raw best_end recording for offline histogram/CDF (read after each deep call)
            "last_deep_n_past": o.get_last_deep_n_past(),
            "last_deep_best_end": o.get_last_deep_best_end(),
        }

    # =========================================================================
    # V1.1 Depth-Decoupled Execution Sidecar
    # =========================================================================

    def _allocate_sidecar_seq(self):
        """Fresh sidecar seq from a reserved band at the top of the FSM seq space.

        Returns a RAW seq_id in [SIDECAR_SEQ_BASE, SIDECAR_SEQ_MAX) -- NOT via
        allocate_request_seq, which (base 64, %4096) can exceed MAX_SEQUENCES=1024
        and would be rejected by the FSM (nexus_fsm.hpp:14). The ring reuses slots;
        sidecars are torn down synchronously per call, so a 64-slot ring is ample.
        """
        with self._sidecar_seq_lock:
            slot = self._sidecar_seq_counter % self.SIDECAR_SEQ_RING
            self._sidecar_seq_counter += 1
        return int(self.SIDECAR_SEQ_BASE + slot)

    def _prefill_sequence(self, tokens, seq_id):
        """Prefill tokens into seq_id at positions 0..n-1 (logits on last only).

        Decoded in n_batch-sized chunks: a single llama_decode batch may not exceed
        cparams.n_batch (GGML_ASSERT n_tokens_all <= n_batch). Holds _ctx_lock:
        existing raw-decode sites take no Python lock and C++ guards its decode with
        a separate, unexposed context_mutex_; two raw llama_decode calls on one
        context from different threads segfault the backend, so every Python raw-ctx
        section serializes on this lock.
        """
        import llama_cpp
        n = len(tokens)
        if n == 0:
            return 0
        try:
            n_batch = int(llama_cpp.llama_n_batch(self.ctx_addr))
        except Exception:
            n_batch = 512
        n_batch = max(1, n_batch)
        with self._ctx_lock:
            for start in range(0, n, n_batch):
                chunk = tokens[start:start + n_batch]
                m = len(chunk)
                batch = llama_cpp.llama_batch_init(m, 0, 1)
                try:
                    batch.n_tokens = m
                    for j, tok in enumerate(chunk):
                        pos = start + j
                        batch.token[j] = int(tok)
                        batch.pos[j] = pos
                        batch.n_seq_id[j] = 1
                        batch.seq_id[j][0] = seq_id
                        batch.logits[j] = 1 if pos == n - 1 else 0
                    if llama_cpp.llama_decode(self.ctx_addr, batch) != 0:
                        raise RuntimeError("sidecar prefill llama_decode failed")
                finally:
                    llama_cpp.llama_batch_free(batch)
        return n

    def _render_pruned_turn(self, msg, prune=True):
        """One history turn, with bulky payloads replaced by semantic surrogates."""
        role = msg.get("role", "user")
        content = msg.get("content", "") or ""
        if role == "tool" or msg.get("tool_call_id"):
            nm = msg.get("name", "tool")
            status = "error" if "error" in content[:200].lower() else "success"
            return content if not prune else f"<tool_executed: {nm}, status: {status}>\n"
        if prune and len(content) > 280:
            return f"{role}: <{role}_payload_omitted len={len(content)}c>\n"
        return f"{role}: {content}\n"

    def _build_sidecar_context(self, messages) -> str:
        """Routing prompt + pruned newest-first sliding window, clamped so the
        SPLICE ANCHOR (system + history) stays <= max_splice_pos - SAFETY.

        The latest user turn is the routing query; it is passed to route_and_splice
        separately and prefilled AFTER the anchor (orchestrator.cpp:525). The Path-A
        guard checks only n_past (orchestrator.cpp:69), so query length is exempt
        from this budget and must NOT be truncated here.
        """
        SAFETY = 16
        sys_toks = self.llm.tokenize(
            self.ROUTING_SYSTEM_PROMPT.encode("utf-8"), add_bos=True, special=True
        )
        budget = self.max_splice_pos - SAFETY - len(sys_toks)
        history = messages[:-1] if (messages and messages[-1].get("role") == "user") else messages
        chosen, kept_latest = [], False
        for msg in reversed(history):                       # newest-first
            keep_full = msg.get("role") == "tool" and not kept_latest
            line = self._render_pruned_turn(msg, prune=not keep_full)
            if msg.get("role") == "tool":
                kept_latest = True
            toks = self.llm.tokenize(line.encode("utf-8"), add_bos=False, special=False)
            if len(toks) > budget:
                if keep_full:                              # too big even un-pruned -> prune it
                    line = self._render_pruned_turn(msg, prune=True)
                    toks = self.llm.tokenize(line.encode("utf-8"), add_bos=False, special=False)
                    if len(toks) > budget:
                        break
                else:
                    break
            budget -= len(toks)
            chosen.append(line)
        chosen.reverse()
        return self.ROUTING_SYSTEM_PROMPT + "".join(chosen)

    def _compress_schema_to_ir(self, tool_schema: dict, desc_chars: int = 0) -> str:
        """MCP JSON schema -> dense TYPE-HINTED SEMANTIC signature (~tens of tokens,
        far smaller than the full schema). Optional params (absent from `required`)
        get a '?'. v1.6: inline descriptions default OFF (desc_chars=0) and the
        one-shot JSON exemplar is gone -- both were token bloat that pushed TTFT over
        the 500ms bar, and the exemplar leaked literals into generated args. Pass
        desc_chars>0 to re-enable truncated `/* desc */` comments per field.

        {"name":"create_repository","inputSchema":{"properties":{
          "name":{"type":"string"},"private":{"type":"boolean"}},
          "required":["name"]}}
          -> "create_repository(name: string, private?: boolean)"
        """
        import json
        name = tool_schema.get("name") or tool_schema.get("title") or "tool"
        schema = (tool_schema.get("inputSchema") or tool_schema.get("input_schema")
                  or tool_schema or {})
        props = schema.get("properties", {}) or {}
        required = set(schema.get("required", []) or [])

        def _ts(spec):
            if not isinstance(spec, dict):
                return "any"
            if isinstance(spec.get("enum"), list):
                return "|".join(json.dumps(v, separators=(",", ":")) for v in spec["enum"])
            t = spec.get("type", "any")
            if isinstance(t, list):
                return "|".join(str(x) for x in t)
            if t == "array":
                return f"{_ts(spec.get('items', {}))}[]"
            return "object" if t == "object" else t

        def _desc(spec):
            d = spec.get("description", "") if isinstance(spec, dict) else ""
            d = " ".join((d or "").split())[:desc_chars]   # strip newlines, hard cap
            return f" /* {d} */" if d else ""

        # --- v1.4: deterministic field ordering (high-entropy ids first) ---
        def _rank(p):
            pl = p.lower()
            is_req = p in required
            if is_req and pl in IR_ID_NAMES:
                return 0                       # P1: required identifiers
            if pl in IR_ID_NAMES:
                return 1                       # identifiers (optional)
            if pl in IR_PAYLOAD_NAMES:
                return 2                       # P2: semantic payloads
            if is_req:
                return 3                       # other required
            return 4                           # P3: optional / boolean / default

        # sorted() is stable -> ties preserve the schema's own field order.
        ordered = sorted(props, key=_rank)

        params = ", ".join(
            f"{p}{'' if p in required else '?'}: {_ts(props[p])}{_desc(props[p])}"
            for p in ordered
        )
        sig = f"{name}({params})"
        # v1.6: no exemplar. The typed signature + native GBNF grammar are the only
        # grounding; the JSON exemplar only ever leaked literals into the args.
        return sig

    def route_via_sidecar(self, messages):
        """V1.2 routing: resolve the tool NAME via the authoritative Dense+CE router.

        Routing is pure retrieval over the query embedding + tool embeddings -- it
        never touches the deep llama_context, so NO llama 'sidecar' sequence and NO
        ATB splice are needed. (The splice only ever existed to host KV-based argument
        generation, which the Hybrid removes; see generate_via_hybrid.) It does NOT
        call _generate_arguments. Returns (tool_name, resolved_id, meta).
        """
        test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../test"))
        if test_dir not in sys.path:
            sys.path.append(test_dir)
        from nexus_retrieval import embed_query, route_tool

        query = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        emb_model = self.embedding_llm if self.embedding_llm is not None else self.llm
        emb_str = query[-self.EMBED_CHAR_LIMIT:] if len(query) > self.EMBED_CHAR_LIMIT else query
        q_emb = embed_query(emb_model, emb_str)
        tool_names = [self.tool_id_to_name[tid] for tid in sorted(self.tool_id_to_name)]
        tool_embs = np.stack([self.tool_embeddings[self.tool_name_to_id[n]] for n in tool_names])
        tools_by_name = {t["name"]: t for t in self.tool_records}
        selected, rerank_used, rerank_us, margin, _ = route_tool(
            emb_str, q_emb, tool_embs, tool_names, self.tool_records, tools_by_name,
            margin_threshold=self.rerank_margin, llm_rerank=self.llm_rerank,
            cross_encoder=self.cross_encoder,
            rerank_mode="cross_encoder" if self.cross_encoder is not None else "llm",
        )
        rid = self.tool_name_to_id.get(selected, 0)
        return selected, rid, {"margin": margin, "rerank_used": rerank_used, "rerank_us": rerank_us}

    def generate_via_hybrid(self, messages, max_tokens=160, main_seq_id=0, schema_text=None):
        """V1.2 Hybrid turn: route via retrieval, then generate ARGUMENTS in the main
        context grounded on a compressed TEXT IR of the schema (NOT spliced KV --
        spliced-KV arg-gen degrades fidelity at the splice seam).

        Coarse-grained concurrency: the ENTIRE turn is one uninterrupted critical
        section under _ctx_lock.
        WARNING: NexusAgent is single-threaded per llama_context. Concurrent requests
        are serialized by _ctx_lock (the C++ context_mutex_ is not exposed to Python,
        so granular locking segfaults the backend).

        `schema_text` overrides the IR (the benchmark passes full schema text to build
        the Oracle arm through this exact code path). Returns (tool_name, args_json, meta).
        """
        import llama_cpp
        with self._ctx_lock:
            tool_name, resolved_id, meta = self.route_via_sidecar(messages)
            if resolved_id == 0:
                return None, None, {**meta, "error": "router_failed"}

            record = next((r for r in self.tool_records
                           if self.tool_name_to_id.get(r["name"]) == resolved_id), None)
            grounding = schema_text if schema_text is not None else (
                self._compress_schema_to_ir(record) if record else tool_name)

            # Main-context prompt: full text history (for coreference) + text grounding
            # + an arg cue. The GBNF grammar then emits the JSON object.
            history_text = "".join(self._render_pruned_turn(m, prune=False) for m in messages)
            prompt = (f"{history_text}<tool_schema>{grounding}</tool_schema>\n"
                      f"{self.ARG_GEN_DIRECTIVE}"
                      f"assistant (calling {tool_name}): ")
            tokens = [int(t) for t in self.llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True)]
            max_ctx = llama_cpp.llama_n_ctx(self.ctx_addr)
            if len(tokens) + max_tokens > max_ctx:
                tokens = tokens[-(max_ctx - max_tokens):]

            llama_cpp.llama_kv_cache_seq_rm(self.ctx_addr, main_seq_id, -1, -1)
            self._prefill_sequence(tokens, main_seq_id)
            _, _, args_json, _ = self._generate_arguments(
                resolved_id, main_seq_id, len(tokens), 0, 0, max_tokens)
            llama_cpp.llama_kv_cache_seq_rm(self.ctx_addr, main_seq_id, -1, -1)

            meta["ir"] = grounding
            meta["ir_tokens"] = len(self.llm.tokenize(grounding.encode("utf-8"),
                                                      add_bos=False, special=False))
            return tool_name, args_json, meta

    def route_with_retrieval(self, query, seq_id=0):
        """Python hybrid retrieval + margin-gated rerank, then orchestrator splice."""
        from nexus_retrieval import embed_query, embed_tools, route_tool

        emb_model = self.embedding_llm if self.embedding_llm is not None else self.llm
        q_emb = embed_query(emb_model, query)
        tool_names = [self.tool_id_to_name[tid] for tid in sorted(self.tool_id_to_name)]
        tools = self.tool_records
        tool_embs = np.stack([self.tool_embeddings[self.tool_name_to_id[n]] for n in tool_names])
        tools_by_name = {t["name"]: t for t in tools}
        selected, rerank_used, rerank_us, margin, _ = route_tool(
            query, q_emb, tool_embs, tool_names, tools, tools_by_name,
            margin_threshold=self.rerank_margin,
            llm_rerank=self.llm_rerank,
            cross_encoder=self.cross_encoder,
            rerank_mode="cross_encoder" if self.cross_encoder is not None else "llm",
        )
        tool_id = self.tool_name_to_id.get(selected, 0)
        resolved_id, tool_name = self.route_and_splice(query, seq_id, force_tool_id=tool_id)
        if resolved_id == 0 and tool_id:
            resolved_id = self._prefix_cache_text_fallback(query, tool_id, seq_id)
            tool_name = self.tool_id_to_name.get(resolved_id)
        return resolved_id, tool_name, {
            "rerank_used": rerank_used,
            "rerank_us": rerank_us,
            "margin": margin,
            "selected": selected,
        }

    def _prefix_cache_text_fallback(self, query, tool_id, seq_id=0):
        """B3pc-style text prefill when splice guard blocks ATB inject (P > max_splice_pos)."""
        import json

        tool_name = self.tool_id_to_name.get(tool_id)
        record = next((t for t in self.tool_records if t["name"] == tool_name), None)
        if not record:
            return 0
        schema_str = json.dumps({
            "name": record["name"],
            "description": record.get("description", ""),
            "inputSchema": record.get("inputSchema") or {},
        }, sort_keys=True)
        schema_tokens = [int(t) for t in self.llm.tokenize(schema_str.encode("utf-8"), add_bos=False, special=False)]
        query_tokens = [int(t) for t in self.llm.tokenize(query.encode("utf-8"), add_bos=False, special=False)]
        n_past = self.llm.n_tokens
        nexus_fsm_ext.invalidate_sequence(self.ctx_addr, seq_id, n_past, -1)
        if schema_tokens:
            nexus_fsm_ext.decode_tokens(self.ctx_addr, schema_tokens, n_past, seq_id)
        if query_tokens:
            nexus_fsm_ext.decode_tokens(self.ctx_addr, query_tokens, n_past + len(schema_tokens), seq_id)
        self.last_splice_pos = n_past
        self.last_schema_len = len(schema_tokens)
        self.last_query_len = len(query_tokens)
        return tool_id

    def decode_tool_name_fsm(self, seq_id=0, max_tokens=32):
        """Greedy FSM-masked tool-name decode (production routing guarantee).

        Locking mirrors _generate_arguments: {llama_decode + logits-copy} held
        atomically on _ctx_lock; mask/argmax run unlocked. Logits copied via NumPy
        (HF1 -- no per-token 152k Python loop).
        """
        import llama_cpp

        decoded = []
        pos = self.llm.n_tokens
        n_vocab = llama_cpp.llama_n_vocab(self.llm.model)

        def _copy_logits():
            lp = llama_cpp.llama_get_logits(self.ctx_addr)
            return np.ctypeslib.as_array(lp, shape=(n_vocab,)).copy()

        with self._ctx_lock:                               # ONE coarse critical section
            logits = _copy_logits()
            for _ in range(max_tokens):
                self.fsm.apply_logit_mask(logits, seq_id)
                pred = int(np.argmax(logits))
                state = self.fsm.advance(pred, seq_id)
                decoded.append(pred)
                if state == nexus_fsm_ext.RoutingState.LEAF_REACHED:
                    break
                one_batch = llama_cpp.llama_batch_init(1, 0, 1)
                try:
                    one_batch.n_tokens = 1
                    one_batch.token[0] = pred
                    one_batch.pos[0] = pos
                    one_batch.n_seq_id[0] = 1
                    one_batch.seq_id[0][0] = seq_id
                    one_batch.logits[0] = 1
                    if llama_cpp.llama_decode(self.ctx_addr, one_batch) != 0:
                        raise RuntimeError(f"fsm decode failed at pos {pos}")
                finally:
                    llama_cpp.llama_batch_free(one_batch)
                logits = _copy_logits()
                pos += 1
        return self.llm.detokenize(decoded).decode("utf-8", errors="replace")

    def route_and_splice(self, query, seq_id=0, force_tool_id=0):
        """
        Invokes C++ route_and_splice.

        Args:
            query: String query from user.
            seq_id: Sequence ID for continuous batching.
            force_tool_id: When non-zero, bypass SLB and splice this tool (Python reranker decision).

        Returns:
            Tuple of (resolved_tool_id, tool_name)
        """
        query_tokens = self.llm.tokenize(query.encode("utf-8"), add_bos=False, special=False)
        query_tokens_list = [int(t) for t in query_tokens]
        Q = len(query_tokens_list)
        self.last_query_len = Q

        # Use dedicated embedding model if present, otherwise fallback to generative
        emb_model = self.embedding_llm if self.embedding_llm is not None else self.llm
        test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../test"))
        if test_dir not in sys.path:
            sys.path.append(test_dir)
        from nexus_retrieval import lexical_hashes, embed_query

        # Embed via the shared helper so this path matches route_with_retrieval: it applies
        # the nomic "search_query:" task prefix AND L2-normalizes. The previous raw
        # emb_model.embed(query) did neither, which collapsed SLB recall on the splice path
        # (probe: prefix + name-in-document lifts R@1 40%->74%, routing 68%->83% at N=250).
        query_embedding_list = [float(x) for x in embed_query(emb_model, query)]

        n_past = self.llm.n_tokens
        prefix_tokens: list[int] = []
        if hasattr(self.llm, "input_ids") and n_past > 0:
            prefix_tokens = [int(t) for t in self.llm.input_ids[:n_past]]

        resolved_id = self.orchestrator.route_and_splice(
            query_tokens_list, query_embedding_list, n_past, seq_id, lexical_hashes(query),
            prefix_tokens, int(force_tool_id),
        )
        
        self.last_splice_pos = n_past
        self.last_schema_len = self.tool_schema_lengths.get(resolved_id, 0) if resolved_id != 0 else 0
        
        return resolved_id, self.tool_id_to_name.get(resolved_id, None)

    def generate_with_tool(self, query, seq_id=0, max_tokens=512):
        """
        Legacy splice path: routing, tool KV-splicing, grammar-constrained generation,
        and unsplicing. NOTE: argument fidelity over spliced KV is degraded at the
        splice seam (see results/v2.0_canonical/raw/sidecar/accuracy.json) -- prefer
        generate_via_hybrid. Kept for the splice regression path.

        Coarse-grained concurrency (Decision 2): the whole turn is one _ctx_lock
        critical section (single-threaded per llama_context).
        """
        with self._ctx_lock:
            with NexusSpliceContext(self, query, seq_id) as ctx:
                if ctx.resolved_tool_id == 0:
                    return 0, None, ""
                resolved_id, tool_name, generated_json, gen_count = self._generate_arguments(
                    ctx.resolved_tool_id, seq_id, ctx._n_past, ctx._schema_len, ctx._query_len, max_tokens
                )
                ctx.local_generated_count = gen_count   # preserve unsplice contract (__exit__)

            query_tokens = self.llm.tokenize(query.encode("utf-8"), add_bos=False, special=False)
            generated_tokens = self.llm.tokenize(generated_json.encode("utf-8"), add_bos=False, special=False)
            generated_tokens_list = [int(t) for t in generated_tokens]
            n_past = self.last_splice_pos
            Q = self.last_query_len
            for i, tok in enumerate(query_tokens):
                self.llm.input_ids[n_past + i] = tok
            for i, tok in enumerate(generated_tokens_list):
                self.llm.input_ids[n_past + Q + i] = tok
            new_n_tokens = n_past + Q + len(generated_tokens_list)
            self.llm.n_tokens = new_n_tokens
            if hasattr(self.llm, "_ctx") and hasattr(self.llm._ctx, "n_tokens"):
                self.llm._ctx.n_tokens = new_n_tokens
            return resolved_id, tool_name, generated_json

    def _generate_arguments(self, resolved_tool_id, seq_id, n_past, schema_len, query_len, max_tokens):
        """GBNF-constrained argument generation at seq_id, starting at
        n_past+schema_len+query_len (pass schema_len=query_len=0 for the Hybrid
        text path, where the schema/query are already in the prefill).

        Concurrency (Decision 2, coarse-grained): the ENTIRE decode loop runs under a
        SINGLE _ctx_lock acquisition -- granular per-token locking raced the C++
        context_mutex_ and SIGSEGV'd. _ctx_lock is an RLock, so this nests safely
        inside a turn-level lock (generate_via_hybrid / generate_with_tool).
        HF1: logits copied via NumPy (no 152k-iteration Python loop).

        Returns (resolved_tool_id, tool_name, generated_json, generated_count).
        """
        import llama_cpp

        grammar_str = self.tool_grammars.get(resolved_tool_id, "")
        native_grammar = llama_cpp.LlamaGrammar.from_string(grammar_str) if grammar_str else None

        model = self.llm
        n_vocab = llama_cpp.llama_n_vocab(model.model)
        current_pos = n_past + schema_len + query_len
        generated_json = ""
        generated_count = 0

        # Allocate the candidate struct array ONCE; map it with NumPy (zero-copy view).
        # dtype mirrors llama_token_data {int32 id; float logit; float p;} (12B, no pad).
        candidates = (llama_cpp.llama_token_data * n_vocab)()
        cand_dtype = np.dtype([("id", np.int32), ("logit", np.float32), ("p", np.float32)])
        cand_np = np.frombuffer(candidates, dtype=cand_dtype)
        cand_np["id"] = np.arange(n_vocab, dtype=np.int32)
        cand_np["p"] = 0.0

        def _copy_logits():                                # vectorized C->numpy copy (<1ms)
            lp = llama_cpp.llama_get_logits(self.ctx_addr)
            return np.ctypeslib.as_array(lp, shape=(n_vocab,)).copy()

        with self._ctx_lock:                               # ONE coarse critical section
            raw_logits = _copy_logits()                    # prime: post-prefill logits
            while generated_count < max_tokens:
                cand_np["logit"] = raw_logits
                candidates_p = llama_cpp.llama_token_data_array(candidates, n_vocab, False)

                if native_grammar:
                    llama_cpp.llama_sample_grammar(
                        self.ctx_addr, ctypes.byref(candidates_p), native_grammar.grammar
                    )
                token_id = llama_cpp.llama_sample_token_greedy(
                    self.ctx_addr, ctypes.byref(candidates_p)
                )
                if native_grammar:
                    llama_cpp.llama_grammar_accept_token(
                        self.ctx_addr, native_grammar.grammar, token_id
                    )
                if llama_cpp.llama_token_is_eog(model.model, token_id):
                    break

                generated_json += model.detokenize([token_id]).decode("utf-8", errors="ignore")
                generated_count += 1

                one_batch = llama_cpp.llama_batch_init(1, 0, 1)
                try:
                    one_batch.n_tokens = 1
                    one_batch.token[0] = token_id
                    one_batch.pos[0] = current_pos
                    one_batch.n_seq_id[0] = 1
                    one_batch.seq_id[0][0] = seq_id
                    one_batch.logits[0] = 1
                    if llama_cpp.llama_decode(self.ctx_addr, one_batch) != 0:
                        raise RuntimeError(f"Failed to decode token {token_id} at position {current_pos}")
                finally:
                    llama_cpp.llama_batch_free(one_batch)
                raw_logits = _copy_logits()
                current_pos += 1

        return (resolved_tool_id, self.tool_id_to_name.get(resolved_tool_id),
                generated_json, generated_count)

    def create_routing_processor(self, seq_id=0):
        """Creates a standalone NexusRoutingProcessor for use with llm.generate()."""
        return NexusRoutingProcessor(self, seq_id)

    def create_splice_context(self, query, seq_id=0):
        """Creates a NexusSpliceContext for use with `with` statements.

        This is the preferred integration point for RAII VRAM management.

        Args:
            query: User query string.
            seq_id: Sequence ID for this request.

        Returns:
            NexusSpliceContext that guarantees cleanup on exit.
        """
        return NexusSpliceContext(self, query, seq_id)
