import ctypes
import os
import sys
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
    def __init__(self, llm, embedding_llm=None, dim=384, base_pos=256):
        """
        Integrates GGUF inference (llama_cpp.Llama) with Project Nexus C++ extensions.

        Args:
            llm: An instance of llama_cpp.Llama.
            embedding_llm: Optional separate embedding llama_cpp.Llama instance.
            dim: Dimension of embeddings (e.g. 768 for Nomic Embed, 384 for MiniLM).
            base_pos: Starting position offset in RoPE context for tools injection.
        """
        self.llm = llm
        self.embedding_llm = embedding_llm
        self.dim = embedding_llm.n_embd() if embedding_llm is not None else dim
        self.base_pos = base_pos
        self.tool_id_to_name = {}
        self.tool_schema_lengths = {}
        self.tool_grammars = {}
        self.last_query_len = 0
        self.last_splice_pos = 0
        self.last_schema_len = 0

        # 1. Safely extract the raw llama_context C pointer address
        self.ctx_addr = self._extract_context_pointer(llm)
        if not self.ctx_addr:
            raise ValueError("Extracted llama_context pointer address is null (0).")

        # 2. Instantiate FFI C++ components
        self.slb = nexus_fsm_ext.NexusSemanticSLB(self.dim)
        self.fsm = nexus_fsm_ext.NexusRadixFSM()
        self.orchestrator = nexus_fsm_ext.NexusOrchestrator(
            self.ctx_addr, self.slb, self.fsm, self.base_pos
        )

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

    def register_tool(self, tool_id, name, embedding, scent_tokens, atb_path, schema=None):
        """
        Registers a tool in the Radix FSM, Semantic SLB, and Orchestrator.

        Args:
            tool_id: Unique uint32 ID for the tool.
            name: String name of the tool.
            embedding: Vector embedding (numpy array or list) of shape (dim,).
            scent_tokens: List/tuple of exactly 5 scent token IDs.
            atb_path: Filepath to compiled .atb block for the tool.
            schema: Optional JSON schema (dict or JSON string) for the tool arguments.
        """
        if len(embedding) != self.dim:
            raise ValueError(f"Embedding size must match agent dim {self.dim}, got {len(embedding)}")
        if len(scent_tokens) != 5:
            raise ValueError(f"Scent tokens must have exactly 5 elements, got {len(scent_tokens)}")

        # Tokenize the tool name (this forms the Radix FSM route)
        route_tokens = self.llm.tokenize(name.encode("utf-8"), add_bos=False, special=False)
        route_tokens_list = [int(t) for t in route_tokens]

        # FFI registers
        self.fsm.add_route(tool_id, route_tokens_list)
        
        vec_np = np.asarray(embedding, dtype=np.float32)
        scent_list = [int(t) for t in scent_tokens]
        self.slb.register_tool(tool_id, vec_np, scent_list)
        
        self.orchestrator.preload_tool(tool_id, atb_path)
        self.tool_id_to_name[tool_id] = name

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

    def route_and_splice(self, query, seq_id=0):
        """
        Invokes C++ route_and_splice.

        Args:
            query: String query from user.
            seq_id: Sequence ID for continuous batching.

        Returns:
            Tuple of (resolved_tool_id, tool_name)
        """
        query_tokens = self.llm.tokenize(query.encode("utf-8"), add_bos=False, special=False)
        query_tokens_list = [int(t) for t in query_tokens]
        Q = len(query_tokens_list)
        self.last_query_len = Q

        # Use dedicated embedding model if present, otherwise fallback to generative
        emb_model = self.embedding_llm if self.embedding_llm is not None else self.llm
        query_embedding_raw = emb_model.embed(query)
        query_embedding = query_embedding_raw[0] if (len(query_embedding_raw) > 0 and isinstance(query_embedding_raw[0], list)) else query_embedding_raw
        query_embedding_list = [float(x) for x in query_embedding]

        n_past = self.llm.n_tokens
        resolved_id = self.orchestrator.route_and_splice(query_tokens_list, query_embedding_list, n_past, seq_id)
        
        self.last_splice_pos = n_past
        self.last_schema_len = self.tool_schema_lengths.get(resolved_id, 0) if resolved_id != 0 else 0
        
        return resolved_id, self.tool_id_to_name.get(resolved_id, None)

    def generate_with_tool(self, query, seq_id=0, max_tokens=512):
        """
        Executes routing, tool splicing, grammar-constrained generation, and unsplicing.

        Uses NexusSpliceContext (RAII Fix 4) to guarantee VRAM cleanup on any exit,
        and delegates JSON grammar constraint to the native host LlamaGrammar.

        Returns a tuple of (resolved_tool_id, tool_name, generated_json_arguments_str).
        """
        import llama_cpp

        with NexusSpliceContext(self, query, seq_id) as ctx:
            if ctx.resolved_tool_id == 0:
                return 0, None, ""

            # Delegate JSON constrained decoding to the native host grammar engine.
            # Nexus is an MMU — it does NOT parse grammars in C++.
            grammar_str = self.tool_grammars.get(ctx.resolved_tool_id, "")
            native_grammar = None
            if grammar_str:
                native_grammar = llama_cpp.LlamaGrammar.from_string(grammar_str)

            n_past = ctx._n_past
            schema_len = ctx._schema_len
            Q = ctx._query_len
            current_pos = n_past + schema_len + Q

            generated_json = ""
            generated_tokens_count = 0
            model = self.llm

            while generated_tokens_count < max_tokens:
                # Sample using the native library with grammar delegation
                token_id = llama_cpp.llama_sample_token_greedy(
                    self.ctx_addr,
                    None  # candidates built internally
                )

                # Fallback: manual sampling from logits
                logits_ptr = llama_cpp.llama_get_logits(self.ctx_addr)
                n_vocab = llama_cpp.llama_n_vocab(model.model)

                # Build candidates and apply grammar natively
                candidates = (llama_cpp.llama_token_data * n_vocab)()
                for v in range(n_vocab):
                    candidates[v].id = v
                    candidates[v].logit = logits_ptr[v]
                    candidates[v].p = 0.0

                candidates_p = llama_cpp.llama_token_data_array(
                    candidates, n_vocab, False
                )

                if native_grammar:
                    llama_cpp.llama_sample_grammar(
                        self.ctx_addr, ctypes.byref(candidates_p),
                        native_grammar.grammar
                    )

                token_id = llama_cpp.llama_sample_token_greedy(
                    self.ctx_addr, ctypes.byref(candidates_p)
                )

                if native_grammar:
                    llama_cpp.llama_grammar_accept_token(
                        self.ctx_addr, native_grammar.grammar, token_id
                    )

                # Check for EOG token
                if llama_cpp.llama_token_is_eog(model.model, token_id):
                    break

                # Detokenize and append
                piece = model.detokenize([token_id]).decode("utf-8", errors="ignore")
                generated_json += piece
                generated_tokens_count += 1
                ctx.local_generated_count += 1

                # Decode the single generated token
                one_batch = llama_cpp.llama_batch_init(1, 0, 1)
                one_batch.n_tokens = 1
                one_batch.token[0] = token_id
                one_batch.pos[0] = current_pos
                one_batch.n_seq_id[0] = 1
                one_batch.seq_id[0][0] = seq_id
                one_batch.logits[0] = 1

                res = llama_cpp.llama_decode(self.ctx_addr, one_batch)
                llama_cpp.llama_batch_free(one_batch)
                if res != 0:
                    raise RuntimeError(f"Failed to decode token {token_id} at position {current_pos}")

                current_pos += 1

        # Sync Python-side state after unsplice
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

        return ctx.resolved_tool_id, ctx.tool_name, generated_json

    def create_routing_processor(self, seq_id=0):
        """Creates a standalone NexusRoutingProcessor for use with llm.generate().

        This is the preferred integration point for continuous batching servers.
        The server calls:
            llm.generate(..., logits_processor=LogitsProcessorList([proc]),
                          grammar=LlamaGrammar.from_string(gbnf))

        Args:
            seq_id: Sequence ID for this request.

        Returns:
            NexusRoutingProcessor instance conforming to llama_cpp.LogitsProcessor.
        """
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
