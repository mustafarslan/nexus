#include "nexus_orchestrator.hpp"
#include <stdexcept>
#include <iostream>
#include <cmath>

extern "C" {
    float llama_model_rope_freq_scale(const struct llama_model * model);
    uint32_t llama_model_rope_scaling_type(const struct llama_model * model);
    struct ggml_tensor * llama_kv_cache_get_k(struct llama_context * ctx, int il);
    void llama_backend_sync(struct llama_context * ctx);
    bool llama_context_is_metal(struct llama_context * ctx);
    bool llama_context_is_cuda(struct llama_context * ctx);
}

NexusOrchestrator::NexusOrchestrator(llama_context* ctx, NexusSemanticSLB* slb, NexusRadixFSM* fsm, uint32_t base_pos,
                                     float speculative_threshold, float speculative_margin, size_t max_pinned_bytes)
    : ctx_(ctx), slb_(slb), fsm_(fsm), base_pos_(base_pos),
      speculative_threshold_(speculative_threshold), speculative_margin_(speculative_margin),
      active_guards_(4096) {
    block_cache_ = std::shared_ptr<NexusBlockCache>(new NexusBlockCache(max_pinned_bytes));
}

NexusOrchestrator::~NexusOrchestrator() {}

void NexusOrchestrator::register_tool_path(uint32_t tool_id, const std::string& path) {
    tool_paths_[tool_id] = path;
    if (block_cache_) {
        block_cache_->register_tool_path(tool_id, path);
    }
}

void NexusOrchestrator::release_hazard(llama_seq_id seq_id) {
    if (seq_id >= 0 && static_cast<size_t>(seq_id) < active_guards_.size()) {
        active_guards_[seq_id] = HazardGuard{};
    }
}

void NexusOrchestrator::preload_tool(uint32_t tool_id, const std::string& path) {
    register_tool_path(tool_id, path);
    if (block_cache_) {
        block_cache_->prefetch(path);
    }
}

uint32_t NexusOrchestrator::route_and_splice(const std::vector<int32_t>& user_query_tokens, const std::vector<float>& query_embedding, uint32_t n_past, llama_seq_id seq_id) {
    uint32_t Q = user_query_tokens.size();
    if (Q == 0) {
        throw std::runtime_error("Orchestrator: User query is empty.");
    }

    uint32_t max_ctx = llama_n_ctx(ctx_);
    if (n_past + Q > max_ctx) {
        throw std::length_error("NexusOrchestrator: Context boundary overflow. Required base context length " +
                                std::to_string(n_past + Q) + " exceeds model capacity " + std::to_string(max_ctx));
    }

    // 1. Perform L1 SLB SIMD scan to match Top-K semantic scents (Lock-free read-only operation)
    std::vector<NexusSemanticSLB::SearchResult> matches = slb_->search(query_embedding, 3);

    // Proactive context boundary check using the top matched candidate's schema length.
    // This must run before routing because even if speculation is not injected, the final resolved
    // tool will eventually be loaded and spliced at n_past, which must not overflow max_ctx.
    if (!matches.empty()) {
        uint32_t top_tool_id = matches[0].tool_id;
        auto path_it = tool_paths_.find(top_tool_id);
        if (path_it != tool_paths_.end() && !path_it->second.empty()) {
            try {
                std::shared_ptr<AeonToolBlock> top_block = block_cache_->get_or_load(path_it->second);
                if (top_block) {
                    top_block->wait_until_resident();
                    const auto* header = top_block->get_header();
                    if (header) {
                        uint32_t top_schema_len = header->seq_len;
                        uint32_t required_ctx = n_past + top_schema_len + Q;
                        if (required_ctx > max_ctx) {
                            throw std::length_error("NexusOrchestrator: Context boundary overflow. Required context length " +
                                                    std::to_string(required_ctx) + " exceeds model capacity " + std::to_string(max_ctx));
                        }
                    }
                }
            } catch (const std::length_error&) {
                throw;
            } catch (...) {
                // Ignore other load errors at this stage
            }
        }
    }

    // 2. Pre-emptively load, prefetch and speculative inject the top matched candidate
    std::shared_ptr<AeonToolBlock> speculative_block = nullptr;
    uint32_t speculative_tool_id = 0;
    uint32_t schema_len = 0;
    bool spec_injected = false;

    if (!matches.empty()) {
        speculative_tool_id = matches[0].tool_id;
        float score = matches[0].score;
        auto path_it = tool_paths_.find(speculative_tool_id);
        if (path_it != tool_paths_.end() && !path_it->second.empty()) {
            // Issue only an OS memory pre-fetch hint (posix_madvise)
            block_cache_->prefetch(path_it->second);

            bool passes_threshold = (score >= speculative_threshold_);
            bool passes_margin = (matches.size() >= 2) ? ((matches[0].score - matches[1].score) >= speculative_margin_) : true;
            if (passes_threshold && passes_margin) {
                try {
                    // Load/fetch block (maps it & moves to front of LRU)
                    speculative_block = block_cache_->get_or_load(path_it->second);
                    if (speculative_block) {
                        speculative_block->wait_until_resident();
                        const AeonToolBlockHeader* header = speculative_block->get_header();
                        if (header) {
                            schema_len = header->seq_len;

                            // Assert YaRN/RoPE alignment
                            const struct llama_model* model = llama_get_model(ctx_);
                            float runtime_scale = llama_model_rope_freq_scale(model);
                            uint32_t runtime_scaling_type = llama_model_rope_scaling_type(model);
                            if (std::abs(header->rope_freq_scale - runtime_scale) > 1e-5f ||
                                header->rope_scaling_type != runtime_scaling_type) {
                                throw std::runtime_error("NexusOrchestrator: RoPE scaling parameters mismatch.");
                            }
                            spec_injected = true;
                        }
                    }
                } catch (...) {
                    speculative_block = nullptr;
                    schema_len = 0;
                    speculative_tool_id = 0;
                    spec_injected = false;
                }
            }
        }
    }

    // Verify context boundary with the speculative/predicted candidate
    if (spec_injected && schema_len > 0) {
        uint32_t required_ctx = n_past + schema_len + Q;
        if (required_ctx > max_ctx) {
            throw std::length_error("NexusOrchestrator: Context boundary overflow. Required context length " +
                                    std::to_string(required_ctx) + " exceeds model capacity " + std::to_string(max_ctx));
        }
    }

    // Determine FSM navigation starting position.
    // If we speculative-injected a candidate, FSM navigation starts downstream of the candidate page in the KV cache
    // to prevent physical cell collision and overlapping logical position conflicts.
    uint32_t fsm_nav_start = n_past + (spec_injected ? schema_len : 0);

    // Speculative inject top SLB match into the KV cache asynchronously (launches DMA in background)
    if (spec_injected && speculative_block && schema_len > 0) {
        std::lock_guard<std::mutex> lock(context_mutex_);
        if (seq_id >= 0 && static_cast<size_t>(seq_id) < active_guards_.size()) {
            active_guards_[seq_id] = block_cache_->acquire_hazard(speculative_block->get_mapped_data());
        }
        // Invalidate region starting at n_past (where speculative block will go)
        NexusKVSplicer::invalidate_sequence(ctx_, seq_id, n_past, -1);
        // Inject block asynchronously (copies K/V + launches async H2D DMA transfers)
        NexusKVSplicer::inject_tool_page(ctx_, speculative_block, n_past, seq_id);
    }

    // 3. Clear query and scent region from fsm_nav_start onwards to prepare for FSM routing
    {
        std::lock_guard<std::mutex> lock(context_mutex_);
        NexusKVSplicer::invalidate_sequence(ctx_, seq_id, fsm_nav_start, -1);
    }

    // Flatten all scent tokens
    std::vector<int32_t> scent_tokens;
    for (const auto& match : matches) {
        for (size_t i = 0; i < SCENT_TOKENS; ++i) {
            scent_tokens.push_back(match.scent_tokens[i]);
        }
    }
    uint32_t S = scent_tokens.size();

    // Prefill scents + user query
    uint32_t T = S + Q;
    llama_batch prefill_batch = llama_batch_init(T, 0, 1);
    prefill_batch.n_tokens = T;

    // Inject scents
    for (uint32_t i = 0; i < S; ++i) {
        prefill_batch.token[i] = scent_tokens[i];
        prefill_batch.pos[i] = fsm_nav_start + i;
        prefill_batch.n_seq_id[i] = 1;
        prefill_batch.seq_id[i][0] = seq_id;
        prefill_batch.logits[i] = 0;
    }

    // Inject user query
    for (uint32_t i = 0; i < Q; ++i) {
        prefill_batch.token[S + i] = user_query_tokens[i];
        prefill_batch.pos[S + i] = fsm_nav_start + S + i;
        prefill_batch.n_seq_id[S + i] = 1;
        prefill_batch.seq_id[S + i][0] = seq_id;
        prefill_batch.logits[S + i] = (i == Q - 1) ? 1 : 0;
    }

    int decode_res = 0;
    {
        std::lock_guard<std::mutex> lock(context_mutex_);
        decode_res = llama_decode(ctx_, prefill_batch);
    }
    llama_batch_free(prefill_batch);
    if (decode_res != 0) {
        throw std::runtime_error("Orchestrator: Scent + query prefill decode failed.");
    }

    // 4. Decode loop constrained by Radix FSM
    fsm_->reset(seq_id);
    fsm_->begin_routing(seq_id);

    RoutingState fsm_state = fsm_->get_state(seq_id);
    uint32_t current_pos = fsm_nav_start + T;
    int32_t last_batch_idx = T - 1;

    while (fsm_state == RoutingState::NAVIGATING) {
        float* logits = nullptr;
        int32_t n_vocab = 0;
        {
            std::lock_guard<std::mutex> lock(context_mutex_);
            logits = llama_get_logits_ith(ctx_, last_batch_idx);
            n_vocab = llama_n_vocab(llama_get_model(ctx_));
        }

        std::span<float> logits_span(logits, n_vocab);
        fsm_->apply_logit_mask(logits_span, seq_id);

        int32_t sampled_token = -1;
        float max_logit = -INFINITY;
        for (int32_t v = 0; v < n_vocab; ++v) {
            if (logits_span[v] > max_logit) {
                max_logit = logits_span[v];
                sampled_token = v;
            }
        }

        if (sampled_token == -1 || max_logit == -INFINITY) {
            std::cerr << "Orchestrator Routing Error: All logit options masked out by FSM.\n";
            return 0;
        }

        fsm_state = fsm_->advance(sampled_token, seq_id);

        if (fsm_state != RoutingState::NAVIGATING) {
            break;
        }

        llama_batch batch_one = llama_batch_init(1, 0, 1);
        batch_one.n_tokens = 1;
        batch_one.token[0] = sampled_token;
        batch_one.pos[0] = current_pos;
        batch_one.n_seq_id[0] = 1;
        batch_one.seq_id[0][0] = seq_id;
        batch_one.logits[0] = 1;

        {
            std::lock_guard<std::mutex> lock(context_mutex_);
            decode_res = llama_decode(ctx_, batch_one);
        }
        llama_batch_free(batch_one);
        if (decode_res != 0) {
            throw std::runtime_error("Orchestrator: Failed to decode routed token.");
        }

        current_pos++;
        last_batch_idx = 0;
    }

    // --- STAGE 2: Execution Context ---
    if (fsm_state == RoutingState::LEAF_REACHED) {
        uint32_t resolved_tool_id = fsm_->get_resolved_tool_id(seq_id);
        
        auto path_it = tool_paths_.find(resolved_tool_id);
        if (path_it == tool_paths_.end()) {
            std::cerr << "Orchestrator Error: No registered ATB file path for resolved tool ID: " << resolved_tool_id << "\n";
            return 0;
        }
        std::string atb_path = path_it->second;

        // Verify if speculative injection was correct
        if (resolved_tool_id == speculative_tool_id && spec_injected && speculative_block) {
            // Speculation succeeded!
            {
                std::lock_guard<std::mutex> lock(context_mutex_);
                // Invalidate FSM navigation tokens (scents + intermediate routed tokens)
                NexusKVSplicer::invalidate_sequence(ctx_, seq_id, fsm_nav_start, -1);
                // Synchronize GPU backend to ensure speculative async DMA transfers are fully complete
                llama_backend_sync(ctx_);
                if (llama_context_is_metal(ctx_)) {
                    std::cout << "[Topology: UMA Zero-Copy] VRAM Splice Latency synchronized.\n";
                } else if (llama_context_is_cuda(ctx_)) {
                    std::cout << "[Topology: NUMA/PCIe Gen4] Asynchronous H2D DMA Splice Latency synchronized.\n";
                }
            }
        } else {
            // Speculation failed or was not performed.
            release_hazard(seq_id);
 
            // Invalidate everything from n_past onwards
            {
                std::lock_guard<std::mutex> lock(context_mutex_);
                NexusKVSplicer::invalidate_sequence(ctx_, seq_id, n_past, -1);
            }
 
            // Load and inject the correct tool block
            std::shared_ptr<AeonToolBlock> correct_block = block_cache_->get_or_load(atb_path);
            if (!correct_block) {
                std::cerr << "Orchestrator Error: Failed to load block: " << atb_path << "\n";
                return 0;
            }
            correct_block->wait_until_resident();
            const AeonToolBlockHeader* header = correct_block->get_header();
            if (!header) {
                std::cerr << "Orchestrator Error: Failed to parse header: " << atb_path << "\n";
                return 0;
            }
            schema_len = header->seq_len;
 
            // Assert YaRN/RoPE alignment for the correct block
            const struct llama_model* model = llama_get_model(ctx_);
            float runtime_scale = llama_model_rope_freq_scale(model);
            uint32_t runtime_scaling_type = llama_model_rope_scaling_type(model);
            if (std::abs(header->rope_freq_scale - runtime_scale) > 1e-5f ||
                header->rope_scaling_type != runtime_scaling_type) {
                throw std::runtime_error("NexusOrchestrator: RoPE scaling parameters mismatch for correct block.");
            }
 
            // Context boundary assertion
            uint32_t required_ctx = n_past + schema_len + Q;
            uint32_t max_ctx_final = llama_n_ctx(ctx_);
            if (required_ctx > max_ctx_final) {
                throw std::length_error("NexusOrchestrator: Context boundary overflow. Required context length " +
                                        std::to_string(required_ctx) + " exceeds model capacity " + std::to_string(max_ctx_final));
            }
 
            {
                std::lock_guard<std::mutex> lock(context_mutex_);
                if (seq_id >= 0 && static_cast<size_t>(seq_id) < active_guards_.size()) {
                    active_guards_[seq_id] = block_cache_->acquire_hazard(correct_block->get_mapped_data());
                }
 
                // Inject the correct block (non-blocking launch)
                NexusKVSplicer::inject_tool_page(ctx_, correct_block, n_past, seq_id);
                // Synchronize GPU backend to ensure DMA transfers are fully complete
                llama_backend_sync(ctx_);
                if (llama_context_is_metal(ctx_)) {
                    std::cout << "[Topology: UMA Zero-Copy] VRAM Splice Latency synchronized.\n";
                } else if (llama_context_is_cuda(ctx_)) {
                    std::cout << "[Topology: NUMA/PCIe Gen4] Asynchronous H2D DMA Splice Latency synchronized.\n";
                }
            }
        }

        // Prefill the original user query downstream of the tool schema
        llama_batch query_batch = llama_batch_init(Q, 0, 1);
        query_batch.n_tokens = Q;
        for (uint32_t i = 0; i < Q; ++i) {
            query_batch.token[i] = user_query_tokens[i];
            query_batch.pos[i] = n_past + schema_len + i;
            query_batch.n_seq_id[i] = 1;
            query_batch.seq_id[i][0] = seq_id;
            query_batch.logits[i] = (i == Q - 1) ? 1 : 0;
        }

        {
            std::lock_guard<std::mutex> lock(context_mutex_);
            decode_res = llama_decode(ctx_, query_batch);
        }
        llama_batch_free(query_batch);
        if (decode_res != 0) {
            throw std::runtime_error("Orchestrator: Failed to decode user query during re-prefill.");
        }

        return resolved_tool_id;
    }

    return 0; // Routing failed to hit leaf node
}
