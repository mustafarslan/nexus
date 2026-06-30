#include <algorithm>
#include "nexus_orchestrator.hpp"
#include <stdexcept>
#include <iostream>
#include <cmath>

extern "C" {
    float llama_model_rope_freq_scale(const struct llama_model * model);
    uint32_t llama_model_rope_scaling_type(const struct llama_model * model);
    void llama_backend_sync(struct llama_context * ctx);
    bool llama_context_is_metal(struct llama_context * ctx);
    bool llama_context_is_cuda(struct llama_context * ctx);
}

NexusOrchestrator::NexusOrchestrator(llama_context* ctx, NexusSemanticSLB* slb, NexusRadixFSM* fsm, uint32_t base_pos,
                                     float speculative_threshold, float speculative_margin, float auto_route_margin, size_t max_pinned_bytes,
                                     uint32_t max_splice_pos)
    : ctx_(ctx), slb_(slb), fsm_(fsm), base_pos_(base_pos),
      speculative_threshold_(speculative_threshold), speculative_margin_(speculative_margin), auto_route_margin_(auto_route_margin),
      max_splice_pos_(max_splice_pos) {
    block_cache_ = std::shared_ptr<NexusBlockCache>(new NexusBlockCache(max_pinned_bytes));
}

NexusOrchestrator::~NexusOrchestrator() {}

void NexusOrchestrator::register_tool_path(uint32_t tool_id, const std::string& path) {
    tool_paths_[tool_id] = path;
    if (block_cache_) {
        block_cache_->register_tool_path(tool_id, path);
    }
}

void NexusOrchestrator::register_tool_schema_tokens(uint32_t tool_id, const std::vector<int32_t>& tokens) {
    tool_schema_tokens_[tool_id] = tokens;
}

void NexusOrchestrator::release_hazard(llama_seq_id seq_id) {
    uint8_t shard_idx = static_cast<uint8_t>(seq_id & 255);
    auto& shard = active_guards_[shard_idx];
    std::lock_guard<std::mutex> lock(shard.mutex);
    shard.guards.erase(seq_id);
}

void NexusOrchestrator::preload_tool(uint32_t tool_id, const std::string& path) {
    register_tool_path(tool_id, path);
    if (block_cache_) {
        block_cache_->prefetch(path);
    }
}

uint32_t NexusOrchestrator::route_and_splice(const std::vector<int32_t>& user_query_tokens,
                                             const std::vector<float>& query_embedding,
                                             uint32_t n_past,
                                             llama_seq_id seq_id,
                                             const std::vector<uint32_t>& query_lexical_hashes,
                                             const std::vector<int32_t>& prefix_tokens,
                                             uint32_t force_tool_id) {
    const uint32_t Q = static_cast<uint32_t>(user_query_tokens.size());
    if (Q == 0) {
        throw std::runtime_error("Orchestrator: User query is empty.");
    }

    const uint32_t max_ctx = llama_n_ctx(ctx_);
    if (n_past + Q > max_ctx) {
        throw std::length_error("NexusOrchestrator: Context boundary overflow. Required base context length " +
                                std::to_string(n_past + Q) + " exceeds model capacity " + std::to_string(max_ctx));
    }

    if (n_past > max_splice_pos_) {
        // Path B (deep prefix): RoPE Δθ blocks the offline .atb splice, so the
        // orchestrator declines and Python falls back to text prefill.
        // F0 (read-only): probe the position-safe L0 prefix tier for measurement
        // ONLY. We deliberately do NOT seq_cp/consume the match and do NOT alter
        // the text-prefill fallback this phase — whether a deep-path L0 hit saves
        // real work is unproven, and the warm pool is not populated on the deep
        // path (no write-back), so full hits are expected to be rare. try_copy_to
        // (tool-schema warm copy) stays disabled here: it would reintroduce the
        // same Δθ drift the gate exists to prevent.
        splice_guard_fallback_count_.fetch_add(1, std::memory_order_relaxed);
        deep_path_entered_.fetch_add(1, std::memory_order_relaxed);
        const uint32_t deep_best_end =
            prefix_tokens.empty() ? 0u : seq_warm_cache_.probe_prefix(prefix_tokens);
        last_deep_n_past_.store(n_past, std::memory_order_relaxed);
        last_deep_best_end_.store(deep_best_end, std::memory_order_relaxed);
        if (deep_best_end >= n_past) {
            deep_path_l0_hit_.fetch_add(1, std::memory_order_relaxed);
        } else {
            deep_path_l0_miss_.fetch_add(1, std::memory_order_relaxed);
        }
        if (!deep_splice_enabled_) {
            // Legacy behavior: decline so Python performs the text prefill. RoPE Δθ would
            // otherwise drift the offline .atb splice past max_splice_pos_.
            deep_path_text_fallback_.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }
        // Depth-invariant path enabled: fall through and splice, with the depth-adaptive
        // recompute fraction (eff_recompute_pct below) repairing the Δθ drift. No return.
    }

    // Depth-adaptive recompute fraction (never-regress). Ramps linearly from recompute_pct_
    // at the cap to 100% (== full re-prefill, KL=0) by max_splice_pos_*recompute_full_mult_.
    float eff_recompute_pct = recompute_pct_;
    if (n_past > max_splice_pos_) {
        const float lo = static_cast<float>(max_splice_pos_);
        const float hi = lo * recompute_full_mult_;
        const float frac = (hi > lo)
            ? std::clamp((static_cast<float>(n_past) - lo) / (hi - lo), 0.0f, 1.0f)
            : 1.0f;
        eff_recompute_pct = recompute_pct_ + frac * (100.0f - recompute_pct_);
    }

    std::vector<NexusSemanticSLB::SearchResult> matches;
    bool python_force_route = false;
    if (force_tool_id != 0 && tool_paths_.count(force_tool_id)) {
        NexusSemanticSLB::SearchResult forced{};
        forced.tool_id = force_tool_id;
        forced.score = 1.0f;
        matches.push_back(forced);
        fsm_->reset(seq_id);
        fsm_->force_resolved(force_tool_id, seq_id);
        python_force_route = true;
    } else if (!query_lexical_hashes.empty()) {
        matches = slb_->search_hybrid(query_embedding, query_lexical_hashes, 3);
    } else {
        matches = slb_->search(query_embedding, 3);
    }

    float margin = 0.0f;
    if (matches.size() >= 2) {
        margin = matches[0].score - matches[1].score;
    } else if (!matches.empty()) {
        margin = 1e20f;
    }

    if (!matches.empty()) {
        const uint32_t top_tool_id = matches[0].tool_id;
        auto path_it = tool_paths_.find(top_tool_id);
        if (path_it != tool_paths_.end() && !path_it->second.empty()) {
            try {
                auto res = block_cache_->get_or_load(path_it->second);
                if (res.has_value()) {
                    std::shared_ptr<AeonToolBlock> top_block = res.value();
                    top_block->wait_until_resident();
                    const auto* header = top_block->get_header();
                    if (header) {
                        const uint32_t top_schema_len = header->seq_len;
                        const uint32_t required_ctx = n_past + top_schema_len + Q;
                        if (required_ctx > max_ctx) {
                            throw std::length_error("NexusOrchestrator: Context boundary overflow. Required context length " +
                                                    std::to_string(required_ctx) + " exceeds model capacity " + std::to_string(max_ctx));
                        }
                    }
                }
            } catch (const std::length_error&) {
                throw;
            } catch (...) {
            }
        }
    }

    std::shared_ptr<AeonToolBlock> speculative_block = nullptr;
    uint32_t speculative_tool_id = 0;
    uint32_t schema_len = 0;
    bool spec_injected = false;

    if (!matches.empty()) {
        speculative_tool_id = matches[0].tool_id;
        const float score = matches[0].score;
        auto path_it = tool_paths_.find(speculative_tool_id);
        if (path_it != tool_paths_.end() && !path_it->second.empty()) {
            block_cache_->prefetch(path_it->second);

            const bool passes_threshold = (score >= speculative_threshold_) || (margin >= auto_route_margin_);
            const bool passes_margin = (matches.size() >= 2) ? (margin >= speculative_margin_) : true;
            if (passes_threshold && passes_margin) {
                try {
                    auto res = block_cache_->get_or_load(path_it->second);
                    if (res.has_value()) {
                        speculative_block = res.value();
                        speculative_block->wait_until_resident();
                        const AeonToolBlockHeader* header = speculative_block->get_header();
                        if (header) {
                            schema_len = header->seq_len;

                            const struct llama_model* model = llama_get_model(ctx_);
                            const float runtime_scale = llama_model_rope_freq_scale(model);
                            const uint32_t runtime_scaling_type = llama_model_rope_scaling_type(model);
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

    if (spec_injected && schema_len > 0) {
        const uint32_t required_ctx = n_past + schema_len + Q;
        if (required_ctx > max_ctx) {
            throw std::length_error("NexusOrchestrator: Context boundary overflow. Required context length " +
                                    std::to_string(required_ctx) + " exceeds model capacity " + std::to_string(max_ctx));
        }
    }

    const uint32_t fsm_nav_start = n_past + (spec_injected ? schema_len : 0);

    if (spec_injected && speculative_block && schema_len > 0) {
        std::lock_guard<std::mutex> lock(context_mutex_);
        {
            const uint8_t shard_idx = static_cast<uint8_t>(seq_id & 255);
            auto& shard = active_guards_[shard_idx];
            std::lock_guard<std::mutex> lock_shard(shard.mutex);
            shard.guards[seq_id] = block_cache_->acquire_hazard(speculative_block->get_mapped_data());
        }
        NexusKVSplicer::invalidate_sequence(ctx_, seq_id, n_past, -1);
        NexusKVSplicer::inject_tool_page(ctx_, speculative_block, n_past, seq_id);
    }

    {
        std::lock_guard<std::mutex> lock(context_mutex_);
        NexusKVSplicer::invalidate_sequence(ctx_, seq_id, fsm_nav_start, -1);
    }

    bool auto_routed = python_force_route;
    auto t_fsm_start = std::chrono::steady_clock::now();
    if (!python_force_route) {
        fsm_->reset(seq_id);

        if (margin >= auto_route_margin_ && !matches.empty()) {
            fsm_->force_resolved(matches[0].tool_id, seq_id);
            auto_routed = true;
        } else {
            fsm_->begin_routing(seq_id);
        }
    }

    RoutingState fsm_state = fsm_->get_state(seq_id);
    uint32_t current_pos = fsm_nav_start;
    int decode_res = 0;

    if (!auto_routed) {
        const struct llama_model* model = llama_get_model(ctx_);

        const std::string prefix_str = "You are a tool routing agent. Select the single most appropriate tool name from the list below that matches the user's intent.\nYou MUST output ONLY the name of the tool, with no other text, punctuation, explanation, or markdown.\n\nAvailable Tools:\n";
        const std::string middle_str = "\nQuery: ";
        const std::string suffix_str = "\nSelected Tool Name:";

        std::vector<int32_t> prefix_tok(512);
        int32_t n_prefix = llama_tokenize(model, prefix_str.c_str(), prefix_str.size(), prefix_tok.data(), prefix_tok.size(), false, false);
        if (n_prefix < 0) {
            prefix_tok.resize(-n_prefix);
            n_prefix = llama_tokenize(model, prefix_str.c_str(), prefix_str.size(), prefix_tok.data(), prefix_tok.size(), false, false);
        }
        prefix_tok.resize(n_prefix);

        std::vector<int32_t> middle_tok(128);
        int32_t n_middle = llama_tokenize(model, middle_str.c_str(), middle_str.size(), middle_tok.data(), middle_tok.size(), false, false);
        if (n_middle < 0) {
            middle_tok.resize(-n_middle);
            n_middle = llama_tokenize(model, middle_str.c_str(), middle_str.size(), middle_tok.data(), middle_tok.size(), false, false);
        }
        middle_tok.resize(n_middle);

        std::vector<int32_t> suffix_tok(128);
        int32_t n_suffix = llama_tokenize(model, suffix_str.c_str(), suffix_str.size(), suffix_tok.data(), suffix_tok.size(), false, false);
        if (n_suffix < 0) {
            suffix_tok.resize(-n_suffix);
            n_suffix = llama_tokenize(model, suffix_str.c_str(), suffix_str.size(), suffix_tok.data(), suffix_tok.size(), false, false);
        }
        suffix_tok.resize(n_suffix);

        std::vector<int32_t> newline_tok(16);
        int32_t n_newline = llama_tokenize(model, "\n", 1, newline_tok.data(), newline_tok.size(), false, false);
        if (n_newline < 0) {
            newline_tok.resize(-n_newline);
            n_newline = llama_tokenize(model, "\n", 1, newline_tok.data(), newline_tok.size(), false, false);
        }
        newline_tok.resize(n_newline);

        std::vector<int32_t> prefill_tokens;
        prefill_tokens.insert(prefill_tokens.end(), prefix_tok.begin(), prefix_tok.end());

        for (size_t m_idx = 0; m_idx < matches.size(); ++m_idx) {
            if (m_idx > 0) {
                prefill_tokens.insert(prefill_tokens.end(), newline_tok.begin(), newline_tok.end());
            }
            auto tokens = slb_->get_digest_tokens(matches[m_idx].digest_offset, matches[m_idx].digest_len);
            prefill_tokens.insert(prefill_tokens.end(), tokens.begin(), tokens.end());
        }

        prefill_tokens.insert(prefill_tokens.end(), middle_tok.begin(), middle_tok.end());
        prefill_tokens.insert(prefill_tokens.end(), user_query_tokens.begin(), user_query_tokens.end());
        prefill_tokens.insert(prefill_tokens.end(), suffix_tok.begin(), suffix_tok.end());

        const uint32_t T = static_cast<uint32_t>(prefill_tokens.size());
        if (fsm_nav_start + T > max_ctx) {
            throw std::length_error("NexusOrchestrator: Context boundary overflow with FSM digests.");
        }

        llama_batch prefill_batch = llama_batch_init(T, 0, 1);
        prefill_batch.n_tokens = static_cast<int32_t>(T);

        for (uint32_t i = 0; i < T; ++i) {
            prefill_batch.token[i] = prefill_tokens[i];
            prefill_batch.pos[i] = fsm_nav_start + i;
            prefill_batch.n_seq_id[i] = 1;
            prefill_batch.seq_id[i][0] = seq_id;
            prefill_batch.logits[i] = (i == T - 1) ? 1 : 0;
        }

        {
            std::lock_guard<std::mutex> lock(context_mutex_);
            decode_res = llama_decode(ctx_, prefill_batch);
        }
        llama_batch_free(prefill_batch);
        if (decode_res != 0) {
            throw std::runtime_error("Orchestrator: Digest + query prefill decode failed.");
        }

        current_pos = fsm_nav_start + T;
        int32_t last_batch_idx = static_cast<int32_t>(T) - 1;

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
    }
    const auto t_fsm_end = std::chrono::steady_clock::now();

    if (fsm_state == RoutingState::LEAF_REACHED) {
        const uint32_t resolved_tool_id = fsm_->get_resolved_tool_id(seq_id);

        auto path_it = tool_paths_.find(resolved_tool_id);
        if (path_it == tool_paths_.end()) {
            std::cerr << "Orchestrator Error: No registered ATB file path for resolved tool ID: " << resolved_tool_id << "\n";
            return 0;
        }
        const std::string& atb_path = path_it->second;

        if (resolved_tool_id == speculative_tool_id && spec_injected && speculative_block) {
            telemetry_block_.speculative_hit_count.fetch_add(1, std::memory_order_relaxed);
            {
                std::lock_guard<std::mutex> lock(context_mutex_);
                NexusKVSplicer::invalidate_sequence(ctx_, seq_id, fsm_nav_start, -1);

                const auto t_wait_start = std::chrono::steady_clock::now();
                llama_backend_sync(ctx_);
                const auto t_wait_end = std::chrono::steady_clock::now();

                const auto wait_us = std::chrono::duration_cast<std::chrono::microseconds>(t_wait_end - t_wait_start).count();
                telemetry_block_.exposed_splice_latency_us.fetch_add(wait_us, std::memory_order_relaxed);

                const auto fsm_us = std::chrono::duration_cast<std::chrono::microseconds>(t_fsm_end - t_fsm_start).count();
                telemetry_block_.fsm_hidden_latency_us.fetch_add(fsm_us, std::memory_order_relaxed);

                if (llama_context_is_metal(ctx_)) {
                    std::cout << "[Topology: UMA Zero-Copy] VRAM Splice Latency synchronized.\n";
                } else if (llama_context_is_cuda(ctx_)) {
                    std::cout << "[Topology: NUMA/PCIe Gen4] Asynchronous H2D DMA Splice Latency synchronized.\n";
                }
            }
        } else {
            if (spec_injected) {
                telemetry_block_.speculative_miss_count.fetch_add(1, std::memory_order_relaxed);
            }
            release_hazard(seq_id);

            {
                std::lock_guard<std::mutex> lock(context_mutex_);
                NexusKVSplicer::invalidate_sequence(ctx_, seq_id, n_past, -1);
            }

            const auto t_wait_start = std::chrono::steady_clock::now();
            bool used_l0_warm = false;
            {
                std::lock_guard<std::mutex> lock(context_mutex_);
                if (!prefix_tokens.empty()) {
                    used_l0_warm = seq_warm_cache_.try_copy_prefix(ctx_, prefix_tokens, seq_id) >= n_past;
                }
                if (!used_l0_warm) {
                    used_l0_warm = seq_warm_cache_.try_copy_to(ctx_, resolved_tool_id, n_past, seq_id);
                }
            }

            if (used_l0_warm) {
                const auto t_wait_end = std::chrono::steady_clock::now();
                const auto wait_us = std::chrono::duration_cast<std::chrono::microseconds>(t_wait_end - t_wait_start).count();
                telemetry_block_.exposed_splice_latency_us.fetch_add(wait_us, std::memory_order_relaxed);
            } else {
                auto res = block_cache_->get_or_load(atb_path);
                if (!res.has_value()) {
                    if (res.error() == NexusErrorCode::RESOURCE_EXHAUSTED) {
                        throw ResourceExhaustedException("NUMA Node Capacity Fully Saturated");
                    }
                    throw std::runtime_error("Internal Cache Error or Failed to load block: " + atb_path);
                }
                std::shared_ptr<AeonToolBlock> correct_block = res.value();
                correct_block->wait_until_resident();
                const AeonToolBlockHeader* header = correct_block->get_header();
                if (!header) {
                    std::cerr << "Orchestrator Error: Failed to parse header: " << atb_path << "\n";
                    return 0;
                }
                schema_len = header->seq_len;

                const struct llama_model* model = llama_get_model(ctx_);
                const float runtime_scale = llama_model_rope_freq_scale(model);
                const uint32_t runtime_scaling_type = llama_model_rope_scaling_type(model);
                if (std::abs(header->rope_freq_scale - runtime_scale) > 1e-5f ||
                    header->rope_scaling_type != runtime_scaling_type) {
                    throw std::runtime_error("NexusOrchestrator: RoPE scaling parameters mismatch for correct block.");
                }

                const uint32_t required_ctx = n_past + schema_len + Q;
                const uint32_t max_ctx_final = llama_n_ctx(ctx_);
                if (required_ctx > max_ctx_final) {
                    throw std::length_error("NexusOrchestrator: Context boundary overflow. Required context length " +
                                            std::to_string(required_ctx) + " exceeds model capacity " + std::to_string(max_ctx_final));
                }

                {
                    std::lock_guard<std::mutex> lock(context_mutex_);
                    {
                        const uint8_t shard_idx = static_cast<uint8_t>(seq_id & 255);
                        auto& shard = active_guards_[shard_idx];
                        std::lock_guard<std::mutex> lock_shard(shard.mutex);
                        shard.guards[seq_id] = block_cache_->acquire_hazard(correct_block->get_mapped_data());
                    }

                    NexusKVSplicer::inject_tool_page(ctx_, correct_block, n_past, seq_id);
                    llama_backend_sync(ctx_);

                    if (!prefix_tokens.empty()) {
                        seq_warm_cache_.update_from_seq(ctx_, prefix_tokens, n_past + schema_len, seq_id, resolved_tool_id);
                    } else {
                        seq_warm_cache_.update_warm_from_seq(ctx_, resolved_tool_id, n_past, schema_len, seq_id);
                    }
                    const auto t_wait_end = std::chrono::steady_clock::now();

                    const auto wait_us = std::chrono::duration_cast<std::chrono::microseconds>(t_wait_end - t_wait_start).count();
                    telemetry_block_.exposed_splice_latency_us.fetch_add(wait_us, std::memory_order_relaxed);

                    if (llama_context_is_metal(ctx_)) {
                        std::cout << "[Topology: UMA Zero-Copy] VRAM Splice Latency synchronized.\n";
                    } else if (llama_context_is_cuda(ctx_)) {
                        std::cout << "[Topology: NUMA/PCIe Gen4] Asynchronous H2D DMA Splice Latency synchronized.\n";
                    }
                }
            }
        }

        auto schema_tok_it = tool_schema_tokens_.find(resolved_tool_id);
        const bool fused_recompute = schema_tok_it != tool_schema_tokens_.end() && eff_recompute_pct > 0.0f;

        uint32_t suffix_start = schema_len;
        if (fused_recompute) {
            const auto& schema_tokens = schema_tok_it->second;
            if (schema_tokens.size() != schema_len) {
                schema_len = static_cast<uint32_t>(schema_tokens.size());
            }
            const uint32_t n_sel = std::max(1u, static_cast<uint32_t>(
                std::ceil(static_cast<double>(schema_len) * static_cast<double>(eff_recompute_pct) / 100.0)));
            suffix_start = (schema_len > n_sel) ? (schema_len - n_sel) : 0u;
            if (suffix_start < schema_len) {
                std::lock_guard<std::mutex> lock(context_mutex_);
                NexusKVSplicer::invalidate_sequence(ctx_, seq_id, n_past + suffix_start, n_past + schema_len);
            }
        }

        const uint32_t n_suffix = fused_recompute ? (schema_len - suffix_start) : 0u;
        const uint32_t batch_n = n_suffix + Q;
        llama_batch query_batch = llama_batch_init(static_cast<int32_t>(batch_n), 0, 1);
        query_batch.n_tokens = static_cast<int32_t>(batch_n);
        uint32_t bi = 0;
        if (fused_recompute) {
            const auto& schema_tokens = schema_tok_it->second;
            for (uint32_t i = suffix_start; i < schema_len; ++i) {
                query_batch.token[bi] = schema_tokens[i];
                query_batch.pos[bi] = n_past + i;
                query_batch.n_seq_id[bi] = 1;
                query_batch.seq_id[bi][0] = seq_id;
                query_batch.logits[bi] = 0;
                ++bi;
            }
        }
        for (uint32_t i = 0; i < Q; ++i) {
            query_batch.token[bi] = user_query_tokens[i];
            query_batch.pos[bi] = n_past + schema_len + i;
            query_batch.n_seq_id[bi] = 1;
            query_batch.seq_id[bi][0] = seq_id;
            query_batch.logits[bi] = (i == Q - 1) ? 1 : 0;
            ++bi;
        }

        {
            std::lock_guard<std::mutex> lock(context_mutex_);
            decode_res = llama_decode(ctx_, query_batch);
            llama_backend_sync(ctx_);
        }
        llama_batch_free(query_batch);
        if (decode_res != 0) {
            throw std::runtime_error("Orchestrator: Failed to decode user query during re-prefill.");
        }

        return resolved_tool_id;
    }

    return 0;
}
