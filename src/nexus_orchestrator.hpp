#pragma once
#include <vector>
#include <unordered_map>
#include <memory>
#include <string>
#include <cstdint>
#include <mutex>
#include "llama.h"
#include "nexus_slb.hpp"
#include "nexus_fsm.hpp"
#include "nexus_kv_splicer.hpp"
#include "nexus_block_cache.hpp"

// Nexus is a Memory Management Unit (MMU), NOT a grammar engine.
// JSON constrained decoding is delegated to the native host server
// (e.g., llama_cpp.LlamaGrammar). Nexus only performs:
//   1. Semantic SLB routing
//   2. Radix FSM logit masking
//   3. KV cache splicing/unsplicing

class NexusOrchestrator {
private:
    llama_context* ctx_;
    NexusSemanticSLB* slb_;
    NexusRadixFSM* fsm_;
    uint32_t base_pos_; // kept for backward compatibility, unused
    std::unordered_map<uint32_t, std::string> tool_paths_;
    std::shared_ptr<NexusBlockCache> block_cache_;
    float speculative_threshold_;
    float speculative_margin_;

    // Array of active hazard guards per sequence ID (pre-allocated to avoid mutexes)
    std::vector<HazardGuard> active_guards_;

    // Shared llama_context mutex. Required ONLY for llama_decode and
    // llama_get_logits_ith calls which mutate the shared context state.
    std::mutex context_mutex_;

public:
    NexusOrchestrator(llama_context* ctx, NexusSemanticSLB* slb, NexusRadixFSM* fsm, uint32_t base_pos,
                      float speculative_threshold = 0.88f, float speculative_margin = 0.05f,
                      size_t max_pinned_bytes = 16ULL * 1024 * 1024 * 1024);
    ~NexusOrchestrator();

    // Disable copy
    NexusOrchestrator(const NexusOrchestrator&) = delete;
    NexusOrchestrator& operator=(const NexusOrchestrator&) = delete;

    void register_tool_path(uint32_t tool_id, const std::string& path);
    void preload_tool(uint32_t tool_id, const std::string& path);

    // Executes Stage 1 (Routing) and Stage 2 (Execution)
    // Returns the final tool_id selected by the FSM
    uint32_t route_and_splice(const std::vector<int32_t>& user_query_tokens, const std::vector<float>& query_embedding, uint32_t n_past, llama_seq_id seq_id = 0);

    std::shared_ptr<NexusBlockCache> get_cache() const { return block_cache_; }
    void release_hazard(llama_seq_id seq_id);
    llama_context* get_context() const { return ctx_; }
};
