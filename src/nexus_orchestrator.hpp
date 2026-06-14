#pragma once
#include <vector>
#include <unordered_map>
#include <array>
#include <memory>
#include <string>
#include <cstdint>
#include <mutex>
#include "llama.h"
#include "nexus_slb.hpp"
#include "nexus_fsm.hpp"
#include "nexus_kv_splicer.hpp"
#include "nexus_block_cache.hpp"
#include "nexus_seq_warm_cache.hpp"

#include "nexus_os_compat.hpp"

#include <chrono>
#include <atomic>

struct PyTelemetryBlock {
    uint64_t speculative_hit_count;
    uint64_t speculative_miss_count;
    uint64_t fsm_hidden_latency_us;
    uint64_t exposed_splice_latency_us;
};

struct alignas(128) TelemetryBlock {
    std::atomic<uint64_t> speculative_hit_count{0};
    std::atomic<uint64_t> speculative_miss_count{0};
    std::atomic<uint64_t> fsm_hidden_latency_us{0};
    std::atomic<uint64_t> exposed_splice_latency_us{0};
};

class NexusOrchestrator {
private:
    llama_context* ctx_;
    NexusSemanticSLB* slb_;
    NexusRadixFSM* fsm_;
    uint32_t base_pos_;
    std::unordered_map<uint32_t, std::string> tool_paths_;
    std::unordered_map<uint32_t, std::vector<int32_t>> tool_schema_tokens_;
    float recompute_pct_{5.0f};
    std::shared_ptr<NexusBlockCache> block_cache_;
    float speculative_threshold_;
    float speculative_margin_;
    float auto_route_margin_;
    uint32_t max_splice_pos_;
    std::atomic<uint64_t> splice_guard_fallback_count_{0};

    struct alignas(NEXUS_CACHE_LINE) HazardShard {
        std::mutex mutex;
        std::unordered_map<llama_seq_id, HazardGuard> guards;
    };
    std::array<HazardShard, 256> active_guards_;

    std::mutex context_mutex_;

    TelemetryBlock telemetry_block_;
    NexusSeqWarmCache seq_warm_cache_;

public:
    NexusOrchestrator(llama_context* ctx, NexusSemanticSLB* slb, NexusRadixFSM* fsm, uint32_t base_pos,
                      float speculative_threshold = 0.88f, float speculative_margin = 0.05f,
                      float auto_route_margin = 0.10f,
                      size_t max_pinned_bytes = 16ULL * 1024 * 1024 * 1024,
                      uint32_t max_splice_pos = 256);
    ~NexusOrchestrator();

    NexusOrchestrator(const NexusOrchestrator&) = delete;
    NexusOrchestrator& operator=(const NexusOrchestrator&) = delete;

    void register_tool_path(uint32_t tool_id, const std::string& path);
    void register_tool_schema_tokens(uint32_t tool_id, const std::vector<int32_t>& tokens);
    void set_recompute_pct(float pct) { recompute_pct_ = pct; }
    float get_recompute_pct() const { return recompute_pct_; }
    void set_max_splice_pos(uint32_t pos) { max_splice_pos_ = pos; }
    void preload_tool(uint32_t tool_id, const std::string& path);
    void pin_warm_tool(uint32_t tool_id) { seq_warm_cache_.pin_tool(tool_id); }
    void pin_block_tool(uint32_t tool_id) { if (block_cache_) block_cache_->pin_tool(tool_id); }
    uint64_t get_splice_guard_fallback_count() const { return splice_guard_fallback_count_.load(std::memory_order_relaxed); }
    uint32_t get_max_splice_pos() const { return max_splice_pos_; }

    uint32_t route_and_splice(const std::vector<int32_t>& user_query_tokens,
                              const std::vector<float>& query_embedding,
                              uint32_t n_past,
                              llama_seq_id seq_id = 0,
                              const std::vector<uint32_t>& query_lexical_hashes = {},
                              const std::vector<int32_t>& prefix_tokens = {},
                              uint32_t force_tool_id = 0);

    void register_prefix_tokens(uint32_t tool_id, const std::vector<int32_t>& prefix_tokens) {
        seq_warm_cache_.register_prefix_for_tool(tool_id, prefix_tokens);
    }

    std::shared_ptr<NexusBlockCache> get_cache() const { return block_cache_; }
    void release_hazard(llama_seq_id seq_id);
    llama_context* get_context() const { return ctx_; }

    PyTelemetryBlock get_telemetry() const {
        return PyTelemetryBlock {
            telemetry_block_.speculative_hit_count.load(std::memory_order_relaxed),
            telemetry_block_.speculative_miss_count.load(std::memory_order_relaxed),
            telemetry_block_.fsm_hidden_latency_us.load(std::memory_order_relaxed),
            telemetry_block_.exposed_splice_latency_us.load(std::memory_order_relaxed)
        };
    }
};
