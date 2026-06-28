#pragma once
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "llama.h"

/** Lock-free-read exact-token LCP radix cache over a pool of llama_seq_id slots. */
class NexusRadixPrefixCache {
public:
    static constexpr llama_seq_id POOL_BASE = 1;
    static constexpr size_t POOL_SIZE = 32;
    static constexpr llama_seq_id REQUEST_SEQ_BASE = 64;
    static constexpr uint32_t NODE_ARENA_RESERVE = 4096;
    static constexpr uint32_t TOKEN_ARENA_RESERVE = 65536;

#pragma pack(push, 1)
    struct RadixNode {
        uint32_t token_start_idx = UINT32_MAX;
        uint16_t token_length = 0;
        uint32_t first_child_idx = UINT32_MAX;
        uint32_t next_sibling_idx = UINT32_MAX;
        uint32_t pool_index = UINT32_MAX;
        uint32_t prefix_end_pos = 0;
    };
#pragma pack(pop)

    struct alignas(128) PoolSlotMeta {
        llama_seq_id seq_id = 0;
        uint32_t end_pos = 0;
        uint32_t tool_id = 0;
        bool occupied = false;
        std::atomic<uint64_t> last_access_tick{0};
        std::atomic<uint32_t> ref_count{0};
    };

    NexusRadixPrefixCache();
    ~NexusRadixPrefixCache() = default;

    NexusRadixPrefixCache(const NexusRadixPrefixCache&) = delete;
    NexusRadixPrefixCache& operator=(const NexusRadixPrefixCache&) = delete;

    bool is_pinned(uint32_t tool_id) const;
    void pin_tool(uint32_t tool_id);
    void unpin_tool(uint32_t tool_id);
    void clear_pins();

    uint32_t try_copy_prefix(llama_context* ctx, std::span<const int32_t> prefix_tokens,
                             llama_seq_id dst_seq);
    bool try_copy_to(llama_context* ctx, uint32_t tool_id, uint32_t n_past, llama_seq_id dst_seq);

    // F0 deep-path probe: read-only. Returns the number of position-safe prefix
    // positions that try_copy_prefix WOULD copy (0 = miss), WITHOUT performing the
    // seq_cp or touching LRU/ref_count. Used to measure deep-path L0 reachability
    // without mutating KV state or the fallback path.
    uint32_t probe_prefix(std::span<const int32_t> prefix_tokens) const;

    void update_from_seq(llama_context* ctx, std::span<const int32_t> prefix_tokens, uint32_t end_pos,
                         llama_seq_id src_seq, uint32_t tool_id = 0);
    void update_warm_from_seq(llama_context* ctx, uint32_t tool_id, uint32_t n_past, uint32_t schema_len,
                              llama_seq_id src_seq);
    void release_pool_ref(size_t pool_index);
    void invalidate_tool(uint32_t tool_id);
    void clear();
    void register_prefix_for_tool(uint32_t tool_id, std::span<const int32_t> prefix_tokens);

    static llama_seq_id allocate_request_seq(uint32_t request_index);

private:
    static uint64_t steady_tick() noexcept;

    uint32_t find_child(uint32_t parent_idx, int32_t first_token) const;
    // Shared LCP matcher. Caller MUST hold a (shared) lock on trie_rw_lock_.
    // Returns best_end (positions matched, 0 = miss) and sets out_best_pool.
    // Pure reads only — used by both try_copy_prefix (Path A) and probe_prefix (Path B).
    uint32_t match_prefix_locked(std::span<const int32_t> prefix_tokens, uint32_t& out_best_pool) const;
    uint32_t alloc_node();
    uint32_t append_tokens(std::span<const int32_t> tokens);
    size_t acquire_pool_slot();
    void insert_radix_path(std::span<const int32_t> prefix_tokens, uint32_t end_pos, size_t slot_idx);

    mutable std::shared_mutex trie_rw_lock_;
    std::vector<RadixNode> node_arena_;
    std::vector<int32_t> token_arena_;
    alignas(128) std::array<PoolSlotMeta, POOL_SIZE> pool_meta_{};

    mutable std::mutex pin_mutex_;
    std::unordered_map<uint32_t, std::vector<int32_t>> tool_prefixes_;
    std::unordered_set<uint32_t> pinned_tools_;
};

using NexusSeqWarmCache = NexusRadixPrefixCache;
