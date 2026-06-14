#pragma once
#include <vector>
#include <mutex>
#include <cstdint>
#include <memory>
#include <span>
#include "llama.h"

// Assume llama_token is an int32_t as per llama.cpp
using llama_token = int32_t;

// Maximum concurrent sequences for lock-free flat array indexing.
// seq_id must satisfy 0 <= seq_id < MAX_SEQUENCES.
static constexpr size_t MAX_SEQUENCES = 1024;

struct RadixNode {
    llama_token token = 0;
    uint32_t first_child_idx = UINT32_MAX;
    uint32_t next_sibling_idx = UINT32_MAX;
    uint32_t tool_id = UINT32_MAX; // Use UINT32_MAX to denote non-leaf
};

enum class RoutingState {
    IDLE,         // Normal generation, no masking
    NAVIGATING,   // Inside the tool routing tree
    LEAF_REACHED, // Tool successfully selected
    INVALID       // Mathematically impossible if mask is applied correctly
};

// Per-sequence FSM state. Each slot is aligned to 128 bytes to prevent
// false sharing on Apple Silicon (M1-M4), which tracks L2 coherency in
// 128-byte sectors via the AMBA Coherence Protocol. alignas(64) would
// allow two adjacent FSMState slots to reside in the same coherency block.
// The scratch_buffer is pre-allocated once (reserve(512)) and .clear()'d
// on each hot-path call, guaranteeing ZERO heap allocations during generation.
struct alignas(128) FSMState {
    uint32_t current_node_idx = UINT32_MAX;
    RoutingState state = RoutingState::IDLE;
    uint32_t resolved_tool_id = 0;
    bool active = false; // Slot occupancy flag

    // Pre-allocated scratch buffer for apply_logit_mask().
    // Reserved once at begin_routing(), cleared per-use. Zero malloc on hot path.
    std::vector<std::pair<llama_token, float>> scratch_buffer;
};

class NexusRadixFSM {
private:
    std::vector<RadixNode> node_arena_;

    // Lock-free flat array. seq_id is a direct O(1) index.
    // Thread safety: each concurrent request operates on a disjoint seq_id.
    // No two threads ever access active_states_[i] for the same i simultaneously
    // (guaranteed by the inference server's sequence scheduler).
    // alignas(128) on FSMState prevents hardware-level false sharing
    // on Apple Silicon's 128-byte L2 coherency sectors.
    std::vector<FSMState> active_states_;

    // Cold-path mutex ONLY for trie mutation (add_route at init time).
    // Never acquired on the generation hot path.
    std::mutex trie_mutex_;

public:
    NexusRadixFSM();
    
    // Built at initialization (cold path, mutex-protected)
    void add_route(uint32_t tool_id, std::span<const llama_token> token_sequence);
    
    // State management (lock-free, direct array index)
    void reset(llama_seq_id seq_id = 0);
    void begin_routing(llama_seq_id seq_id = 0); // Transitions from IDLE to NAVIGATING
    void force_resolved(uint32_t tool_id, llama_seq_id seq_id = 0);
    RoutingState advance(llama_token sampled_token, llama_seq_id seq_id = 0);
    
    // The Zero-Copy Masking Hot-Path. LOCK-FREE.
    // Takes a raw span of logits (size = vocab_size).
    // Sets all elements to -INFINITY, except the valid children of current_node_.
    void apply_logit_mask(std::span<float> logits, llama_seq_id seq_id = 0);
    
    RoutingState get_state(llama_seq_id seq_id = 0) const;
    uint32_t get_resolved_tool_id(llama_seq_id seq_id = 0) const;
};
