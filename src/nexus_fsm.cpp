#include "nexus_fsm.hpp"
#include <limits>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <stdexcept>

NexusRadixFSM::NexusRadixFSM() {
    // Pre-reserve node arena capacity to prevent re-allocations during initialization.
    node_arena_.reserve(4096);

    // Default construct the root node at index 0.
    RadixNode root{};
    root.token = 0;
    root.first_child_idx = UINT32_MAX;
    root.next_sibling_idx = UINT32_MAX;
    root.tool_id = UINT32_MAX;
    node_arena_.push_back(root);

    // Pre-allocate the flat array. Each FSMState is default-initialized.
    active_states_.resize(MAX_SEQUENCES);
}

void NexusRadixFSM::reset(llama_seq_id seq_id) {
    // Bounds check (lock-free)
    if (seq_id < 0 || static_cast<size_t>(seq_id) >= MAX_SEQUENCES) {
        std::cerr << "NEXUS ERROR: seq_id " << seq_id << " out of bounds [0, " << MAX_SEQUENCES << ").\n";
        return;
    }
    auto& state = active_states_[seq_id];
    state.current_node_idx = UINT32_MAX;
    state.state = RoutingState::IDLE;
    state.resolved_tool_id = 0;
    state.active = false;
    // Note: we do NOT deallocate scratch_buffer here.
    // The capacity is retained for reuse (zero-allocation invariant).
    state.scratch_buffer.clear();
}

void NexusRadixFSM::begin_routing(llama_seq_id seq_id) {
    // Bounds check (lock-free)
    if (seq_id < 0 || static_cast<size_t>(seq_id) >= MAX_SEQUENCES) {
        std::cerr << "NEXUS ERROR: seq_id " << seq_id << " out of bounds [0, " << MAX_SEQUENCES << ").\n";
        return;
    }
    auto& state = active_states_[seq_id];
    state.current_node_idx = 0; // Root node is index 0
    state.state = RoutingState::NAVIGATING;
    state.resolved_tool_id = 0;
    state.active = true;

    // Pre-allocate scratch buffer on first use. Subsequent calls reuse capacity.
    state.scratch_buffer.reserve(512);
}

void NexusRadixFSM::force_resolved(uint32_t tool_id, llama_seq_id seq_id) {
    if (seq_id < 0 || static_cast<size_t>(seq_id) >= MAX_SEQUENCES) {
        std::cerr << "NEXUS ERROR: seq_id " << seq_id << " out of bounds [0, " << MAX_SEQUENCES << ").\n";
        return;
    }
    auto& state = active_states_[seq_id];
    state.resolved_tool_id = tool_id;
    state.state = RoutingState::LEAF_REACHED;
    state.active = true;
}

void NexusRadixFSM::add_route(uint32_t tool_id, std::span<const llama_token> token_sequence) {
    // Cold path — mutex is acceptable here (called at init, not on generation hot path)
    std::lock_guard<std::mutex> lock(trie_mutex_);
    uint32_t curr_idx = 0; // Start at root
    for (llama_token token : token_sequence) {
        uint32_t next_idx = UINT32_MAX;
        uint32_t child_idx = node_arena_[curr_idx].first_child_idx;
        while (child_idx != UINT32_MAX) {
            if (node_arena_[child_idx].token == token) {
                next_idx = child_idx;
                break;
            }
            child_idx = node_arena_[child_idx].next_sibling_idx;
        }

        if (next_idx == UINT32_MAX) {
            // Allocate a new node in the arena
            next_idx = static_cast<uint32_t>(node_arena_.size());
            RadixNode new_node{};
            new_node.token = token;
            new_node.first_child_idx = UINT32_MAX;
            new_node.next_sibling_idx = UINT32_MAX;
            new_node.tool_id = UINT32_MAX;
            node_arena_.push_back(new_node);

            // Link to parent/sibling
            if (node_arena_[curr_idx].first_child_idx == UINT32_MAX) {
                node_arena_[curr_idx].first_child_idx = next_idx;
            } else {
                uint32_t last_sibling_idx = node_arena_[curr_idx].first_child_idx;
                while (node_arena_[last_sibling_idx].next_sibling_idx != UINT32_MAX) {
                    last_sibling_idx = node_arena_[last_sibling_idx].next_sibling_idx;
                }
                node_arena_[last_sibling_idx].next_sibling_idx = next_idx;
            }
        }
        curr_idx = next_idx;
    }
    node_arena_[curr_idx].tool_id = tool_id;
}

RoutingState NexusRadixFSM::advance(llama_token sampled_token, llama_seq_id seq_id) {
    // Lock-free: direct array index on disjoint seq_id
    if (seq_id < 0 || static_cast<size_t>(seq_id) >= MAX_SEQUENCES) {
        return RoutingState::INVALID;
    }
    
    FSMState& state = active_states_[seq_id];
    if (!state.active) {
        return RoutingState::INVALID;
    }
    if (state.state != RoutingState::NAVIGATING) {
        return state.state;
    }
    
    uint32_t curr_idx = state.current_node_idx;
    if (curr_idx == UINT32_MAX || curr_idx >= node_arena_.size()) {
        state.state = RoutingState::INVALID;
        return state.state;
    }

    uint32_t next_idx = UINT32_MAX;
    uint32_t child_idx = node_arena_[curr_idx].first_child_idx;
    while (child_idx != UINT32_MAX) {
        if (node_arena_[child_idx].token == sampled_token) {
            next_idx = child_idx;
            break;
        }
        child_idx = node_arena_[child_idx].next_sibling_idx;
    }
    
    if (next_idx == UINT32_MAX) {
        state.state = RoutingState::INVALID;
        return state.state;
    }
    
    state.current_node_idx = next_idx;
    if (node_arena_[next_idx].tool_id != UINT32_MAX) {
        state.resolved_tool_id = node_arena_[next_idx].tool_id;
        state.state = RoutingState::LEAF_REACHED;
    }
    
    return state.state;
}

void NexusRadixFSM::apply_logit_mask(std::span<float> logits, llama_seq_id seq_id) {
    // ==========================================
    // LOCK-FREE HOT PATH — ZERO MUTEX, ZERO MALLOC
    // ==========================================
    if (seq_id < 0 || static_cast<size_t>(seq_id) >= MAX_SEQUENCES) {
        return;
    }

    FSMState& state = active_states_[seq_id];
    if (!state.active || state.state != RoutingState::NAVIGATING) {
        return;
    }
    
    uint32_t curr_idx = state.current_node_idx;
    if (curr_idx == UINT32_MAX || curr_idx >= node_arena_.size()) {
        return;
    }

    uint32_t child_idx = node_arena_[curr_idx].first_child_idx;
    if (child_idx == UINT32_MAX) {
        std::cerr << "NEXUS FATAL ERROR: FSM is in NAVIGATING state but active RadixNode has no valid children.\n";
        std::abort();
    }
    
    // ZERO ALLOCATION: reuse pre-allocated scratch buffer.
    state.scratch_buffer.clear();
    
    // Step 1: Save original logits for valid child tokens
    while (child_idx != UINT32_MAX) {
        llama_token token = node_arena_[child_idx].token;
        if (token < 0 || static_cast<size_t>(token) >= logits.size()) {
            std::cerr << "NEXUS FATAL ERROR: Child token ID " << token << " is out of bounds for vocab size " << logits.size() << ".\n";
            std::abort();
        }
        state.scratch_buffer.push_back({ token, logits[token] });
        child_idx = node_arena_[child_idx].next_sibling_idx;
    }
    
    // Step 2: Vectorized memory wipe setting everything to -INFINITY
    std::fill(logits.begin(), logits.end(), -std::numeric_limits<float>::infinity());
    
    // Step 3: Write back saved original logits
    for (const auto& item : state.scratch_buffer) {
        logits[item.first] = item.second;
    }
}

RoutingState NexusRadixFSM::get_state(llama_seq_id seq_id) const {
    // Lock-free read
    if (seq_id < 0 || static_cast<size_t>(seq_id) >= MAX_SEQUENCES) {
        return RoutingState::IDLE;
    }
    return active_states_[seq_id].state;
}

uint32_t NexusRadixFSM::get_resolved_tool_id(llama_seq_id seq_id) const {
    // Lock-free read
    if (seq_id < 0 || static_cast<size_t>(seq_id) >= MAX_SEQUENCES) {
        return 0;
    }
    return active_states_[seq_id].resolved_tool_id;
}
