#include "nexus_fsm.hpp"
#include <limits>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <stdexcept>

NexusRadixFSM::NexusRadixFSM() {
    root_ = std::make_unique<RadixNode>();
    // Pre-allocate the flat array. Each FSMState is default-initialized (active=false).
    active_states_.resize(MAX_SEQUENCES);
}

void NexusRadixFSM::reset(llama_seq_id seq_id) {
    // Bounds check (lock-free)
    if (seq_id < 0 || static_cast<size_t>(seq_id) >= MAX_SEQUENCES) {
        std::cerr << "NEXUS ERROR: seq_id " << seq_id << " out of bounds [0, " << MAX_SEQUENCES << ").\n";
        return;
    }
    auto& state = active_states_[seq_id];
    state.current_node = nullptr;
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
    state.current_node = root_.get();
    state.state = RoutingState::NAVIGATING;
    state.resolved_tool_id = 0;
    state.active = true;

    // Pre-allocate scratch buffer on first use. Subsequent calls reuse capacity.
    // reserve() is a no-op if capacity() >= 512.
    state.scratch_buffer.reserve(512);
}

void NexusRadixFSM::add_route(uint32_t tool_id, std::span<const llama_token> token_sequence) {
    // Cold path — mutex is acceptable here (called at init, not on generation hot path)
    std::lock_guard<std::mutex> lock(trie_mutex_);
    RadixNode* curr = root_.get();
    for (llama_token token : token_sequence) {
        RadixNode* next = nullptr;
        for (const auto& child : curr->children) {
            if (child.first == token) {
                next = child.second.get();
                break;
            }
        }
        if (!next) {
            auto new_node = std::make_unique<RadixNode>();
            next = new_node.get();
            curr->children.emplace_back(token, std::move(new_node));
        }
        curr = next;
    }
    curr->is_leaf = true;
    curr->tool_id = tool_id;
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
    
    RadixNode* next = nullptr;
    for (const auto& child : state.current_node->children) {
        if (child.first == sampled_token) {
            next = child.second.get();
            break;
        }
    }
    
    if (!next) {
        state.state = RoutingState::INVALID;
        return state.state;
    }
    
    state.current_node = next;
    if (state.current_node->is_leaf) {
        state.resolved_tool_id = state.current_node->tool_id;
        state.state = RoutingState::LEAF_REACHED;
    }
    
    return state.state;
}

void NexusRadixFSM::apply_logit_mask(std::span<float> logits, llama_seq_id seq_id) {
    // ==========================================
    // LOCK-FREE HOT PATH — ZERO MUTEX, ZERO MALLOC
    // ==========================================
    // Thread safety: each seq_id maps to a disjoint FSMState slot.
    // The inference server guarantees no two threads process the same seq_id.
    // alignas(64) on FSMState prevents hardware false sharing.

    if (seq_id < 0 || static_cast<size_t>(seq_id) >= MAX_SEQUENCES) {
        return;
    }

    FSMState& state = active_states_[seq_id];
    if (!state.active || state.state != RoutingState::NAVIGATING) {
        return;
    }
    
    const auto& children = state.current_node->children;
    if (children.empty()) {
        std::cerr << "NEXUS FATAL ERROR: FSM is in NAVIGATING state but active RadixNode has no valid children.\n";
        std::abort();
    }
    
    // ZERO ALLOCATION: reuse pre-allocated scratch buffer.
    // .clear() sets size to 0 but retains allocated capacity.
    state.scratch_buffer.clear();
    
    // Step 1: Save original logits for valid child tokens
    for (const auto& child : children) {
        llama_token token = child.first;
        if (token < 0 || static_cast<size_t>(token) >= logits.size()) {
            std::cerr << "NEXUS FATAL ERROR: Child token ID " << token << " is out of bounds for vocab size " << logits.size() << ".\n";
            std::abort();
        }
        state.scratch_buffer.push_back({ token, logits[token] });
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
