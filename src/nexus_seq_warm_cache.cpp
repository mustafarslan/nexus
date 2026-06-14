#include "nexus_seq_warm_cache.hpp"
#include <algorithm>
#include <limits>

NexusRadixPrefixCache::NexusRadixPrefixCache() {
    node_arena_.reserve(NODE_ARENA_RESERVE);
    token_arena_.reserve(TOKEN_ARENA_RESERVE);
    RadixNode root{};
    node_arena_.push_back(root);
    for (size_t i = 0; i < POOL_SIZE; ++i) {
        pool_meta_[i].seq_id = static_cast<llama_seq_id>(POOL_BASE + i);
        pool_meta_[i].occupied = false;
        pool_meta_[i].end_pos = 0;
        pool_meta_[i].tool_id = 0;
        pool_meta_[i].last_access_tick.store(0, std::memory_order_relaxed);
        pool_meta_[i].ref_count.store(0, std::memory_order_relaxed);
    }
}

uint64_t NexusRadixPrefixCache::steady_tick() noexcept {
    return static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
}

bool NexusRadixPrefixCache::is_pinned(uint32_t tool_id) const {
    std::lock_guard<std::mutex> lock(pin_mutex_);
    return pinned_tools_.count(tool_id) > 0;
}

void NexusRadixPrefixCache::pin_tool(uint32_t tool_id) {
    std::lock_guard<std::mutex> lock(pin_mutex_);
    pinned_tools_.insert(tool_id);
}

void NexusRadixPrefixCache::unpin_tool(uint32_t tool_id) {
    std::lock_guard<std::mutex> lock(pin_mutex_);
    pinned_tools_.erase(tool_id);
}

void NexusRadixPrefixCache::clear_pins() {
    std::lock_guard<std::mutex> lock(pin_mutex_);
    pinned_tools_.clear();
}

uint32_t NexusRadixPrefixCache::find_child(uint32_t parent_idx, int32_t first_token) {
    if (parent_idx >= node_arena_.size()) {
        return UINT32_MAX;
    }
    uint32_t child_idx = node_arena_[parent_idx].first_child_idx;
    while (child_idx != UINT32_MAX && child_idx < node_arena_.size()) {
        const RadixNode& child = node_arena_[child_idx];
        if (child.token_length > 0) {
            const int32_t edge_first = token_arena_[child.token_start_idx];
            if (edge_first == first_token) {
                return child_idx;
            }
        }
        child_idx = child.next_sibling_idx;
    }
    return UINT32_MAX;
}

uint32_t NexusRadixPrefixCache::alloc_node() {
    node_arena_.emplace_back();
    return static_cast<uint32_t>(node_arena_.size() - 1);
}

uint32_t NexusRadixPrefixCache::append_tokens(std::span<const int32_t> tokens) {
    const uint32_t start = static_cast<uint32_t>(token_arena_.size());
    token_arena_.insert(token_arena_.end(), tokens.begin(), tokens.end());
    return start;
}

uint32_t NexusRadixPrefixCache::try_copy_prefix(llama_context* ctx, std::span<const int32_t> prefix_tokens,
                                                llama_seq_id dst_seq) {
    if (!ctx || prefix_tokens.empty()) {
        return 0;
    }

    std::shared_lock<std::shared_mutex> lock(trie_rw_lock_);

    uint32_t curr = 0;
    size_t off = 0;
    uint32_t best_pool = UINT32_MAX;
    uint32_t best_end = 0;

    while (off < prefix_tokens.size() && curr < node_arena_.size()) {
        const RadixNode& node = node_arena_[curr];
        if (node.pool_index < POOL_SIZE && pool_meta_[node.pool_index].occupied) {
            best_pool = node.pool_index;
            best_end = node.prefix_end_pos;
        }

        const int32_t key = prefix_tokens[off];
        const uint32_t child = find_child(curr, key);
        if (child == UINT32_MAX || child >= node_arena_.size()) {
            break;
        }

        const RadixNode& edge = node_arena_[child];
        uint16_t matched = 0;
        for (uint16_t i = 0; i < edge.token_length && off + i < prefix_tokens.size(); ++i) {
            if (token_arena_[edge.token_start_idx + i] != prefix_tokens[off + i]) {
                break;
            }
            ++matched;
        }
        if (matched == 0) {
            break;
        }
        off += matched;
        if (matched < edge.token_length) {
            break;
        }
        curr = child;
    }

    if (curr != UINT32_MAX && curr < node_arena_.size()) {
        const RadixNode& terminal = node_arena_[curr];
        if (terminal.pool_index < POOL_SIZE && pool_meta_[terminal.pool_index].occupied) {
            if (off >= best_end) {
                best_pool = terminal.pool_index;
                best_end = static_cast<uint32_t>(off);
            }
        }
    }

    if (best_pool == UINT32_MAX || best_end == 0) {
        return 0;
    }

    const PoolSlotMeta& slot = pool_meta_[best_pool];
    llama_kv_cache_seq_cp(ctx, slot.seq_id, dst_seq, 0, static_cast<llama_pos>(best_end));

    const uint64_t now = steady_tick();
    pool_meta_[best_pool].last_access_tick.store(now, std::memory_order_relaxed);
    pool_meta_[best_pool].ref_count.fetch_add(1, std::memory_order_relaxed);
    return best_end;
}

bool NexusRadixPrefixCache::try_copy_to(llama_context* ctx, uint32_t tool_id, uint32_t n_past,
                                        llama_seq_id dst_seq) {
    std::vector<int32_t> prefix;
    {
        std::shared_lock<std::shared_mutex> lock(trie_rw_lock_);
        auto it = tool_prefixes_.find(tool_id);
        if (it == tool_prefixes_.end() || it->second.size() < n_past) {
            return false;
        }
        prefix.assign(it->second.begin(), it->second.begin() + static_cast<ptrdiff_t>(n_past));
    }
    return try_copy_prefix(ctx, prefix, dst_seq) >= n_past;
}

size_t NexusRadixPrefixCache::acquire_pool_slot() {
    for (size_t i = 0; i < POOL_SIZE; ++i) {
        if (!pool_meta_[i].occupied) {
            return i;
        }
    }

    size_t victim = 0;
    uint64_t oldest = std::numeric_limits<uint64_t>::max();
    for (size_t i = 0; i < POOL_SIZE; ++i) {
        PoolSlotMeta& slot = pool_meta_[i];
        if (!slot.occupied) {
            return i;
        }
        if (slot.ref_count.load(std::memory_order_acquire) > 0) {
            continue;
        }
        if (slot.tool_id != 0 && is_pinned(slot.tool_id)) {
            continue;
        }
        const uint64_t tick = slot.last_access_tick.load(std::memory_order_relaxed);
        if (tick < oldest) {
            oldest = tick;
            victim = i;
        }
    }
    pool_meta_[victim].occupied = false;
    pool_meta_[victim].end_pos = 0;
    pool_meta_[victim].tool_id = 0;
    return victim;
}

void NexusRadixPrefixCache::insert_radix_path(std::span<const int32_t> prefix_tokens, uint32_t end_pos,
                                              size_t slot_idx) {
    if (prefix_tokens.empty()) {
        return;
    }

    uint32_t curr = 0;
    size_t off = 0;
    uint32_t pos_cursor = 0;

    while (off < prefix_tokens.size()) {
        const int32_t key = prefix_tokens[off];
        uint32_t child = find_child(curr, key);

        if (child == UINT32_MAX || child >= node_arena_.size()) {
            const size_t remain = prefix_tokens.size() - off;
            const uint32_t start = append_tokens(prefix_tokens.subspan(off, remain));
            const uint32_t new_idx = alloc_node();
            RadixNode& new_node = node_arena_[new_idx];
            new_node.token_start_idx = start;
            new_node.token_length = static_cast<uint16_t>(remain);
            new_node.pool_index = static_cast<uint32_t>(slot_idx);
            pos_cursor += static_cast<uint32_t>(remain);
            new_node.prefix_end_pos = std::min(end_pos, pos_cursor);

            RadixNode& parent = node_arena_[curr];
            new_node.next_sibling_idx = parent.first_child_idx;
            parent.first_child_idx = new_idx;
            return;
        }

        RadixNode& edge = node_arena_[child];
        uint16_t matched = 0;
        for (uint16_t i = 0; i < edge.token_length && off + i < prefix_tokens.size(); ++i) {
            if (token_arena_[edge.token_start_idx + i] != prefix_tokens[off + i]) {
                break;
            }
            ++matched;
        }

        if (matched < edge.token_length) {
            const uint32_t split_idx = alloc_node();
            RadixNode& split_node = node_arena_[split_idx];
            split_node.token_start_idx = edge.token_start_idx + matched;
            split_node.token_length = static_cast<uint16_t>(edge.token_length - matched);
            split_node.first_child_idx = edge.first_child_idx;
            split_node.next_sibling_idx = edge.next_sibling_idx;
            split_node.pool_index = edge.pool_index;
            split_node.prefix_end_pos = edge.prefix_end_pos;

            edge.token_length = matched;
            edge.first_child_idx = split_idx;
            edge.next_sibling_idx = UINT32_MAX;
            edge.pool_index = UINT32_MAX;

            off += matched;
            if (off >= prefix_tokens.size()) {
                edge.pool_index = static_cast<uint32_t>(slot_idx);
                pos_cursor += matched;
                edge.prefix_end_pos = std::min(end_pos, pos_cursor);
                return;
            }

            const size_t remain = prefix_tokens.size() - off;
            const uint32_t start = append_tokens(prefix_tokens.subspan(off, remain));
            const uint32_t new_idx = alloc_node();
            RadixNode& new_node = node_arena_[new_idx];
            new_node.token_start_idx = start;
            new_node.token_length = static_cast<uint16_t>(remain);
            new_node.pool_index = static_cast<uint32_t>(slot_idx);
            pos_cursor += static_cast<uint32_t>(matched + remain);
            new_node.prefix_end_pos = std::min(end_pos, pos_cursor);
            split_node.first_child_idx = new_idx;
            return;
        }

        off += matched;
        pos_cursor += matched;
        edge.pool_index = static_cast<uint32_t>(slot_idx);
        edge.prefix_end_pos = std::min(end_pos, pos_cursor);
        curr = child;
    }
}

void NexusRadixPrefixCache::update_from_seq(llama_context* ctx, std::span<const int32_t> prefix_tokens,
                                            uint32_t end_pos, llama_seq_id src_seq, uint32_t tool_id) {
    if (!ctx || prefix_tokens.empty() || end_pos == 0) {
        return;
    }

    std::unique_lock<std::shared_mutex> lock(trie_rw_lock_);
    if (tool_id != 0) {
        tool_prefixes_[tool_id] = std::vector<int32_t>(prefix_tokens.begin(), prefix_tokens.end());
    }

    const size_t slot_idx = acquire_pool_slot();
    PoolSlotMeta& slot = pool_meta_[slot_idx];
    llama_kv_cache_seq_cp(ctx, src_seq, slot.seq_id, 0, static_cast<llama_pos>(end_pos));
    slot.end_pos = end_pos;
    slot.occupied = true;
    slot.tool_id = tool_id;
    slot.ref_count.store(0, std::memory_order_relaxed);
    slot.last_access_tick.store(steady_tick(), std::memory_order_relaxed);

    insert_radix_path(prefix_tokens, end_pos, slot_idx);
}

void NexusRadixPrefixCache::update_warm_from_seq(llama_context* ctx, uint32_t tool_id, uint32_t n_past,
                                                 uint32_t schema_len, llama_seq_id src_seq) {
    std::vector<int32_t> prefix;
    {
        std::shared_lock<std::shared_mutex> lock(trie_rw_lock_);
        auto it = tool_prefixes_.find(tool_id);
        if (it != tool_prefixes_.end()) {
            prefix = it->second;
        } else {
            prefix.resize(n_past, 0);
        }
    }
    update_from_seq(ctx, prefix, n_past + schema_len, src_seq, tool_id);
}

void NexusRadixPrefixCache::release_pool_ref(size_t pool_index) {
    if (pool_index >= POOL_SIZE) {
        return;
    }
    pool_meta_[pool_index].ref_count.fetch_sub(1, std::memory_order_relaxed);
}

void NexusRadixPrefixCache::invalidate_tool(uint32_t tool_id) {
    if (is_pinned(tool_id)) {
        return;
    }
    std::unique_lock<std::shared_mutex> lock(trie_rw_lock_);
    tool_prefixes_.erase(tool_id);
}

void NexusRadixPrefixCache::clear() {
    std::unique_lock<std::shared_mutex> lock(trie_rw_lock_);
    node_arena_.clear();
    token_arena_.clear();
    RadixNode root{};
    node_arena_.push_back(root);
    tool_prefixes_.clear();
    for (size_t i = 0; i < POOL_SIZE; ++i) {
        pool_meta_[i].occupied = false;
        pool_meta_[i].end_pos = 0;
        pool_meta_[i].tool_id = 0;
        pool_meta_[i].ref_count.store(0, std::memory_order_relaxed);
        pool_meta_[i].last_access_tick.store(0, std::memory_order_relaxed);
    }
    {
        std::lock_guard<std::mutex> pin_lock(pin_mutex_);
        pinned_tools_.clear();
    }
}

void NexusRadixPrefixCache::register_prefix_for_tool(uint32_t tool_id,
                                                     std::span<const int32_t> prefix_tokens) {
    std::unique_lock<std::shared_mutex> lock(trie_rw_lock_);
    tool_prefixes_[tool_id] = std::vector<int32_t>(prefix_tokens.begin(), prefix_tokens.end());
}

llama_seq_id NexusRadixPrefixCache::allocate_request_seq(uint32_t request_index) {
    return static_cast<llama_seq_id>(REQUEST_SEQ_BASE + (request_index % 4096));
}
