#pragma once
#include <cstdint>
#include <memory>
#include "llama.h"
#include "nexus_block_cache.hpp"

class NexusKVSplicer {
public:
    // 1. Remove the invalid KV cache (SLB hints, User Prompt, Route)
    // Uses: llama_kv_cache_seq_rm(ctx, seq_id, start_pos, end_pos)
    static void invalidate_sequence(llama_context* ctx, llama_seq_id seq, int32_t start_pos, int32_t end_pos);

    // 2. The GPU Memory Splice
    // Overwrites the active KV cache starting at n_past using mmap'd ATB block data.
    // Also updates the internal llama.cpp cells metadata for correctness.
    static void inject_tool_page(llama_context* ctx, std::shared_ptr<AeonToolBlock> block, uint32_t n_past, llama_seq_id target_seq = 0);

    // Phase 25: Raw Memory Splice (Zero-GC Dependency)
    static void inject_tool_page_raw(llama_context* ctx, void* mapped_data, uint32_t n_past, llama_seq_id target_seq = 0);

    // 3. Ephemeral Context Pop (Unsplice)
    // Erases the tool schema from KV Cache and shifts downstream query and generated tokens backward.
    // Returns the number of tokens removed.
    static uint32_t unsplice_tool(llama_context* ctx, llama_seq_id seq_id, uint32_t splice_pos, uint32_t schema_len, uint32_t generated_len);
};
