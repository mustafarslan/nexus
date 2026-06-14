#pragma once
#include <cstdint>
#include <memory>
#include <span>
#include <vector>
#include "llama.h"
#include "nexus_block_cache.hpp"

class NexusKVSplicer {
public:
    static void invalidate_sequence(llama_context* ctx, llama_seq_id seq, int32_t start_pos, int32_t end_pos);

    static void inject_tool_page(llama_context* ctx, std::shared_ptr<AeonToolBlock> block, uint32_t n_past, llama_seq_id target_seq = 0);

    static void inject_tool_page_raw(llama_context* ctx, void* mapped_data, uint32_t n_past, llama_seq_id target_seq = 0);

    static uint32_t unsplice_tool(llama_context* ctx, llama_seq_id seq_id, uint32_t splice_pos, uint32_t schema_len, uint32_t generated_len);

    static std::vector<float> read_kv_slice(
        llama_context* ctx,
        int il,
        int32_t p0,
        int32_t p1,
        bool read_v = false);
};
