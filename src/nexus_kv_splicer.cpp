#include "nexus_kv_splicer.hpp"
#include "nexus_rope_math.hpp"
#include <vector>
#include <cassert>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#include <ggml.h>
#include <ggml-backend.h>

// Declarations of bridge functions from llama.cpp
extern "C" {
    struct ggml_tensor * llama_kv_cache_get_k(struct llama_context * ctx, int il);
    struct ggml_tensor * llama_kv_cache_get_v(struct llama_context * ctx, int il);
    bool llama_kv_cache_get_v_trans(struct llama_context * ctx);
    uint32_t llama_kv_cache_get_size(struct llama_context * ctx);
    uint32_t llama_model_n_head_kv(const struct llama_model * model, int il);
    uint32_t llama_model_d_head(const struct llama_model * model);
    void llama_kv_cache_set_cell(struct llama_context * ctx, uint32_t cell_idx, llama_pos pos, llama_seq_id seq_id);
    int32_t llama_kv_cache_get_physical_pos(struct llama_context * ctx, llama_pos pos);
    void llama_tensor_set_async(struct llama_context * ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    void llama_backend_sync(struct llama_context * ctx);
    float llama_context_yarn_ext_factor(const struct llama_context * ctx);
    float llama_context_yarn_beta_fast(const struct llama_context * ctx);
    float llama_context_yarn_beta_slow(const struct llama_context * ctx);
    uint32_t llama_context_yarn_n_ctx_orig(const struct llama_context * ctx);
    void llama_kv_cache_rope_shift_gpu(struct llama_context * ctx, int32_t p, int32_t seq_len, int32_t delta_pos);
}

#include <chrono>

void NexusKVSplicer::invalidate_sequence(llama_context* ctx, llama_seq_id seq, int32_t start_pos, int32_t end_pos) {
    // Invalidate cell metadata and zero KV cache values in standard range
    llama_kv_cache_seq_rm(ctx, seq, static_cast<llama_pos>(start_pos), static_cast<llama_pos>(end_pos));
}

void NexusKVSplicer::inject_tool_page(llama_context* ctx, std::shared_ptr<AeonToolBlock> block, uint32_t n_past, llama_seq_id target_seq) {
    if (!block) {
        throw std::runtime_error("NexusKVSplicer Error: AeonToolBlock is null.");
    }
    block->wait_until_resident();
    inject_tool_page_raw(ctx, block->get_mapped_data(), n_past, target_seq);
}

void NexusKVSplicer::inject_tool_page_raw(llama_context* ctx, void* mapped_data, uint32_t n_past, llama_seq_id target_seq) {
    auto t_start = std::chrono::high_resolution_clock::now();
    if (!mapped_data) {
        throw std::runtime_error("NexusKVSplicer Error: mapped_data is null.");
    }
    const AeonToolBlockHeader* header = static_cast<const AeonToolBlockHeader*>(mapped_data);
    if (!header) {
        throw std::runtime_error("NexusKVSplicer Error: AeonToolBlockHeader is null.");
    }

    const struct llama_model* model = llama_get_model(ctx);
    if (!model) {
        throw std::runtime_error("NexusKVSplicer Error: llama_model is null.");
    }

    uint32_t n_layer = header->n_layer;
    uint32_t n_head_kv = header->n_head_kv;
    uint32_t d_head = header->d_head;
    uint32_t seq_len = header->seq_len;

    uint32_t kv_size = llama_kv_cache_get_size(ctx);
    bool v_trans = llama_kv_cache_get_v_trans(ctx);

    if (v_trans) {
        throw std::runtime_error("NexusKVSplicer Error: Flash Attention must be enabled (v_trans == false).");
    }

    // Look up the physical index assigned to the logical sequence position n_past
    int32_t p = -1;
    if (n_past > 0) {
        int32_t p_prev = llama_kv_cache_get_physical_pos(ctx, static_cast<llama_pos>(n_past - 1));
        if (p_prev != -1) {
            p = (p_prev + 1) % kv_size;
        }
    }
    if (p == -1) {
        p = llama_kv_cache_get_physical_pos(ctx, static_cast<llama_pos>(n_past));
        if (p == -1) {
            p = static_cast<int32_t>(n_past % kv_size);
        }
    }

    // Spans to read-only mapped data
    const uint8_t* base_ptr = static_cast<const uint8_t*>(mapped_data);
    std::span<const uint8_t> k_tensors_span(base_ptr + header->k_tensor_offset, header->k_total_bytes);
    std::span<const uint8_t> v_tensors_span(base_ptr + header->v_tensor_offset, header->v_total_bytes);

    const uint16_t* src_k_base = reinterpret_cast<const uint16_t*>(k_tensors_span.data());
    const uint16_t* src_v_base = reinterpret_cast<const uint16_t*>(v_tensors_span.data());

    size_t layer_elements = n_head_kv * seq_len * d_head;

    auto t_prep = std::chrono::high_resolution_clock::now();
    double prep_us = std::chrono::duration<double, std::micro>(t_prep - t_start).count();

    double k_splice_us = 0.0;
    double v_splice_us = 0.0;

    int32_t delta_pos = static_cast<int32_t>(n_past) - static_cast<int32_t>(header->base_pos);

    // Retrieve YaRN parameters from context
    float ext_factor = llama_context_yarn_ext_factor(ctx);
    float beta_fast = llama_context_yarn_beta_fast(ctx);
    float beta_slow = llama_context_yarn_beta_slow(ctx);
    int n_ctx_orig = static_cast<int>(llama_context_yarn_n_ctx_orig(ctx));

    for (uint32_t il = 0; il < n_layer; ++il) {
        struct ggml_tensor * k_tensor = llama_kv_cache_get_k(ctx, il);
        struct ggml_tensor * v_tensor = llama_kv_cache_get_v(ctx, il);
        if (!k_tensor || !v_tensor) {
            throw std::runtime_error("NexusKVSplicer Error: Failed to retrieve KV cache tensors for layer " + std::to_string(il));
        }

        const uint16_t* src_k_layer = src_k_base + il * layer_elements;
        const uint16_t* src_v_layer = src_v_base + il * layer_elements;
        size_t k_size_row = n_head_kv * d_head * sizeof(uint16_t);
        size_t v_size_row = n_head_kv * d_head * sizeof(uint16_t);

        // Hardware Topology Abstraction (UMA vs. Discrete GPU)
        void* host_k_ptr = k_tensor->data;
        void* host_v_ptr = v_tensor->data;

        if (host_k_ptr != nullptr) {
            // UMA Path (Apple Metal / CPU): Direct zero-copy write + CPU in-place RoPE shift
            auto tk_start = std::chrono::high_resolution_clock::now();
            uint8_t* dst_k = reinterpret_cast<uint8_t*>(host_k_ptr);
            if (p + seq_len <= kv_size) {
                std::memcpy(dst_k + p * k_size_row, src_k_layer, seq_len * k_size_row);
                apply_relative_rope_shift(
                    reinterpret_cast<uint16_t*>(dst_k + p * k_size_row),
                    seq_len, n_head_kv, d_head, delta_pos,
                    header->rope_freq_base, header->rope_freq_scale, header->rope_scaling_type,
                    ext_factor, beta_fast, beta_slow, n_ctx_orig
                );
            } else {
                uint32_t first_part = kv_size - p;
                uint32_t second_part = seq_len - first_part;
                
                std::memcpy(dst_k + p * k_size_row, src_k_layer, first_part * k_size_row);
                apply_relative_rope_shift(
                    reinterpret_cast<uint16_t*>(dst_k + p * k_size_row),
                    first_part, n_head_kv, d_head, delta_pos,
                    header->rope_freq_base, header->rope_freq_scale, header->rope_scaling_type,
                    ext_factor, beta_fast, beta_slow, n_ctx_orig
                );

                std::memcpy(dst_k, src_k_layer + first_part * n_head_kv * d_head, second_part * k_size_row);
                apply_relative_rope_shift(
                    reinterpret_cast<uint16_t*>(dst_k),
                    second_part, n_head_kv, d_head, delta_pos,
                    header->rope_freq_base, header->rope_freq_scale, header->rope_scaling_type,
                    ext_factor, beta_fast, beta_slow, n_ctx_orig
                );
            }
            auto tk_end = std::chrono::high_resolution_clock::now();
            k_splice_us += std::chrono::duration<double, std::micro>(tk_end - tk_start).count();

            auto tv_start = std::chrono::high_resolution_clock::now();
            uint8_t* dst_v = reinterpret_cast<uint8_t*>(host_v_ptr);
            if (p + seq_len <= kv_size) {
                std::memcpy(dst_v + p * v_size_row, src_v_layer, seq_len * v_size_row);
            } else {
                uint32_t first_part = kv_size - p;
                uint32_t second_part = seq_len - first_part;
                std::memcpy(dst_v + p * v_size_row, src_v_layer, first_part * v_size_row);
                std::memcpy(dst_v, src_v_layer + first_part * n_head_kv * d_head, second_part * v_size_row);
            }
            auto tv_end = std::chrono::high_resolution_clock::now();
            v_splice_us += std::chrono::duration<double, std::micro>(tv_end - tv_start).count();
        } else {
            // Discrete Path (CUDA / PCIe): DMA raw unshifted memory immediately + GPU-native Relative RoPE Shift
            auto tk_start = std::chrono::high_resolution_clock::now();
            if (p + seq_len <= kv_size) {
                llama_tensor_set_async(ctx, k_tensor, src_k_layer, p * k_size_row, seq_len * k_size_row);
            } else {
                uint32_t first_part = kv_size - p;
                uint32_t second_part = seq_len - first_part;
                llama_tensor_set_async(ctx, k_tensor, src_k_layer, p * k_size_row, first_part * k_size_row);
                llama_tensor_set_async(ctx, k_tensor, src_k_layer + first_part * n_head_kv * d_head, 0, second_part * k_size_row);
            }
            auto tk_end = std::chrono::high_resolution_clock::now();
            k_splice_us += std::chrono::duration<double, std::micro>(tk_end - tk_start).count();

            auto tv_start = std::chrono::high_resolution_clock::now();
            if (p + seq_len <= kv_size) {
                llama_tensor_set_async(ctx, v_tensor, src_v_layer, p * v_size_row, seq_len * v_size_row);
            } else {
                uint32_t first_part = kv_size - p;
                uint32_t second_part = seq_len - first_part;
                llama_tensor_set_async(ctx, v_tensor, src_v_layer, p * v_size_row, first_part * v_size_row);
                llama_tensor_set_async(ctx, v_tensor, src_v_layer + first_part * n_head_kv * d_head, 0, second_part * v_size_row);
            }
            auto tv_end = std::chrono::high_resolution_clock::now();
            v_splice_us += std::chrono::duration<double, std::micro>(tv_end - tv_start).count();
        }
    }

    // Dispatch the RoPE shift compute graph to execute natively on the GPU
    if (n_layer > 0 && llama_kv_cache_get_k(ctx, 0)->data == nullptr) {
        if (p + seq_len <= kv_size) {
            llama_kv_cache_rope_shift_gpu(ctx, p, seq_len, delta_pos);
        } else {
            uint32_t first_part = kv_size - p;
            uint32_t second_part = seq_len - first_part;
            llama_kv_cache_rope_shift_gpu(ctx, p, first_part, delta_pos);
            llama_kv_cache_rope_shift_gpu(ctx, 0, second_part, delta_pos);
        }
    }

    auto t_splice_done = std::chrono::high_resolution_clock::now();

    // --- 3. Synchronize cell occupancy tracking inside llama.cpp ---
    for (uint32_t is = 0; is < seq_len; ++is) {
        uint32_t physical_cell_idx = (p + is) % kv_size;
        llama_pos logical_pos = static_cast<llama_pos>(n_past + is);
        llama_kv_cache_set_cell(ctx, physical_cell_idx, logical_pos, target_seq);
    }

    auto t_sync_done = std::chrono::high_resolution_clock::now();
    double sync_us = std::chrono::duration<double, std::micro>(t_sync_done - t_splice_done).count();
    double total_us = std::chrono::duration<double, std::micro>(t_sync_done - t_start).count();

    std::printf("  [Splicer Telemetry] Prep: %.2f us | K Splice: %.2f us | V Splice: %.2f us | Cell Sync: %.2f us | Total: %.2f us\n",
                prep_us, k_splice_us, v_splice_us, sync_us, total_us);
}

uint32_t NexusKVSplicer::unsplice_tool(llama_context* ctx, llama_seq_id seq_id, uint32_t splice_pos, uint32_t schema_len, uint32_t generated_len) {
    if (schema_len == 0) return 0;

    // 1. Erase the tool schema from KV Cache
    llama_pos p0_rm = static_cast<llama_pos>(splice_pos);
    llama_pos p1_rm = static_cast<llama_pos>(splice_pos + schema_len - 1);
    llama_kv_cache_seq_rm(ctx, seq_id, p0_rm, p1_rm);

    // 2. Shift downstream tokens (User Query + Generated Arguments) backward to splice_pos
    if (generated_len > 0) {
        llama_pos p0_shift = static_cast<llama_pos>(splice_pos + schema_len);
        llama_pos p1_shift = static_cast<llama_pos>(splice_pos + schema_len + generated_len - 1);
        llama_pos delta = -static_cast<llama_pos>(schema_len);
        llama_kv_cache_seq_shift(ctx, seq_id, p0_shift, p1_shift, delta);
    }

    // 3. Compact VRAM cells physically to eliminate holes and fragmentation (Lazy Defragmentation)
    int32_t used = llama_get_kv_cache_used_cells(ctx);
    int32_t max = static_cast<int32_t>(llama_n_ctx(ctx));
    if (max > 0 && (static_cast<float>(used) / max) > 0.85f) {
        llama_kv_cache_defrag(ctx);
    }

    return schema_len;
}
