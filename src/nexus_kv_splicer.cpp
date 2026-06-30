#include "nexus_kv_splicer.hpp"
#include "nexus_rope_math.hpp"
#include <vector>
#include <cassert>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <latch>
#include "nexus_thread_pool.hpp"
#include "nexus_os_compat.hpp"

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
    void llama_kv_cache_rope_reanchor_gpu(struct llama_context * ctx, int32_t p, int32_t seq_len,
                                          int32_t compiled_base_pos, int32_t target_base_pos);
    bool llama_context_is_metal(struct llama_context * ctx);
    bool llama_context_is_cuda(struct llama_context * ctx);
}

#include <chrono>

// ─── Phase 71: RAII Fork-Join Synchronization ──────────────────────────────
class LatchGuard {
    std::latch* l_;
public:
    explicit LatchGuard(std::latch& l) : l_(&l) {}
    ~LatchGuard() { if (l_) l_->count_down(); }

    // Move-only semantics
    LatchGuard(const LatchGuard&) = delete;
    LatchGuard& operator=(const LatchGuard&) = delete;
    LatchGuard(LatchGuard&& other) noexcept : l_(other.l_) { other.l_ = nullptr; }
    LatchGuard& operator=(LatchGuard&& other) noexcept {
        if (this != &other) {
            if (l_) l_->count_down();
            l_ = other.l_;
            other.l_ = nullptr;
        }
        return *this;
    }
};

struct RoPEShiftTask {
    llama_context* ctx;
    const uint8_t* base_ptr;
    const AeonToolBlockHeader* header;
    uint32_t il_start;
    uint32_t il_end;
    int32_t p;
    int32_t delta_pos;
    uint32_t kv_size;
    float ext_factor;
    float beta_fast;
    float beta_slow;
    int n_ctx_orig;
    bool v_trans;
};

struct BlitLayerTask {
    llama_context* ctx;
    const uint16_t* src_k_base;
    const uint16_t* src_v_base;
    size_t layer_elements;
    uint32_t il_start;
    uint32_t il_end;
    int32_t p;
    uint32_t kv_size;
    uint32_t seq_len;
    uint32_t n_head_kv;
    uint32_t d_head;
    bool v_trans;
};

// Splice one layer's V rows into the live cache, handling both KV layouts.
//   v_trans == false (FlashAttention): V is row-major [n_tokens, n_head_kv, d_head];
//     the schema's seq_len rows are one contiguous blit at byte offset p*v_size_row.
//   v_trans == true  (soft-cap models, e.g. Gemma2, FlashAttn forced off): V is
//     transposed [n_head_kv, d_head, n_tokens] with the token axis innermost
//     (stride 1) and a kv_size-length row per (head,dim) channel. The .atb is packed
//     transposed by the compiler as src[j*seq_len + is] for channel j = ih*d_head+id
//     (nexus_kv_compiler.cpp:436-443); the live channel base element is j*kv_size, so
//     each of the n_head_kv*d_head channels is copied independently with its own
//     ring-buffer wrap. K is unaffected by v_trans (it is never transposed).
static void splice_v_layer(uint8_t* dst_v, const uint16_t* src_v_layer,
                           int32_t p, uint32_t seq_len, uint32_t kv_size,
                           uint32_t n_head_kv, uint32_t d_head, bool v_trans) {
    const size_t v_size_row = static_cast<size_t>(n_head_kv) * d_head * sizeof(uint16_t);
    if (!v_trans) {
        if (static_cast<uint32_t>(p) + seq_len <= kv_size) {
            std::memcpy(dst_v + p * v_size_row, src_v_layer, seq_len * v_size_row);
        } else {
            uint32_t first_part = kv_size - p;
            uint32_t second_part = seq_len - first_part;
            std::memcpy(dst_v + p * v_size_row, src_v_layer, first_part * v_size_row);
            std::memcpy(dst_v, src_v_layer + static_cast<size_t>(first_part) * n_head_kv * d_head,
                        second_part * v_size_row);
        }
        return;
    }

    // Transposed: per-channel contiguous copy along the token axis.
    uint16_t* dst_u16 = reinterpret_cast<uint16_t*>(dst_v);
    const size_t n_chan = static_cast<size_t>(n_head_kv) * d_head;
    const bool no_wrap = (static_cast<uint32_t>(p) + seq_len <= kv_size);
    const uint32_t first_part = no_wrap ? seq_len : (kv_size - static_cast<uint32_t>(p));
    const uint32_t second_part = no_wrap ? 0u : (seq_len - first_part);
    for (size_t j = 0; j < n_chan; ++j) {
        const uint16_t* src_ch = src_v_layer + j * seq_len;
        uint16_t* dst_ch = dst_u16 + j * kv_size;
        std::memcpy(dst_ch + p, src_ch, first_part * sizeof(uint16_t));
        if (second_part) {
            std::memcpy(dst_ch, src_ch + first_part, second_part * sizeof(uint16_t));
        }
    }
}

static void blit_kv_layer_range(BlitLayerTask* task, uint32_t il,
                                double* k_us, double* v_us) {
    size_t k_size_row = task->n_head_kv * task->d_head * sizeof(uint16_t);
    struct ggml_tensor* k_tensor = llama_kv_cache_get_k(task->ctx, il);
    struct ggml_tensor* v_tensor = llama_kv_cache_get_v(task->ctx, il);
    if (!k_tensor || !v_tensor || k_tensor->data == nullptr) {
        return;
    }
    const uint16_t* src_k_layer = task->src_k_base + il * task->layer_elements;
    const uint16_t* src_v_layer = task->src_v_base + il * task->layer_elements;
    uint8_t* dst_k = reinterpret_cast<uint8_t*>(k_tensor->data);
    uint8_t* dst_v = reinterpret_cast<uint8_t*>(v_tensor->data);

    auto tk_start = std::chrono::high_resolution_clock::now();
    if (task->p + task->seq_len <= task->kv_size) {
        std::memcpy(dst_k + task->p * k_size_row, src_k_layer, task->seq_len * k_size_row);
    } else {
        uint32_t first_part = task->kv_size - task->p;
        uint32_t second_part = task->seq_len - first_part;
        std::memcpy(dst_k + task->p * k_size_row, src_k_layer, first_part * k_size_row);
        std::memcpy(dst_k, src_k_layer + first_part * task->n_head_kv * task->d_head, second_part * k_size_row);
    }
    auto tk_end = std::chrono::high_resolution_clock::now();
    *k_us += std::chrono::duration<double, std::micro>(tk_end - tk_start).count();

    auto tv_start = std::chrono::high_resolution_clock::now();
    splice_v_layer(dst_v, src_v_layer, task->p, task->seq_len, task->kv_size,
                   task->n_head_kv, task->d_head, task->v_trans);
    auto tv_end = std::chrono::high_resolution_clock::now();
    *v_us += std::chrono::duration<double, std::micro>(tv_end - tv_start).count();
}

static void parallel_blit_worker(BlitLayerTask* task) {
    for (uint32_t il = task->il_start; il < task->il_end; ++il) {
        double k_us = 0.0;
        double v_us = 0.0;
        blit_kv_layer_range(task, il, &k_us, &v_us);
    }
}

static void parallel_rope_shift_worker(RoPEShiftTask* task) {
    uint32_t n_head_kv = task->header->n_head_kv;
    uint32_t d_head = task->header->d_head;
    uint32_t seq_len = task->header->seq_len;
    size_t layer_elements = n_head_kv * seq_len * d_head;
    size_t k_size_row = n_head_kv * d_head * sizeof(uint16_t);
    const uint16_t* src_k_base = reinterpret_cast<const uint16_t*>(task->base_ptr + task->header->k_tensor_offset);
    const uint16_t* src_v_base = reinterpret_cast<const uint16_t*>(task->base_ptr + task->header->v_tensor_offset);

    for (uint32_t il = task->il_start; il < task->il_end; ++il) {
        struct ggml_tensor * k_tensor = llama_kv_cache_get_k(task->ctx, il);
        struct ggml_tensor * v_tensor = llama_kv_cache_get_v(task->ctx, il);
        
        const uint16_t* src_k_layer = src_k_base + il * layer_elements;
        const uint16_t* src_v_layer = src_v_base + il * layer_elements;
        
        void* host_k_ptr = k_tensor->data;
        void* host_v_ptr = v_tensor->data;
        
        uint8_t* dst_k = reinterpret_cast<uint8_t*>(host_k_ptr);
        uint8_t* dst_v = reinterpret_cast<uint8_t*>(host_v_ptr);
        
        const uint32_t compiled_base = task->header->base_pos;
        const uint32_t target_base = static_cast<uint32_t>(static_cast<int32_t>(compiled_base) + task->delta_pos);

        // Copy and RoPE shift K
        if (task->p + seq_len <= task->kv_size) {
            std::memcpy(dst_k + task->p * k_size_row, src_k_layer, seq_len * k_size_row);
            apply_absolute_rope_reanchor(
                reinterpret_cast<uint16_t*>(dst_k + task->p * k_size_row),
                seq_len, n_head_kv, d_head, compiled_base, target_base,
                task->header->rope_freq_base, task->header->rope_freq_scale, task->header->rope_scaling_type,
                task->ext_factor, task->beta_fast, task->beta_slow, task->n_ctx_orig
            );
        } else {
            uint32_t first_part = task->kv_size - task->p;
            uint32_t second_part = seq_len - first_part;

            std::memcpy(dst_k + task->p * k_size_row, src_k_layer, first_part * k_size_row);
            apply_absolute_rope_reanchor(
                reinterpret_cast<uint16_t*>(dst_k + task->p * k_size_row),
                first_part, n_head_kv, d_head, compiled_base, target_base,
                task->header->rope_freq_base, task->header->rope_freq_scale, task->header->rope_scaling_type,
                task->ext_factor, task->beta_fast, task->beta_slow, task->n_ctx_orig
            );

            std::memcpy(dst_k, src_k_layer + first_part * n_head_kv * d_head, second_part * k_size_row);
            apply_absolute_rope_reanchor(
                reinterpret_cast<uint16_t*>(dst_k),
                second_part, n_head_kv, d_head,
                compiled_base + first_part, target_base + first_part,
                task->header->rope_freq_base, task->header->rope_freq_scale, task->header->rope_scaling_type,
                task->ext_factor, task->beta_fast, task->beta_slow, task->n_ctx_orig
            );
        }
        
        // Copy V (RoPE never applies to V; layout follows the live cache's v_trans).
        splice_v_layer(dst_v, src_v_layer, task->p, seq_len, task->kv_size,
                       n_head_kv, d_head, task->v_trans);
    }
}

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
    // v_trans == true when FlashAttention is off (soft-cap models such as Gemma2 force it
    // off). The V cache is then stored transposed; splice_v_layer() handles that layout,
    // so we no longer reject it here — the per-layout stride check happens in the validator.
    bool v_trans = llama_kv_cache_get_v_trans(ctx);

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

    // Phase 34: Validate ggml KV cache tensor memory layout is contiguous.
    // K tensor layout: [head_dim, n_heads, n_tokens] where head_dim is innermost.
    // Expected strides: nb[0] = type_size, nb[1] = d_head * type_size, nb[2] = n_head_kv * nb[1]
    // This validation ensures our single-blit memcpy/DMA is correct.
    {
        struct ggml_tensor* k0 = llama_kv_cache_get_k(ctx, 0);
        struct ggml_tensor* v0 = llama_kv_cache_get_v(ctx, 0);
        if (k0 && v0) {
            size_t type_size = ggml_type_size(k0->type);
            bool k_contiguous = false;
            if (k0->ne[1] == 1 && k0->ne[2] == 1) {
                k_contiguous = (k0->nb[0] == type_size);
            } else {
                k_contiguous = (k0->nb[0] == type_size)
                             && (k0->nb[1] == d_head * type_size)
                             && (k0->nb[2] == static_cast<size_t>(n_head_kv) * d_head * type_size);
            }
            if (!k_contiguous) {
                throw std::runtime_error(
                    "NexusKVSplicer: K tensor layout is non-contiguous. "
                    "nb[0]=" + std::to_string(k0->nb[0]) +
                    " nb[1]=" + std::to_string(k0->nb[1]) +
                    " nb[2]=" + std::to_string(k0->nb[2]) +
                    " expected nb[1]=" + std::to_string(d_head * type_size) +
                    " nb[2]=" + std::to_string(static_cast<size_t>(n_head_kv) * d_head * type_size));
            }
            // V validation depends on layout. Non-transposed: token-major rows like K
            // (nb[1]=d_head*ts, nb[2]=n_head_kv*d_head*ts). Transposed (v_trans): token axis
            // innermost over kv_size, channel stride kv_size (nb[1]=kv_size*ts,
            // nb[2]=kv_size*d_head*ts). splice_v_layer() relies on exactly these strides.
            const size_t exp_v_nb1 = v_trans ? (static_cast<size_t>(kv_size) * type_size)
                                             : (static_cast<size_t>(d_head) * type_size);
            const size_t exp_v_nb2 = v_trans ? (static_cast<size_t>(kv_size) * d_head * type_size)
                                             : (static_cast<size_t>(n_head_kv) * d_head * type_size);
            bool v_contiguous = false;
            if (v0->ne[1] == 1 && v0->ne[2] == 1) {
                v_contiguous = (v0->nb[0] == type_size);
            } else {
                v_contiguous = (v0->nb[0] == type_size)
                             && (v0->nb[1] == exp_v_nb1)
                             && (v0->nb[2] == exp_v_nb2);
            }
            if (!v_contiguous) {
                throw std::runtime_error(
                    "NexusKVSplicer: V tensor layout mismatch (v_trans=" + std::to_string(v_trans) + "). "
                    "nb[0]=" + std::to_string(v0->nb[0]) +
                    " nb[1]=" + std::to_string(v0->nb[1]) +
                    " nb[2]=" + std::to_string(v0->nb[2]) +
                    " expected nb[1]=" + std::to_string(exp_v_nb1) +
                    " nb[2]=" + std::to_string(exp_v_nb2));
            }
        }
    }

    // Retrieve YaRN parameters from context (only needed when delta_pos != 0)
    float ext_factor = llama_context_yarn_ext_factor(ctx);
    float beta_fast = llama_context_yarn_beta_fast(ctx);
    float beta_slow = llama_context_yarn_beta_slow(ctx);
    int n_ctx_orig = static_cast<int>(llama_context_yarn_n_ctx_orig(ctx));

    bool is_uma = false;
    if (n_layer > 0) {
        struct ggml_tensor* k0 = llama_kv_cache_get_k(ctx, 0);
        if (k0 && k0->data != nullptr) {
            is_uma = true;
        }
    }

    const bool use_gpu_rope = llama_context_is_metal(ctx) || llama_context_is_cuda(ctx);

    if (is_uma && delta_pos != 0 && !use_gpu_rope) {
        auto tk_start = std::chrono::high_resolution_clock::now();
        int target_node = discover_gpu_numa_node();
        if (target_node < 0) target_node = 0;
        NexusThreadPool& pool = get_numa_pool_manager().get_pool(target_node);
        
        uint32_t num_tasks = 4;
        if (n_layer < num_tasks) {
            num_tasks = n_layer;
        }
        
        std::latch latch(num_tasks);
        std::vector<RoPEShiftTask> tasks(num_tasks);
        uint32_t chunk_size = (n_layer + num_tasks - 1) / num_tasks;
        
        // Populate all tasks
        for (uint32_t t = 0; t < num_tasks; ++t) {
            tasks[t].ctx = ctx;
            tasks[t].base_ptr = base_ptr;
            tasks[t].header = header;
            tasks[t].il_start = t * chunk_size;
            tasks[t].il_end = std::min(tasks[t].il_start + chunk_size, n_layer);
            tasks[t].p = p;
            tasks[t].delta_pos = delta_pos;
            tasks[t].kv_size = kv_size;
            tasks[t].ext_factor = ext_factor;
            tasks[t].beta_fast = beta_fast;
            tasks[t].beta_slow = beta_slow;
            tasks[t].n_ctx_orig = n_ctx_orig;
            tasks[t].v_trans = v_trans;
        }

        // Phase 71: RAII Fork-Join Synchronization.
        // Bind the std::latch decrement to the lambda's destruction lifecycle (LatchGuard),
        // guaranteeing the main thread resumes regardless of execution state, exceptions, or queue flushes.
        for (uint32_t t = 0; t + 1 < num_tasks; ++t) {
            if (!pool.submit([guard = LatchGuard(latch), &task_ref = tasks[t]]() {
                parallel_rope_shift_worker(&task_ref);
                // NO manual count_down() here. The guard's destructor handles it.
            })) {
                // Inline fallback if submission is rejected.
                // The submitted lambda is destroyed upon rejection, decrementing the latch via guard's destructor.
                // Thus, we run the worker inline without another LatchGuard to avoid double decrement.
                parallel_rope_shift_worker(&tasks[t]);
            }
        }

        // Execute the last task inline on the main calling thread
        {
            LatchGuard inline_guard(latch);
            parallel_rope_shift_worker(&tasks[num_tasks - 1]);
        }

        latch.wait();
        
        auto tk_end = std::chrono::high_resolution_clock::now();
        // parallel_rope_shift_worker copies K (with RoPE) and V; attribute total to both for honest telemetry
        double kv_parallel_us = std::chrono::duration<double, std::micro>(tk_end - tk_start).count();
        k_splice_us += kv_parallel_us * 0.5;
        v_splice_us += kv_parallel_us * 0.5;
    } else if (is_uma) {
        // UMA zero-delta: parallel layer blit (no RoPE shift needed)
        auto t_blit_start = std::chrono::high_resolution_clock::now();
        int target_node = discover_gpu_numa_node();
        if (target_node < 0) target_node = 0;
        NexusThreadPool& pool = get_numa_pool_manager().get_pool(target_node);

        uint32_t num_tasks = 4;
        if (n_layer < num_tasks) {
            num_tasks = n_layer;
        }
        std::latch latch(num_tasks);
        std::vector<BlitLayerTask> blit_tasks(num_tasks);
        uint32_t chunk_size = (n_layer + num_tasks - 1) / num_tasks;

        for (uint32_t t = 0; t < num_tasks; ++t) {
            blit_tasks[t].ctx = ctx;
            blit_tasks[t].src_k_base = src_k_base;
            blit_tasks[t].src_v_base = src_v_base;
            blit_tasks[t].layer_elements = layer_elements;
            blit_tasks[t].il_start = t * chunk_size;
            blit_tasks[t].il_end = std::min(blit_tasks[t].il_start + chunk_size, n_layer);
            blit_tasks[t].p = p;
            blit_tasks[t].kv_size = kv_size;
            blit_tasks[t].seq_len = seq_len;
            blit_tasks[t].n_head_kv = n_head_kv;
            blit_tasks[t].d_head = d_head;
            blit_tasks[t].v_trans = v_trans;
        }

        for (uint32_t t = 0; t + 1 < num_tasks; ++t) {
            if (!pool.submit([guard = LatchGuard(latch), &task_ref = blit_tasks[t]]() {
                parallel_blit_worker(&task_ref);
            })) {
                parallel_blit_worker(&blit_tasks[t]);
            }
        }
        {
            LatchGuard inline_guard(latch);
            parallel_blit_worker(&blit_tasks[num_tasks - 1]);
        }
        latch.wait();

        auto t_blit_end = std::chrono::high_resolution_clock::now();
        double kv_blit_us = std::chrono::duration<double, std::micro>(t_blit_end - t_blit_start).count();
        k_splice_us += kv_blit_us * 0.5;
        v_splice_us += kv_blit_us * 0.5;
    } else {
        // Discrete-GPU (non-UMA) async DMA path. The transposed-V (v_trans) layout would
        // need n_head_kv*d_head strided async sets per layer; that combination (soft-cap
        // model on a discrete GPU) is not exercised on this UMA/Metal host, so it is
        // explicitly rejected here rather than shipped unvalidated. The UMA path above
        // fully supports v_trans.
        if (v_trans) {
            throw std::runtime_error(
                "NexusKVSplicer: transposed-V (v_trans) splice on the discrete-GPU async "
                "path is not yet supported; run with unified memory (Metal/UMA).");
        }
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

    // GPU RoPE re-anchor: discrete GPU (host ptr null) or UMA+Metal/CUDA (avoid double CPU+GPU rotation)
    if (delta_pos != 0 && n_layer > 0 && (use_gpu_rope || !is_uma)) {
        const int32_t compiled_base = static_cast<int32_t>(header->base_pos);
        const int32_t target_base = static_cast<int32_t>(n_past);
        if (p + seq_len <= kv_size) {
            llama_kv_cache_rope_reanchor_gpu(ctx, p, static_cast<int32_t>(seq_len), compiled_base, target_base);
        } else {
            uint32_t first_part = kv_size - static_cast<uint32_t>(p);
            uint32_t second_part = seq_len - first_part;
            llama_kv_cache_rope_reanchor_gpu(ctx, p, static_cast<int32_t>(first_part), compiled_base, target_base);
            llama_kv_cache_rope_reanchor_gpu(
                ctx, 0, static_cast<int32_t>(second_part),
                compiled_base + static_cast<int32_t>(first_part),
                target_base + static_cast<int32_t>(first_part));
        }
    }

    auto t_splice_done = std::chrono::high_resolution_clock::now();

    // --- 3. Synchronize cell occupancy tracking inside llama.cpp ---
    for (uint32_t is = 0; is < seq_len; ++is) {
        uint32_t physical_cell_idx = (p + is) % kv_size;
        llama_pos logical_pos = static_cast<llama_pos>(n_past + is);
        llama_kv_cache_set_cell(ctx, physical_cell_idx, logical_pos, target_seq);
    }

    if (delta_pos > 0 && header->base_pos > 0) {
        const double depth_ratio = std::log(
            static_cast<double>(n_past) / std::max(1.0, static_cast<double>(header->base_pos)));
        std::vector<float> v_scales(seq_len, 1.0f);
        for (uint32_t is = 0; is < seq_len; ++is) {
            v_scales[is] = static_cast<float>(std::exp(-0.02 * depth_ratio));
        }
        llama_kv_cache_apply_attention_bias(
            ctx, -1,
            static_cast<llama_pos>(n_past),
            static_cast<llama_pos>(n_past + seq_len),
            v_scales.data(),
            static_cast<int32_t>(seq_len));
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

std::vector<float> NexusKVSplicer::read_kv_slice(
        llama_context* ctx,
        int il,
        int32_t p0,
        int32_t p1,
        bool read_v) {
    if (!ctx) {
        throw std::runtime_error("read_kv_slice: null context");
    }
    if (il < 0) {
        throw std::runtime_error("read_kv_slice: layer index must be non-negative");
    }
    if (p0 < 0 || p1 <= p0) {
        throw std::runtime_error("read_kv_slice: invalid position range [p0, p1)");
    }

    const struct llama_model* model = llama_get_model(ctx);
    uint32_t n_head_kv = llama_model_n_head_kv(model, il);
    uint32_t d_head = llama_model_d_head(model);
    uint32_t seq_len = static_cast<uint32_t>(p1 - p0);
    size_t row_elements = static_cast<size_t>(n_head_kv) * d_head;
    size_t slice_elements = row_elements * seq_len;

    struct ggml_tensor* tensor = read_v
        ? llama_kv_cache_get_v(ctx, il)
        : llama_kv_cache_get_k(ctx, il);
    if (!tensor) {
        throw std::runtime_error("read_kv_slice: failed to retrieve KV tensor for layer " + std::to_string(il));
    }

    std::vector<uint16_t> fp16_buf(slice_elements);
    size_t byte_offset = static_cast<size_t>(p0) * row_elements * sizeof(uint16_t);
    size_t byte_size = slice_elements * sizeof(uint16_t);

    if (tensor->data != nullptr) {
        const uint8_t* src = reinterpret_cast<const uint8_t*>(tensor->data) + byte_offset;
        std::memcpy(fp16_buf.data(), src, byte_size);
    } else {
        ggml_backend_tensor_get(tensor, fp16_buf.data(), byte_offset, byte_size);
        llama_backend_sync(ctx);
    }

    std::vector<float> out(slice_elements);
    for (size_t i = 0; i < slice_elements; ++i) {
        out[i] = ggml_fp16_to_fp32(fp16_buf[i]);
    }
    return out;
}
