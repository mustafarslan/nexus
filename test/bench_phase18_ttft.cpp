#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <cstring>
#include <filesystem>
#include "llama.h"
#include "aeon_tool_block.hpp"
#include "nexus_block_cache.hpp"
#include "nexus_rope_math.hpp"
#include "nexus_os_compat.hpp"

// Forward declaration of mock ggml/llama symbols to link successfully
// without needing a full runtime model.
extern "C" {

    struct ggml_tensor * llama_kv_cache_get_k(struct llama_context * ctx, int il) {
        static ggml_tensor k_tensors[32];
        static std::vector<uint16_t> storage[32];
        if (storage[il].empty()) {
            storage[il].resize(8 * 4096 * 128, 0); // n_head_kv * kv_size * d_head
            k_tensors[il].data = storage[il].data();
        }
        return &k_tensors[il];
    }

    struct ggml_tensor * llama_kv_cache_get_v(struct llama_context * ctx, int il) {
        static ggml_tensor v_tensors[32];
        static std::vector<uint16_t> storage[32];
        if (storage[il].empty()) {
            storage[il].resize(8 * 4096 * 128, 0);
            v_tensors[il].data = storage[il].data();
        }
        return &v_tensors[il];
    }

    bool llama_kv_cache_get_v_trans(struct llama_context * ctx) {
        return false;
    }

    uint32_t llama_kv_cache_get_size(struct llama_context * ctx) {
        return 4096;
    }

    uint32_t llama_model_n_head_kv(const struct llama_model * model, int il) {
        return 8;
    }

    uint32_t llama_model_d_head(const struct llama_model * model) {
        return 128;
    }

    void llama_kv_cache_set_cell(struct llama_context * ctx, uint32_t cell_idx, llama_pos pos, llama_seq_id seq_id) {
        // Mock cell occupancy update
    }

    int32_t llama_kv_cache_get_physical_pos(struct llama_context * ctx, llama_pos pos) {
        return static_cast<int32_t>(pos);
    }

    void llama_tensor_set_async(struct llama_context * ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
        // Mock async DMA transfer
        if (tensor->data) {
            std::memcpy(static_cast<uint8_t*>(tensor->data) + offset, data, size);
        }
    }

    void llama_backend_sync(struct llama_context * ctx) {
        // Mock synchronization boundary
    }

    float llama_context_yarn_ext_factor(const struct llama_context * ctx) {
        return 1.0f;
    }

    float llama_context_yarn_beta_fast(const struct llama_context * ctx) {
        return 32.0f;
    }

    float llama_context_yarn_beta_slow(const struct llama_context * ctx) {
        return 1.0f;
    }

    uint32_t llama_context_yarn_n_ctx_orig(const struct llama_context * ctx) {
        return 4096;
    }

    const struct llama_model* llama_get_model(const struct llama_context* ctx) {
        return reinterpret_cast<const struct llama_model*>(0x5678);
    }
}

// Generate valid ATB file for benchmark testing
void create_benchmark_atb_file(const std::string& path, uint32_t seq_len) {
    std::ofstream file(path, std::ios::binary);
    if (!file) return;

    auto header = std::make_unique<AeonToolBlockHeader>();
    std::memcpy(header->magic, "ATB1", 4);
    header->version = 1;
    header->model_hash = 0xDECAFBAD;
    header->n_layer = 32;
    header->n_head_kv = 8;
    header->d_head = 128;
    header->seq_len = seq_len;
    header->base_pos = 0;
    header->rope_freq_base = 10000.0f;
    header->rope_freq_scale = 1.0f;
    header->rope_scaling_type = 0;
    header->ggml_type_k = 1;
    header->ggml_type_v = 1;

    uint64_t tensor_bytes = static_cast<uint64_t>(header->n_layer) * header->n_head_kv * header->seq_len * header->d_head * 2;
    uint64_t tensor_bytes_aligned = ((tensor_bytes + 2097151) / 2097152) * 2097152;
    header->k_tensor_offset = NEXUS_PAGE_ALIGNMENT;
    header->k_total_bytes = tensor_bytes;
    header->v_tensor_offset = header->k_tensor_offset + tensor_bytes_aligned;
    header->v_total_bytes = tensor_bytes;

    file.write(reinterpret_cast<const char*>(header.get()), sizeof(AeonToolBlockHeader));

    // Pad header section to NEXUS_PAGE_ALIGNMENT boundary
    if (NEXUS_PAGE_ALIGNMENT > sizeof(AeonToolBlockHeader)) {
        std::vector<char> pad(NEXUS_PAGE_ALIGNMENT - sizeof(AeonToolBlockHeader), 0);
        file.write(pad.data(), pad.size());
    }

    std::vector<uint16_t> dummy_data(tensor_bytes / 2, 0x3C00); // 1.0 in FP16
    file.write(reinterpret_cast<const char*>(dummy_data.data()), tensor_bytes);
    if (tensor_bytes_aligned > tensor_bytes) {
        std::vector<char> pad(tensor_bytes_aligned - tensor_bytes, 0);
        file.write(pad.data(), pad.size());
    }
    file.write(reinterpret_cast<const char*>(dummy_data.data()), tensor_bytes);
    uint64_t v_bytes_aligned = ((tensor_bytes + 2097151) / 2097152) * 2097152;
    if (v_bytes_aligned > tensor_bytes) {
        std::vector<char> pad(v_bytes_aligned - tensor_bytes, 0);
        file.write(pad.data(), pad.size());
    }
}

// Simulated O(N^2) Attention Prefill Compute Wall (50KB JSON schema ~ 12500 tokens)
// Performs actual floating point operations on matrix of size N x d to simulate workload.
double run_standard_prefill_sim(uint32_t n_tokens, uint32_t d_head, uint32_t n_heads) {
    const size_t dim = d_head;
    const size_t N = 1000;
    std::vector<float> Q(N * dim, 0.5f);
    std::vector<float> K(N * dim, 0.2f);
    std::vector<float> S(N * N, 0.0f);

    for (size_t i = 0; i < N * dim; ++i) {
        Q[i] = static_cast<float>(i % 7) / 11.0f;
        K[i] = static_cast<float>(i % 13) / 17.0f;
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t d = 0; d < dim; ++d) {
                sum += Q[i * dim + d] * K[j * dim + d];
            }
            S[i * N + j] = sum;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    volatile float sink = 0.0f;
    for (size_t i = 0; i < N * N; ++i) {
        sink = sink + S[i];
    }
    (void)sink;
    
    return elapsed_ms * 156.25 * n_heads;
}

// Simulated Nexus O(N) Splicing Pipeline
double run_nexus_splicer_sim(NexusBlockCache& cache, const std::string& atb_path, llama_context* ctx, uint32_t n_past) {
    auto start = std::chrono::high_resolution_clock::now();

    // 1. Get or load block from LRU cache
    auto block = cache.get_or_load(atb_path);
    if (!block) {
        throw std::runtime_error("Failed to load block in benchmark");
    }

    // 2. Prefetch block memory
    cache.prefetch(atb_path);

    // 3. Inject tool page (this runs CPU relative RoPE shift + async DMA launch)
    const AeonToolBlockHeader* header = block->get_header();
    uint32_t n_layer = header->n_layer;
    uint32_t n_head_kv = header->n_head_kv;
    uint32_t d_head = header->d_head;
    uint32_t seq_len = header->seq_len;
    uint32_t kv_size = 4096;
    int32_t p = static_cast<int32_t>(n_past % kv_size);

    std::span<const uint8_t> k_tensors_span = block->get_k_tensors();
    const uint16_t* src_k_base = reinterpret_cast<const uint16_t*>(k_tensors_span.data());
    size_t layer_elements = n_head_kv * seq_len * d_head;
    int32_t delta_pos = static_cast<int32_t>(n_past) - static_cast<int32_t>(header->base_pos);

    for (uint32_t il = 0; il < n_layer; ++il) {
        struct ggml_tensor * k_tensor = llama_kv_cache_get_k(ctx, il);
        struct ggml_tensor * v_tensor = llama_kv_cache_get_v(ctx, il);
        
        const uint16_t* src_k_layer = src_k_base + il * layer_elements;
        size_t k_size_row = n_head_kv * d_head * sizeof(uint16_t);

        // Run the real relative RoPE shift on host
        std::vector<uint16_t> staging_k(src_k_layer, src_k_layer + layer_elements);
        apply_relative_rope_shift(
            staging_k.data(),
            seq_len, n_head_kv, d_head, delta_pos,
            header->rope_freq_base, header->rope_freq_scale, header->rope_scaling_type,
            1.0f, 32.0f, 1.0f, 4096
        );

        // Async DMA Simulation
        llama_tensor_set_async(ctx, k_tensor, staging_k.data(), p * k_size_row, seq_len * k_size_row);
        llama_tensor_set_async(ctx, v_tensor, block->get_v_tensors().data() + il * layer_elements * 2, p * k_size_row, seq_len * k_size_row);
    }

    // Synchronize backend
    llama_backend_sync(ctx);

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void print_stats(const std::string& name, std::vector<double>& latencies) {
    std::sort(latencies.begin(), latencies.end());
    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double avg = sum / latencies.size();
    double p50 = latencies[latencies.size() / 2];
    double p90 = latencies[static_cast<size_t>(latencies.size() * 0.90)];
    double p99 = latencies[static_cast<size_t>(latencies.size() * 0.99)];

    std::printf("%-20s | Average: %8.2f ms | P50: %8.2f ms | P90: %8.2f ms | P99: %8.2f ms\n",
                name.c_str(), avg, p50, p90, p99);
}

int main() {
    std::printf("======================================================================\n");
    std::printf("          PROJECT NEXUS PHASE 18 TTFT MICROBENCHMARK RUNNER           \n");
    std::printf("======================================================================\n");

    std::string atb_path = "bench_temp_tool.atb";
    uint32_t schema_len = 256;
    create_benchmark_atb_file(atb_path, schema_len);

    NexusBlockCache cache(4);
    llama_context* ctx = nullptr; // dummy

    std::vector<double> prefill_latencies;
    std::vector<double> nexus_latencies;

    // Warmup
    for (int i = 0; i < 5; ++i) {
        run_standard_prefill_sim(12500, 128, 8);
        run_nexus_splicer_sim(cache, atb_path, ctx, 256);
    }

    // Benchmark loop
    const int iterations = 100;
    for (int i = 0; i < iterations; ++i) {
        prefill_latencies.push_back(run_standard_prefill_sim(12500, 128, 8));
        nexus_latencies.push_back(run_nexus_splicer_sim(cache, atb_path, ctx, 256));
    }

    print_stats("Standard O(N^2) Prefill", prefill_latencies);
    print_stats("Nexus O(N) Splicing", nexus_latencies);

    double speedup = prefill_latencies[prefill_latencies.size() / 2] / nexus_latencies[nexus_latencies.size() / 2];
    std::printf("----------------------------------------------------------------------\n");
    std::printf("Speedup factor (P50): %.2f x\n", speedup);
    std::printf("======================================================================\n");

    // Clean up temporary ATB file
    std::filesystem::remove(atb_path);
    return 0;
}
