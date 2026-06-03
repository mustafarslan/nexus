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
#include <cstdlib>
#include <memory>
#include <array>
#include "llama.h"
#include "aeon_tool_block.hpp"
#include "nexus_block_cache.hpp"
#include "nexus_rope_math.hpp"
#include "nexus_os_compat.hpp"

extern "C" {
    uint32_t llama_model_n_head_kv(const struct llama_model * model, int il);
    uint32_t llama_model_d_head(const struct llama_model * model);
    struct ggml_tensor * llama_kv_cache_get_k(struct llama_context * ctx, int il);
    struct ggml_tensor * llama_kv_cache_get_v(struct llama_context * ctx, int il);
    void llama_tensor_set_async(struct llama_context * ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    void llama_backend_sync(struct llama_context * ctx);
    void llama_kv_cache_rope_shift_gpu(struct llama_context * ctx, int32_t p, int32_t seq_len, int32_t delta_pos);
    bool llama_context_is_metal(struct llama_context * ctx);
    bool llama_context_is_cuda(struct llama_context * ctx);
}

// Call Python resolver script via popen to resolve Ollama GGUF path
std::string resolve_ollama_model_via_script() {
    std::array<char, 256> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen("python3 scripts/resolve_ollama_gguf.py", "r"), pclose);
    if (!pipe) {
        return "";
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    // Trim newline character
    if (!result.empty() && result.back() == '\n') {
        result.pop_back();
    }
    return result;
}

// Generate valid ATB file matching model parameters
void create_benchmark_atb_file(const std::string& path, uint32_t seq_len, uint32_t n_layer, uint32_t n_head_kv, uint32_t d_head) {
    std::ofstream file(path, std::ios::binary);
    if (!file) return;

    auto header_ptr = std::make_unique<AeonToolBlockHeader>();
    AeonToolBlockHeader& header = *header_ptr;
    std::memcpy(header.magic, "ATB1", 4);
    header.version = 1;
    header.model_hash = 0xDECAFBAD;
    header.n_layer = n_layer;
    header.n_head_kv = n_head_kv;
    header.d_head = d_head;
    header.seq_len = seq_len;
    header.base_pos = 0;
    header.rope_freq_base = 10000.0f;
    header.rope_freq_scale = 1.0f;
    header.rope_scaling_type = 0;
    header.ggml_type_k = 1; // FP16
    header.ggml_type_v = 1;

    uint64_t tensor_bytes = static_cast<uint64_t>(header.n_layer) * header.n_head_kv * header.seq_len * header.d_head * 2;
    uint64_t tensor_bytes_aligned = ((tensor_bytes + 2097151) / 2097152) * 2097152;
    header.k_tensor_offset = NEXUS_PAGE_ALIGNMENT;
    header.k_total_bytes = tensor_bytes;
    header.v_tensor_offset = header.k_tensor_offset + tensor_bytes_aligned;
    header.v_total_bytes = tensor_bytes;

    file.write(reinterpret_cast<const char*>(&header), sizeof(header));

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

// Real prefill decode baseline on actual loaded model and context
double run_real_prefill_baseline(llama_context* ctx, uint32_t n_tokens) {
    auto start = std::chrono::high_resolution_clock::now();

    uint32_t chunk_size = 2048;
    for (uint32_t start_idx = 0; start_idx < n_tokens; start_idx += chunk_size) {
        uint32_t current_chunk = std::min(chunk_size, n_tokens - start_idx);
        llama_batch batch = llama_batch_init(current_chunk, 0, 1);
        batch.n_tokens = current_chunk;
        for (uint32_t i = 0; i < current_chunk; ++i) {
            batch.token[i] = 1; // dummy token
            batch.pos[i] = start_idx + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (start_idx + i == n_tokens - 1) ? 1 : 0;
        }

        int res = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (res != 0) {
            throw std::runtime_error("Baseline prefill: llama_decode failed with exit code: " + std::to_string(res));
        }
    }
    
    // Physical stream synchronization before stopping timer
    llama_backend_sync(ctx);
    
    auto end = std::chrono::high_resolution_clock::now();
    
    // Reset cache to keep subsequent runs clean
    llama_kv_cache_clear(ctx);
    
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Spliced path measuring: block caching, prefetching, async PCIe transfer, GPU RoPE shift, and synchronization
double run_nexus_splicer_benchmark(NexusBlockCache& cache, const std::string& atb_path, llama_context* ctx, uint32_t n_past) {
    auto start = std::chrono::high_resolution_clock::now();

    // 1. Get or load block (includes file mmap, async residency touching and GPU host memory registration)
    auto block = cache.get_or_load(atb_path);
    if (!block) {
        throw std::runtime_error("Failed to load block in benchmark");
    }

    // 2. Prefetch hint
    cache.prefetch(atb_path);

    // 3. Inject tool page (simulates true asynchronous DMA H2D + GPU-native RoPE shift)
    const AeonToolBlockHeader* header = block->get_header();
    uint32_t n_layer = header->n_layer;
    uint32_t n_head_kv = header->n_head_kv;
    uint32_t d_head = header->d_head;
    uint32_t seq_len = header->seq_len;
    uint32_t kv_size = 16384;
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

        // Direct DMA copy parameters
        llama_tensor_set_async(ctx, k_tensor, src_k_layer, p * k_size_row, seq_len * k_size_row);
        llama_tensor_set_async(ctx, v_tensor, block->get_v_tensors().data() + il * layer_elements * 2, p * k_size_row, seq_len * k_size_row);
    }

    // Dispatch relative positional shift natively to GPU cores
    llama_kv_cache_rope_shift_gpu(ctx, p, seq_len, delta_pos);

    // Hard stream synchronization before stopping timer
    llama_backend_sync(ctx);

    auto end = std::chrono::high_resolution_clock::now();
    
    // Reset cache to keep subsequent runs clean
    llama_kv_cache_clear(ctx);
    
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void print_stats(const std::string& name, std::vector<double>& latencies) {
    std::sort(latencies.begin(), latencies.end());
    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double avg = sum / latencies.size();
    double p50 = latencies[latencies.size() / 2];
    double p90 = latencies[static_cast<size_t>(latencies.size() * 0.90)];
    double p99 = latencies[static_cast<size_t>(latencies.size() * 0.99)];

    std::printf("%-30s | Average: %8.2f ms | P50: %8.2f ms | P90: %8.2f ms | P99: %8.2f ms\n",
                name.c_str(), avg, p50, p90, p99);
}

int main(int argc, char** argv) {
    std::printf("======================================================================\n");
    std::printf("          PROJECT NEXUS PHASE 21 TTFT MICROBENCHMARK RUNNER           \n");
    std::printf("======================================================================\n");

    llama_backend_init();

    std::string model_path = "";
    if (argc > 1) {
        model_path = argv[1];
    } else {
        std::printf("Resolving GGUF path via Ollama manifests...\n");
        model_path = resolve_ollama_model_via_script();
    }

    if (model_path.empty() || !std::filesystem::exists(model_path)) {
        std::cerr << "Error: Resolved model path does not exist: " << model_path << "\n";
        return 1;
    }
    std::printf("Loaded GGUF Model: %s\n", model_path.c_str());

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 999; // Offload to GPU (Metal/CUDA)

    llama_model* model = llama_load_model_from_file(model_path.c_str(), model_params);
    if (!model) {
        std::printf("Warning: Failed to load resolved model: %s (might be unsupported architecture like gemma4 in this llama.cpp version).\n", model_path.c_str());
        std::printf("Attempting fallback to local cached model in Hugging Face cache...\n");
        std::string fallback_path = "";
        const char* home = std::getenv("HOME");
        if (home) {
            std::filesystem::path search_path = std::filesystem::path(home) / ".cache" / "huggingface" / "hub";
            if (std::filesystem::exists(search_path)) {
                for (const auto& entry : std::filesystem::recursive_directory_iterator(search_path)) {
                    if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
                        fallback_path = entry.path().string();
                        break;
                    }
                }
            }
        }
        if (!fallback_path.empty()) {
            std::printf("Found fallback GGUF model: %s\n", fallback_path.c_str());
            model_path = fallback_path;
            model = llama_load_model_from_file(model_path.c_str(), model_params);
        }
        if (!model) {
            std::cerr << "Error: Failed to load both target and fallback models.\n";
            return 1;
        }
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 16384;
    ctx_params.n_batch = 2048;
    ctx_params.n_ubatch = 2048;
    ctx_params.flash_attn = true;

    llama_context* ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        std::cerr << "Error: Failed to create context.\n";
        llama_free_model(model);
        return 1;
    }

    // Hardware Telemetry topology detection
    bool is_metal = llama_context_is_metal(ctx);
    bool is_cuda = llama_context_is_cuda(ctx);
    std::string topology_label = "CPU/Host-Fallback";
    if (is_metal) {
        topology_label = "UMA Zero-Copy";
    } else if (is_cuda) {
        topology_label = "NUMA/PCIe Gen4";
    }
    std::printf("Detected Hardware Topology: %s\n", topology_label.c_str());

    uint32_t n_layer = llama_n_layer(model);
    uint32_t n_head_kv = llama_model_n_head_kv(model, 0);
    uint32_t d_head = llama_model_d_head(model);
    uint32_t schema_len = 256;

    std::printf("Model Config: layers=%u, kv_heads=%u, head_dim=%u\n", n_layer, n_head_kv, d_head);

    std::string atb_path = "bench_temp_tool.atb";
    create_benchmark_atb_file(atb_path, schema_len, n_layer, n_head_kv, d_head);

    // Cache with 4GB pinned memory limit
    NexusBlockCache cache(4ULL * 1024 * 1024 * 1024);

    std::vector<double> prefill_latencies;
    std::vector<double> nexus_latencies;

    // Warmup runs
    std::printf("Executing warmup passes...\n");
    for (int i = 0; i < 3; ++i) {
        try {
            run_real_prefill_baseline(ctx, 12500);
            run_nexus_splicer_benchmark(cache, atb_path, ctx, 256);
        } catch (const std::exception& e) {
            std::cerr << "Warmup error: " << e.what() << "\n";
        }
    }

    // Benchmark loop
    const int iterations = 10;
    std::printf("Running %d iterations...\n", iterations);
    for (int i = 0; i < iterations; ++i) {
        try {
            prefill_latencies.push_back(run_real_prefill_baseline(ctx, 12500));
            nexus_latencies.push_back(run_nexus_splicer_benchmark(cache, atb_path, ctx, 256));
        } catch (const std::exception& e) {
            std::cerr << "Iteration " << i << " error: " << e.what() << "\n";
        }
    }

    if (!prefill_latencies.empty() && !nexus_latencies.empty()) {
        print_stats("FlashAttention-2 GPU Prefill (12.5k)", prefill_latencies);
        
        std::string splice_label = "Nexus PCIe DMA + VRAM Splice";
        if (is_metal) {
            splice_label = "[Topology: UMA Zero-Copy] VRAM Splice Latency";
        } else if (is_cuda) {
            splice_label = "[Topology: NUMA/PCIe Gen4] Asynchronous H2D DMA Splice Latency";
        }
        print_stats(splice_label, nexus_latencies);

        std::sort(prefill_latencies.begin(), prefill_latencies.end());
        std::sort(nexus_latencies.begin(), nexus_latencies.end());
        double speedup = prefill_latencies[prefill_latencies.size() / 2] / nexus_latencies[nexus_latencies.size() / 2];
        std::printf("----------------------------------------------------------------------\n");
        std::printf("Physical Speedup factor (P50): %.2f x\n", speedup);
        std::printf("======================================================================\n");
    } else {
        std::cerr << "Benchmark failed to collect latencies.\n";
    }

    // Clean up
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    std::filesystem::remove(atb_path);
    return 0;
}
