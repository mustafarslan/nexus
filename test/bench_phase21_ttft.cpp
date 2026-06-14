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
#include "nexus_slb.hpp"
#include "nexus_fsm.hpp"
#include "nexus_kv_splicer.hpp"
#include "nexus_benchmark_results.hpp"

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

double run_nexus_splicer_benchmark(NexusBlockCache& cache, const std::string& atb_path, llama_context* ctx, uint32_t n_past) {
    auto start = std::chrono::high_resolution_clock::now();

    auto res = cache.get_or_load(atb_path);
    if (!res.has_value()) {
        throw std::runtime_error("Failed to load block in benchmark: " + atb_path);
    }
    std::shared_ptr<AeonToolBlock> block = res.value();
    cache.prefetch(atb_path);
    NexusKVSplicer::inject_tool_page(ctx, block, n_past, 0);
    llama_backend_sync(ctx);

    auto end = std::chrono::high_resolution_clock::now();
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

    std::printf("%-50s | Average: %8.2f ms | P50: %8.2f ms | P90: %8.2f ms | P99: %8.2f ms\n",
                name.c_str(), avg, p50, p90, p99);
}

void print_stats_us(const std::string& name, std::vector<double>& latencies) {
    std::sort(latencies.begin(), latencies.end());
    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double avg = sum / latencies.size();
    double p50 = latencies[latencies.size() / 2];
    double p90 = latencies[static_cast<size_t>(latencies.size() * 0.90)];
    double p99 = latencies[static_cast<size_t>(latencies.size() * 0.99)];

    std::printf("%-50s | Average: %8.2f µs | P50: %8.2f µs | P90: %8.2f µs | P99: %8.2f µs\n",
                name.c_str(), avg, p50, p90, p99);
}

struct CliArgs {
    std::string model_path;
    std::string output_path = "results/bench_phase21_ttft.json";
    std::string atb_path = "results/phaseA_tool_match_work/tool_0.isolated.atb";
    int iterations = 100;
    int warmup = 5;
};

CliArgs parse_args(int argc, char** argv) {
    CliArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            args.model_path = argv[++i];
        } else if (arg == "--atb" && i + 1 < argc) {
            args.atb_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            args.output_path = argv[++i];
        } else if (arg == "--iterations" && i + 1 < argc) {
            args.iterations = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            args.warmup = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [--model <gguf>] [--output <json>] [--iterations N] [--warmup N]\n";
            std::exit(0);
        } else if (args.model_path.empty()) {
            args.model_path = arg; // Backward-compatible positional model path.
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    if (args.iterations <= 0) {
        throw std::runtime_error("--iterations must be positive.");
    }
    if (args.warmup < 0) {
        throw std::runtime_error("--warmup must be non-negative.");
    }
    return args;
}

int main(int argc, char** argv) {
    std::printf("======================================================================\n");
    std::printf("          PROJECT NEXUS PHASE 21 TTFT MICROBENCHMARK RUNNER           \n");
    std::printf("======================================================================\n");

    llama_backend_init();

    CliArgs args;
    try {
        args = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Argument error: " << e.what() << "\n";
        return 1;
    }

    std::string model_path = args.model_path;
    if (model_path.empty()) {
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

    std::string atb_path = args.atb_path;
    if (!std::filesystem::exists(atb_path)) {
        std::printf("Real ATB not found at %s; generating synthetic fallback.\n", atb_path.c_str());
        atb_path = "bench_temp_tool.atb";
        create_benchmark_atb_file(atb_path, schema_len, n_layer, n_head_kv, d_head);
    }
    const char* atb_payload_label = (args.atb_path == atb_path && atb_path != "bench_temp_tool.atb")
        ? "real_compiled_kv"
        : (std::filesystem::exists(args.atb_path) ? "real_compiled_kv" : "synthetic_fp16_0x3c00_dummy_kv");
    if (std::filesystem::exists(args.atb_path)) {
        atb_path = args.atb_path;
        atb_payload_label = "real_compiled_kv";
    }

    // Cache with 4GB pinned memory limit
    NexusBlockCache cache(4ULL * 1024 * 1024 * 1024);

    // Initialize SLB
    uint32_t n_embd = llama_n_embd(model);
    uint32_t vocab_size = llama_n_vocab(model);
    NexusSemanticSLB slb(n_embd);
    // Register some dummy tools
    for (uint32_t i = 0; i < 5; ++i) {
        std::vector<float> dummy_vec(n_embd, 0.05f * (i + 1));
        std::vector<int32_t> dummy_scent = {100 + static_cast<int32_t>(i), 101 + static_cast<int32_t>(i), 102 + static_cast<int32_t>(i), 103 + static_cast<int32_t>(i), 104 + static_cast<int32_t>(i)};
        slb.register_tool(1000 + i, dummy_vec, dummy_scent);
    }
    
    // Initialize FSM
    NexusRadixFSM fsm;
    std::vector<llama_token> fsm_route = { 101, 102, 103, 104, 105 };
    fsm.add_route(1000, fsm_route);

    std::vector<double> prefill_latencies;
    std::vector<double> slb_scan_latencies;
    std::vector<double> fsm_routing_latencies;
    std::vector<double> vram_splice_latencies;
    std::vector<double> query_delta_latencies;
    std::vector<double> true_nexus_ttfts;

    // Warmup runs
    std::printf("Executing warmup passes...\n");
    for (int i = 0; i < args.warmup; ++i) {
        try {
            run_real_prefill_baseline(ctx, 12500);
            run_nexus_splicer_benchmark(cache, atb_path, ctx, 256);
            run_real_prefill_baseline(ctx, 64);
            std::vector<float> query_emb(n_embd, 0.1f);
            slb.search(query_emb, 3);
            fsm.reset(0);
            fsm.begin_routing(0);
            std::vector<float> logits(vocab_size, 0.0f);
            fsm.apply_logit_mask(logits, 0);
            fsm.advance(101, 0);
        } catch (const std::exception& e) {
            std::cerr << "Warmup error: " << e.what() << "\n";
        }
    }

    // Benchmark loop
    std::printf("Running %d iterations...\n", args.iterations);
    for (int i = 0; i < args.iterations; ++i) {
        try {
            // 1. Measure standard baseline (12.5k prefill)
            prefill_latencies.push_back(run_real_prefill_baseline(ctx, 12500));

            // 2. Measure SLB Scan
            std::vector<float> query_emb(n_embd, 0.1f);
            auto t0_slb = std::chrono::high_resolution_clock::now();
            auto slb_res = slb.search(query_emb, 3);
            auto t1_slb = std::chrono::high_resolution_clock::now();
            double slb_us = std::chrono::duration<double, std::micro>(t1_slb - t0_slb).count();
            slb_scan_latencies.push_back(slb_us);

            // 3. Measure FSM Routing
            fsm.reset(0);
            fsm.begin_routing(0);
            std::vector<float> logits(vocab_size, 0.0f);
            auto t0_fsm = std::chrono::high_resolution_clock::now();
            fsm.apply_logit_mask(logits, 0);
            fsm.advance(101, 0);
            auto t1_fsm = std::chrono::high_resolution_clock::now();
            double fsm_us = std::chrono::duration<double, std::micro>(t1_fsm - t0_fsm).count();
            fsm_routing_latencies.push_back(fsm_us);

            // 4. Measure VRAM Splice
            double splice_ms = run_nexus_splicer_benchmark(cache, atb_path, ctx, 256);
            vram_splice_latencies.push_back(splice_ms);

            // 5. Measure Query Delta Prefill
            double delta_ms = run_real_prefill_baseline(ctx, 64);
            query_delta_latencies.push_back(delta_ms);

            // True Nexus TTFT = SLB + FSM + VRAM Splice + Query Delta
            double true_ttft = (slb_us / 1000.0) + (fsm_us / 1000.0) + splice_ms + delta_ms;
            true_nexus_ttfts.push_back(true_ttft);
        } catch (const std::exception& e) {
            std::cerr << "Iteration " << i << " error: " << e.what() << "\n";
        }
    }

    if (!prefill_latencies.empty() && !true_nexus_ttfts.empty()) {
        std::printf("\n======================================================================\n");
        std::printf("                   BENCHMARK COMPONENT PERFORMANCE                    \n");
        std::printf("======================================================================\n");
        print_stats("[FlashAttention-2 GPU Prefill] Baseline (12.5k)", prefill_latencies);
        std::printf("----------------------------------------------------------------------\n");
        print_stats_us("[L1 SLB Scan Latency] (SIMD dot-product)", slb_scan_latencies);
        print_stats_us("[FSM Constrained Routing Latency] (Radix Trie)", fsm_routing_latencies);
        print_stats("[VRAM Cache Splice Latency] (ATB Hot-Swap)", vram_splice_latencies);
        print_stats("[Query Delta-Prefill Latency] (64-Token Tail)", query_delta_latencies);
        std::printf("----------------------------------------------------------------------\n");
        print_stats("[True Nexus TTFT] (Sum of the 4 above)", true_nexus_ttfts);
        std::printf("----------------------------------------------------------------------\n");

        std::sort(prefill_latencies.begin(), prefill_latencies.end());
        std::sort(true_nexus_ttfts.begin(), true_nexus_ttfts.end());
        double speedup = prefill_latencies[prefill_latencies.size() / 2] / true_nexus_ttfts[true_nexus_ttfts.size() / 2];
        std::printf("True Empirical Speedup factor (P50): %.2f x\n", speedup);
        std::printf("======================================================================\n");
    } else {
        std::cerr << "Benchmark failed to collect latencies.\n";
    }

    try {
        nexus::bench::write_artifact(
            args.output_path,
            "bench_phase21_ttft",
            model_path,
            {
                {"model_path", model_path},
                {"topology", topology_label},
                {"iterations", std::to_string(args.iterations)},
                {"warmup", std::to_string(args.warmup)},
                {"schema_len_tokens", std::to_string(schema_len)},
                {"baseline_prefill_tokens", "12500"},
                {"query_delta_tokens", "64"},
                {"atb_path", atb_path},
                {"atb_payload", atb_payload_label}
            },
            {
                {"baseline_prefill_ms", "ms", prefill_latencies},
                {"slb_scan_us", "us", slb_scan_latencies},
                {"fsm_routing_us", "us", fsm_routing_latencies},
                {"vram_splice_ms", "ms", vram_splice_latencies},
                {"query_delta_ms", "ms", query_delta_latencies},
                {"true_nexus_ttft_ms", "ms", true_nexus_ttfts}
            });
        std::printf("Wrote JSON artifact: %s\n", args.output_path.c_str());
    } catch (const std::exception& e) {
        std::cerr << "Failed to write benchmark artifact: " << e.what() << "\n";
    }

    // Clean up
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    if (atb_path == "bench_temp_tool.atb") {
        std::filesystem::remove(atb_path);
    }
    return 0;
}
