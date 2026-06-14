#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>
#include <cstring>
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <random>
#include "llama.h"
#include "nexus_block_cache.hpp"
#include "aeon_tool_block.hpp"
#include "nexus_benchmark_results.hpp"

static void create_mock_atb(const std::string& path, uint32_t seq_len, uint32_t n_layer, uint32_t n_head_kv, uint32_t d_head) {
    std::ofstream file(path, std::ios::binary);
    if (!file) return;

    AeonToolBlockHeader header{};
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

struct CliArgs {
    std::string output_path = "results/bench_phase22_concurrent_hazard.json";
    uint32_t seed = 1337;
    int warmup_ms = 250;
    int duration_ms = 5000;
};

static CliArgs parse_args(int argc, char** argv) {
    CliArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--output" && i + 1 < argc) {
            args.output_path = argv[++i];
        } else if (arg == "--seed" && i + 1 < argc) {
            args.seed = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--warmup-ms" && i + 1 < argc) {
            args.warmup_ms = std::stoi(argv[++i]);
        } else if (arg == "--duration-ms" && i + 1 < argc) {
            args.duration_ms = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [--output <json>] [--seed N] [--warmup-ms N] [--duration-ms N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    if (args.warmup_ms < 0 || args.duration_ms <= 0) {
        throw std::runtime_error("warmup must be non-negative and duration must be positive.");
    }
    return args;
}

int main(int argc, char** argv) {
    CliArgs args;
    try {
        args = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Argument error: " << e.what() << "\n";
        return 1;
    }

    llama_backend_init();

    // Set cache capacity to 256MB
    size_t cache_capacity = 256ULL * 1024 * 1024;
    NexusBlockCache cache(cache_capacity);

    // Create 15 mock files. Each file size is ~18MB (32 layers, 8 heads, 128 dim, 256 seq_len)
    // 15 * 18MB = 270MB, which exceeds the 256MB capacity and forces evictions
    std::vector<std::string> atb_files;
    for (int i = 0; i < 15; ++i) {
        std::string filename = "mock_tool_" + std::to_string(i) + ".atb";
        create_mock_atb(filename, 256, 32, 8, 128);
        atb_files.push_back(filename);
    }

    std::atomic<bool> start_signal{false};
    std::atomic<bool> stop_signal{false};
    std::atomic<uint64_t> total_ops{0};

    auto run_thread = [&](int thread_id, std::vector<double>& latencies) {
        std::mt19937 gen(args.seed + static_cast<uint32_t>(thread_id));
        std::uniform_int_distribution<> dis(0, 14);

        while (!start_signal.load(std::memory_order_relaxed)) {
            std::this_thread::yield();
        }

        while (!stop_signal.load(std::memory_order_relaxed)) {
            int file_idx = dis(gen);
            std::string file_path = atb_files[file_idx];

            auto t0 = std::chrono::high_resolution_clock::now();
            
            try {
                auto res = cache.get_or_load(file_path);
                if (res.has_value()) {
                    auto block = res.value();
                    void* mapped_data = block->get_mapped_data();
                    
                    // Acquire HazardGuard
                    HazardGuard guard = cache.acquire_hazard(mapped_data);
                    if (guard.entry) {
                        // Simulate GPU read delay of 500us
                        std::this_thread::sleep_for(std::chrono::microseconds(500));
                    }
                }
            } catch (...) {
                // Ignore load/capacity exceptions to maintain benchmark uptime
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            double duration_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            latencies.push_back(duration_ms);
            total_ops.fetch_add(1, std::memory_order_relaxed);
        }
    };

    std::vector<int> thread_counts = {1, 4, 8, 16, 32, 64};
    std::vector<double> thread_count_samples;
    std::vector<double> ops_samples;
    std::vector<double> p50_samples;
    std::vector<double> p99_samples;

    std::printf("| Threads | Cache MMU Operations Per Second (OPS) | P50 Latency (ms) | P99 Latency (ms) |\n");
    std::printf("|---------|---------------------------------------|------------------|------------------|\n");

    for (int num_threads : thread_counts) {
        start_signal.store(false);
        stop_signal.store(false);
        total_ops.store(0);

        std::vector<std::thread> threads;
        std::vector<std::vector<double>> all_latencies(num_threads);

        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back(run_thread, i, std::ref(all_latencies[i]));
        }

        // Warmup
        std::this_thread::sleep_for(std::chrono::milliseconds(args.warmup_ms));
        start_signal.store(true);

        // Run for the configured measured interval.
        auto start_time = std::chrono::high_resolution_clock::now();
        std::this_thread::sleep_for(std::chrono::milliseconds(args.duration_ms));
        stop_signal.store(true);

        for (auto& t : threads) {
            t.join();
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();

        // Aggregate latencies
        std::vector<double> combined_latencies;
        for (const auto& lat : all_latencies) {
            combined_latencies.insert(combined_latencies.end(), lat.begin(), lat.end());
        }

        double ops = combined_latencies.size() / elapsed_sec;
        double p50 = 0.0;
        double p99 = 0.0;

        if (!combined_latencies.empty()) {
            std::sort(combined_latencies.begin(), combined_latencies.end());
            p50 = combined_latencies[combined_latencies.size() / 2];
            p99 = combined_latencies[static_cast<size_t>(combined_latencies.size() * 0.99)];
        }

        std::printf("| %7d | %37.1f | %16.2f | %16.2f |\n", num_threads, ops, p50, p99);
        thread_count_samples.push_back(static_cast<double>(num_threads));
        ops_samples.push_back(ops);
        p50_samples.push_back(p50);
        p99_samples.push_back(p99);
    }

    try {
        nexus::bench::write_artifact(
            args.output_path,
            "bench_phase22_concurrent_hazard",
            "synthetic_atb",
            {
                {"seed", std::to_string(args.seed)},
                {"warmup_ms", std::to_string(args.warmup_ms)},
                {"duration_ms", std::to_string(args.duration_ms)},
                {"tool_count", std::to_string(atb_files.size())},
                {"access_distribution", "uniform"},
                {"simulated_gpu_read_us", "500"},
                {"cache_capacity_bytes", std::to_string(cache_capacity)}
            },
            {
                {"thread_count", "threads", thread_count_samples},
                {"ops", "ops_per_second", ops_samples},
                {"p50_latency_ms", "ms", p50_samples},
                {"p99_latency_ms", "ms", p99_samples}
            });
        std::printf("Wrote JSON artifact: %s\n", args.output_path.c_str());
    } catch (const std::exception& e) {
        std::cerr << "Failed to write benchmark artifact: " << e.what() << "\n";
    }

    // Clean up
    for (const auto& file : atb_files) {
        std::filesystem::remove(file);
    }
    llama_backend_free();

    return 0;
}
