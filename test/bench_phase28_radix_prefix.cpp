#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include "llama.h"
#include "nexus_benchmark_results.hpp"
#include "nexus_seq_warm_cache.hpp"

extern "C" {
void llama_backend_sync(struct llama_context* ctx);
}

static std::string resolve_model_path() {
    std::array<char, 512> buffer{};
    std::unique_ptr<FILE, decltype(&pclose)> pipe(
        popen("python3 scripts/resolve_ollama_gguf.py 2>/dev/null", "r"), pclose);
    if (!pipe) {
        return "";
    }
    std::string result;
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    if (!result.empty() && result.back() == '\n') {
        result.pop_back();
    }
    return result;
}

static std::vector<int32_t> make_prefix(std::mt19937& rng, size_t len) {
    std::vector<int32_t> out(len);
    for (size_t i = 0; i < len; ++i) {
        out[i] = static_cast<int32_t>(1 + (rng() % 30000));
    }
    return out;
}

static void prefill_tokens(llama_context* ctx, const std::vector<int32_t>& tokens, llama_seq_id seq) {
    if (tokens.empty()) {
        return;
    }
    llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
    batch.n_tokens = static_cast<int32_t>(tokens.size());
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = tokens[static_cast<size_t>(i)];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = seq;
        batch.logits[i] = 0;
    }
    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        throw std::runtime_error("prefill_tokens: llama_decode failed");
    }
    llama_batch_free(batch);
    llama_backend_sync(ctx);
}

int main(int argc, char** argv) {
    std::string output = "results/bench_phase28_radix_prefix.json";
    std::string model_path;
    uint32_t seed = 42;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--output" && i + 1 < argc) {
            output = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = static_cast<uint32_t>(std::stoul(argv[++i]));
        }
    }
    if (model_path.empty()) {
        model_path = resolve_model_path();
    }
    if (model_path.empty()) {
        std::cerr << "bench_phase28_radix_prefix: no model path\n";
        return 1;
    }

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    llama_model* model = llama_load_model_from_file(model_path.c_str(), mparams);
    if (!model) {
        std::cerr << "Failed to load model: " << model_path << "\n";
        return 1;
    }
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 8192;
    cparams.n_seq_max = 128;
    cparams.flash_attn = true;
    llama_context* ctx = llama_new_context_with_model(model, cparams);
    if (!ctx) {
        llama_free_model(model);
        return 1;
    }

    NexusRadixPrefixCache cache;
    std::mt19937 rng(seed);
    std::array<std::vector<int32_t>, 4> canonical{};
    for (size_t i = 0; i < 4; ++i) {
        canonical[i] = make_prefix(rng, 256);
        const llama_seq_id pool_seq = static_cast<llama_seq_id>(NexusRadixPrefixCache::POOL_BASE + i);
        llama_kv_cache_clear(ctx);
        prefill_tokens(ctx, canonical[i], pool_seq);
        cache.update_from_seq(ctx, canonical[i], 256, pool_seq, static_cast<uint32_t>(i + 1));
    }

    uint64_t hits = 0;
    uint64_t tries = 0;
    std::vector<double> latencies;
    latencies.reserve(100);
    for (int req = 0; req < 100; ++req) {
        const bool use_canonical = (rng() % 100) < 70;
        std::vector<int32_t> prefix;
        if (use_canonical) {
            prefix = canonical[static_cast<size_t>(rng() % 4)];
        } else {
            prefix = make_prefix(rng, 64 + (rng() % 64));
        }
        const llama_seq_id dst = NexusRadixPrefixCache::allocate_request_seq(static_cast<uint32_t>(req));
        ++tries;
        auto t0 = std::chrono::steady_clock::now();
        const uint32_t matched = cache.try_copy_prefix(ctx, prefix, dst);
        auto t1 = std::chrono::steady_clock::now();
        latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        if (matched > 0) {
            ++hits;
        } else {
            llama_kv_cache_clear(ctx);
            prefill_tokens(ctx, prefix, dst);
            cache.update_from_seq(ctx, prefix, static_cast<uint32_t>(prefix.size()), dst, 0);
        }
    }

    std::sort(latencies.begin(), latencies.end());
    const double hit_rate = tries ? static_cast<double>(hits) / static_cast<double>(tries) : 0.0;
    const double p50 = latencies.empty() ? 0.0 : latencies[latencies.size() / 2];
    const double p99 = latencies.empty() ? 0.0 : latencies[std::min(latencies.size() - 1, latencies.size() * 99 / 100)];

    nexus::bench::write_artifact(
        output,
        "phase28_radix_prefix",
        model_path,
        {{"seed", std::to_string(seed)}, {"requests", "100"}},
        {
            {"hit_rate", "ratio", {hit_rate}},
            {"copy_p50_us", "us", {p50}},
            {"copy_p99_us", "us", {p99}},
        });

    std::cout << "phase28 hit_rate=" << hit_rate << " copy_p50_us=" << p50 << "\n";

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    if (hit_rate <= 0.8) {
        return 2;
    }
    return 0;
}
