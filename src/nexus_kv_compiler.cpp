#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstring>
#include <filesystem>
#include <memory>
#include <array>
#include <cassert>

#include <llama.h>
#include <ggml.h>
#include <ggml-backend.h>

#include "aeon_tool_block.hpp"

// Redefine internal structures and bridge signatures for opaque context access
extern "C" {
    struct ggml_tensor * llama_kv_cache_get_k(struct llama_context * ctx, int il);
    struct ggml_tensor * llama_kv_cache_get_v(struct llama_context * ctx, int il);
    bool llama_kv_cache_get_v_trans(struct llama_context * ctx);
    uint32_t llama_kv_cache_get_size(struct llama_context * ctx);
    uint32_t llama_model_n_head_kv(const struct llama_model * model, int il);
    uint32_t llama_model_d_head(const struct llama_model * model);
    float llama_model_rope_freq_base(const struct llama_model * model);
    float llama_model_rope_freq_scale(const struct llama_model * model);
    uint32_t llama_model_rope_scaling_type(const struct llama_model * model);
    const char * llama_model_name(const struct llama_model * model);
}

struct CliArgs {
    std::string model_path;
    std::string schema_path;
    std::string output_path;
    std::string static_anchor_path;
    std::string batch_list_path;
    bool verbose = false;
    int32_t threads = -1;
    uint32_t base_pos = 0;
    uint32_t pos_bucket = 0;
    bool sink_prefix = false;
};

void print_usage(const char * argv0) {
    std::cerr << "Usage: " << argv0 << " --model <model_path> --schema <schema_path> --output <output_path> [options]\n"
              << "Options:\n"
              << "  --batch-list <path>  One 'schema_path,output_path' per line (single model load)\n"
              << "  --static-anchor <path>  Path to static anchor text file (optional)\n"
              << "  --base-pos <int>   Prefill filler tokens before schema at this position (default: 0)\n"
              << "  --pos-bucket <int> Snap compile base to positional bucket (256/1024/4096)\n"
              << "  --sink-prefix      Prepend attention-sink system prefix before schema\n"
              << "  --threads <int>    Number of threads for decode (default: auto)\n"
              << "  --verbose          Enable verbose logging\n"
              << "  --help             Show this help message\n";
}

bool parse_args(int argc, char ** argv, CliArgs & args) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model") {
            if (i + 1 < argc) args.model_path = argv[++i];
            else return false;
        } else if (arg == "--schema") {
            if (i + 1 < argc) args.schema_path = argv[++i];
            else return false;
        } else if (arg == "--output") {
            if (i + 1 < argc) args.output_path = argv[++i];
            else return false;
        } else if (arg == "--static-anchor") {
            if (i + 1 < argc) args.static_anchor_path = argv[++i];
            else return false;
        } else if (arg == "--batch-list") {
            if (i + 1 < argc) args.batch_list_path = argv[++i];
            else return false;
        } else if (arg == "--base-pos") {
            if (i + 1 < argc) args.base_pos = static_cast<uint32_t>(std::stoul(argv[++i]));
            else return false;
        } else if (arg == "--pos-bucket") {
            if (i + 1 < argc) args.pos_bucket = static_cast<uint32_t>(std::stoul(argv[++i]));
            else return false;
        } else if (arg == "--sink-prefix") {
            args.sink_prefix = true;
        } else if (arg == "--threads") {
            if (i + 1 < argc) args.threads = std::stoi(argv[++i]);
            else return false;
        } else if (arg == "--verbose") {
            args.verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return false;
        }
    }
    return !args.model_path.empty()
        && ((!args.schema_path.empty() && !args.output_path.empty()) || !args.batch_list_path.empty());
}

std::string read_file(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open schema file: " << path << "\n";
        std::exit(1);
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

uint64_t fnv1a_64(const void * data, size_t num_bytes, uint64_t hash = 0xcbf29ce484222325ULL) {
    const uint8_t * bytes = (const uint8_t *) data;
    for (size_t i = 0; i < num_bytes; ++i) {
        hash ^= bytes[i];
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

uint64_t compute_model_hash(const struct llama_model * model) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    
    // Hash model name
    const char * name = llama_model_name(model);
    hash = fnv1a_64(name, std::strlen(name), hash);
    
    // Hash n_layer
    uint32_t n_layer = llama_n_layer(model);
    hash = fnv1a_64(&n_layer, sizeof(n_layer), hash);
    
    // Hash d_head
    uint32_t d_head = llama_model_d_head(model);
    hash = fnv1a_64(&d_head, sizeof(d_head), hash);
    
    // Hash rope_freq_base
    float rope_freq_base = llama_model_rope_freq_base(model);
    hash = fnv1a_64(&rope_freq_base, sizeof(rope_freq_base), hash);
    
    // Hash n_head_kv for all layers
    for (uint32_t il = 0; il < n_layer; ++il) {
        uint32_t n_head_kv = llama_model_n_head_kv(model, il);
        hash = fnv1a_64(&n_head_kv, sizeof(n_head_kv), hash);
    }
    
    return hash;
}

int main(int argc, char ** argv) {
    CliArgs args;
    if (!parse_args(argc, argv, args)) {
        print_usage(argv[0]);
        return 1;
    }

    if (args.verbose) {
        std::cout << "Initializing llama backend...\n";
    }
    llama_backend_init();

    // Load the GGUF model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 999; // Offload as many layers to GPU as possible

    if (args.verbose) {
        std::cout << "Loading model from " << args.model_path << "...\n";
    }
    struct llama_model * model = llama_load_model_from_file(args.model_path.c_str(), model_params);
    if (!model) {
        std::cerr << "Error: Failed to load model from " << args.model_path << "\n";
        llama_backend_free();
        return 1;
    }

    std::vector<std::pair<std::string, std::string>> jobs;
    if (!args.batch_list_path.empty()) {
        std::ifstream batch_in(args.batch_list_path);
        if (!batch_in.is_open()) {
            std::cerr << "Error: Could not open batch list: " << args.batch_list_path << "\n";
            llama_free_model(model);
            llama_backend_free();
            return 1;
        }
        std::string line;
        while (std::getline(batch_in, line)) {
            if (line.empty() || line[0] == '#') continue;
            auto comma = line.find(',');
            if (comma == std::string::npos) continue;
            jobs.emplace_back(line.substr(0, comma), line.substr(comma + 1));
        }
        std::cout << "Batch mode: " << jobs.size() << " schemas (single model load)\n";
    } else {
        jobs.emplace_back(args.schema_path, args.output_path);
    }

    int exit_code = 0;

    std::vector<llama_token> system_tokens;
    if (!args.static_anchor_path.empty()) {
        std::string system_text = read_file(args.static_anchor_path);
        system_tokens.resize(system_text.size() + 4);
        int32_t n_sys_tokens = llama_tokenize(model, system_text.c_str(), system_text.size(), system_tokens.data(), system_tokens.size(), false, false);
        if (n_sys_tokens < 0) {
            system_tokens.resize(-n_sys_tokens);
            n_sys_tokens = llama_tokenize(model, system_text.c_str(), system_text.size(), system_tokens.data(), system_tokens.size(), false, false);
        }
        if (n_sys_tokens < 0) {
            std::cerr << "Error: Failed to tokenize static anchor text.\n";
            llama_free_model(model);
            llama_backend_free();
            return 1;
        }
        system_tokens.resize(n_sys_tokens);
        if (args.verbose) {
            std::cout << "Tokenized static anchor: " << n_sys_tokens << " tokens.\n";
        }
    } else if (args.sink_prefix) {
        const std::string sink_text = "You are a helpful assistant.\n\n";
        system_tokens.resize(sink_text.size() + 4);
        int32_t n_sys_tokens = llama_tokenize(model, sink_text.c_str(), sink_text.size(), system_tokens.data(), system_tokens.size(), false, false);
        if (n_sys_tokens < 0) {
            system_tokens.resize(-n_sys_tokens);
            n_sys_tokens = llama_tokenize(model, sink_text.c_str(), sink_text.size(), system_tokens.data(), system_tokens.size(), false, false);
        }
        if (n_sys_tokens < 0) {
            std::cerr << "Error: Failed to tokenize sink prefix text.\n";
            llama_free_model(model);
            llama_backend_free();
            return 1;
        }
        system_tokens.resize(n_sys_tokens);
        if (args.verbose) {
            std::cout << "Tokenized sink prefix: " << n_sys_tokens << " tokens.\n";
        }
    }

    for (size_t job_idx = 0; job_idx < jobs.size(); ++job_idx) {
        const std::string & schema_path = jobs[job_idx].first;
        const std::string & output_path = jobs[job_idx].second;
        if (args.verbose) {
            std::cout << "Job " << (job_idx + 1) << "/" << jobs.size() << ": " << schema_path << "\n";
        }

    // Load and tokenize the MCP tool JSON schema
    std::string schema_text = read_file(schema_path);
    std::vector<llama_token> schema_tokens(schema_text.size() + 4);
    int32_t n_schema_tokens = llama_tokenize(model, schema_text.c_str(), schema_text.size(), schema_tokens.data(), schema_tokens.size(), false, false);
    if (n_schema_tokens < 0) {
        schema_tokens.resize(-n_schema_tokens);
        n_schema_tokens = llama_tokenize(model, schema_text.c_str(), schema_text.size(), schema_tokens.data(), schema_tokens.size(), false, false);
    }
    if (n_schema_tokens < 0) {
        std::cerr << "Error: Failed to tokenize schema text.\n";
        exit_code = 1;
        break;
    }
    schema_tokens.resize(n_schema_tokens);

    if (args.verbose) {
        std::cout << "Tokenized schema: " << n_schema_tokens << " tokens.\n";
    }

    uint32_t system_prefix_length = system_tokens.size();
    uint32_t n_tokens = schema_tokens.size();
    uint32_t compile_base = args.base_pos;
    if (args.pos_bucket > 0) {
        compile_base = (args.base_pos / args.pos_bucket) * args.pos_bucket;
    }
    uint32_t total_tokens = compile_base + system_prefix_length + n_tokens;

    // Configure context parameters
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = total_tokens + 64;
    ctx_params.n_batch = ctx_params.n_ctx;
    ctx_params.n_ubatch = ctx_params.n_ctx;
    ctx_params.type_k = GGML_TYPE_F16;
    ctx_params.type_v = GGML_TYPE_F16;
    ctx_params.offload_kqv = true;
    ctx_params.flash_attn = true; // Enable flash attention
    if (args.threads > 0) {
        ctx_params.n_threads = args.threads;
        ctx_params.n_threads_batch = args.threads;
    }

    struct llama_context * ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        std::cerr << "Error: Failed to create llama context.\n";
        llama_free_model(model);
        llama_backend_free();
        return 1;
    }

    // Run the prefill decoding starting at pos = 0
    llama_batch batch = llama_batch_init(total_tokens, 0, 1);
    batch.n_tokens = total_tokens;
    uint32_t pos = 0;
    if (compile_base > 0) {
        for (uint32_t i = 0; i < compile_base; ++i) {
            batch.token[i] = 1;
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 0;
        }
        pos = compile_base;
    }
    for (uint32_t i = 0; i < system_prefix_length; ++i) {
        batch.token[pos + i] = system_tokens[i];
        batch.pos[pos + i] = pos + i;
        batch.n_seq_id[pos + i] = 1;
        batch.seq_id[pos + i][0] = 0;
        batch.logits[pos + i] = 0;
    }
    pos += system_prefix_length;
    for (uint32_t i = 0; i < n_tokens; ++i) {
        batch.token[pos + i] = schema_tokens[i];
        batch.pos[pos + i] = pos + i;
        batch.n_seq_id[pos + i] = 1;
        batch.seq_id[pos + i][0] = 0;
        batch.logits[pos + i] = (i == n_tokens - 1) ? 1 : 0;
    }

    if (args.verbose) {
        std::cout << "Prefilling with compile_base=" << compile_base
                  << " system_prefix=" << system_prefix_length
                  << " schema_tokens=" << n_tokens << "...\n";
    }
    int decode_res = llama_decode(ctx, batch);
    llama_batch_free(batch);

    if (decode_res != 0) {
        std::cerr << "Error: llama_decode failed with code " << decode_res << "\n";
        llama_free(ctx);
        llama_free_model(model);
        llama_backend_free();
        return 1;
    }

    // Ensure all backend calculations are complete
    llama_synchronize(ctx);

    // Retrieve hyper-parameters and check consistency
    uint32_t n_layer = llama_n_layer(model);
    uint32_t d_head = llama_model_d_head(model);
    uint32_t n_head_kv = llama_model_n_head_kv(model, 0);

    for (uint32_t il = 1; il < n_layer; ++il) {
        if (llama_model_n_head_kv(model, il) != n_head_kv) {
            std::cerr << "Error: Varying n_head_kv per layer is not supported in Phase 1.\n";
            llama_free(ctx);
            llama_free_model(model);
            llama_backend_free();
            return 1;
        }
    }

    uint64_t model_hash = compute_model_hash(model);

    // Set up serialization layout
    uint64_t layer_bytes = n_head_kv * n_tokens * d_head * 2; // sizeof(half) = 2 bytes
    uint64_t k_total_bytes = n_layer * layer_bytes;
    uint64_t v_total_bytes = n_layer * layer_bytes;
    
    // Align K data size to 2MB bytes for memory alignment compatibility (Huge Pages)
    uint64_t k_total_bytes_aligned = ((k_total_bytes + 2097151) / 2097152) * 2097152;

    AeonToolBlockHeader header = {};
    std::memset(&header, 0, sizeof(header));
    std::memcpy(header.magic, "ATB1", 4);
    header.version = 1;
    header.model_hash = model_hash;
    header.n_layer = n_layer;
    header.n_head_kv = n_head_kv;
    header.d_head = d_head;
    header.seq_len = n_tokens;
    uint32_t schema_start = compile_base + system_prefix_length;
    header.base_pos = schema_start;
    header.rope_freq_base = llama_model_rope_freq_base(model);
    header.rope_freq_scale = llama_model_rope_freq_scale(model);
    header.rope_scaling_type = llama_model_rope_scaling_type(model);
    header.ggml_type_k = (uint32_t) GGML_TYPE_F16;
    header.ggml_type_v = (uint32_t) GGML_TYPE_F16;
    header.k_tensor_offset = NEXUS_PAGE_ALIGNMENT;
    header.v_tensor_offset = header.k_tensor_offset + k_total_bytes_aligned;
    header.k_total_bytes = k_total_bytes;
    header.v_total_bytes = v_total_bytes;

    std::vector<uint16_t> k_packed(k_total_bytes / 2);
    std::vector<uint16_t> v_packed(v_total_bytes / 2);

    uint32_t kv_size = llama_kv_cache_get_size(ctx);
    bool v_trans = llama_kv_cache_get_v_trans(ctx);

    if (args.verbose) {
        std::cout << "Model stats: n_layer=" << n_layer << ", n_head_kv=" << n_head_kv << ", d_head=" << d_head << "\n";
        std::cout << "KV cache stats: kv_size=" << kv_size << ", v_trans=" << (v_trans ? "true" : "false") << "\n";
        std::cout << "Extracting KV cache tensors...\n";
    }

    size_t layer_elements = n_head_kv * n_tokens * d_head;

    for (uint32_t il = 0; il < n_layer; ++il) {
        struct ggml_tensor * k_tensor = llama_kv_cache_get_k(ctx, il);
        struct ggml_tensor * v_tensor = llama_kv_cache_get_v(ctx, il);

        if (!k_tensor || !v_tensor) {
            std::cerr << "Error: Failed to retrieve KV cache tensors for layer " << il << "\n";
            llama_free(ctx);
            llama_free_model(model);
            llama_backend_free();
            return 1;
        }

        // Extract and pack K [n_tokens, n_head_kv, d_head]
        {
            size_t k_slice_elements = n_head_kv * d_head * n_tokens;
            size_t src_element_offset = schema_start * n_head_kv * d_head;
            size_t layer_offset = il * layer_elements;

            ggml_backend_tensor_get(k_tensor, k_packed.data() + layer_offset, src_element_offset * 2, k_slice_elements * 2);
        }

        // Extract and pack V
        {
            size_t layer_offset = il * layer_elements;
            if (!v_trans) {
                // Un-transposed layout: [n_tokens, n_head_kv, d_head]
                size_t v_slice_elements = n_head_kv * d_head * n_tokens;
                size_t src_element_offset = schema_start * n_head_kv * d_head;

                ggml_backend_tensor_get(v_tensor, v_packed.data() + layer_offset, src_element_offset * 2, v_slice_elements * 2);
            } else {
                // Transposed layout: [n_head_kv, d_head, n_tokens] (pre-transposed offline)
                size_t total_v_elements = n_head_kv * d_head * kv_size;
                std::vector<uint16_t> host_v_full(total_v_elements);

                ggml_backend_tensor_get(v_tensor, host_v_full.data(), 0, total_v_elements * 2);

                for (uint32_t ih = 0; ih < n_head_kv; ++ih) {
                    for (uint32_t id = 0; id < d_head; ++id) {
                        uint32_t j = ih * d_head + id;
                        for (uint32_t is = 0; is < n_tokens; ++is) {
                            uint32_t pos = system_prefix_length + is;
                            size_t src_idx = pos + id * kv_size + ih * kv_size * d_head;
                            size_t dst_idx = layer_offset + j * n_tokens + is;
                            v_packed[dst_idx] = host_v_full[src_idx];
                        }
                    }
                }
            }
        }
    }

    if (args.verbose) {
        std::cout << "Writing AeonToolBlock to " << output_path << "...\n";
    }

    std::ofstream out(output_path, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Error: Failed to open output file: " << output_path << "\n";
        llama_free(ctx);
        exit_code = 1;
        break;
    }

    // Write aligned header
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // Pad header section to NEXUS_PAGE_ALIGNMENT boundary
    if (NEXUS_PAGE_ALIGNMENT > sizeof(AeonToolBlockHeader)) {
        std::vector<char> pad(NEXUS_PAGE_ALIGNMENT - sizeof(AeonToolBlockHeader), 0);
        out.write(pad.data(), pad.size());
    }

    // Write K cache data
    out.write(reinterpret_cast<const char*>(k_packed.data()), k_total_bytes);

    // Pad K data section to 16384 bytes boundary
    if (k_total_bytes_aligned > k_total_bytes) {
        std::vector<char> pad(k_total_bytes_aligned - k_total_bytes, 0);
        out.write(pad.data(), pad.size());
    }

    // Write V cache data
    out.write(reinterpret_cast<const char*>(v_packed.data()), v_total_bytes);

    // Pad V data section to 2MB boundary at the EOF so the total file size is a multiple of 2MB
    uint64_t v_total_bytes_aligned = ((v_total_bytes + 2097151) / 2097152) * 2097152;
    if (v_total_bytes_aligned > v_total_bytes) {
        std::vector<char> pad(v_total_bytes_aligned - v_total_bytes, 0);
        out.write(pad.data(), pad.size());
    }
    out.close();

    std::cout << "AeonToolBlock written successfully to " << output_path << "\n";
    llama_free(ctx);
    } // end batch jobs loop

    llama_free_model(model);
    llama_backend_free();
    return exit_code;
}
