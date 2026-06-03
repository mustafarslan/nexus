#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <filesystem>
#include <cassert>

#include "aeon_tool_block.hpp"

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <atb_file_path>\n";
        return 1;
    }

    std::string path = argv[1];
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open ATB file: " << path << "\n";
        return 1;
    }

    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (file_size < static_cast<std::streamsize>(sizeof(AeonToolBlockHeader))) {
        std::cerr << "Error: File size (" << file_size << ") is smaller than the header size ("
                  << sizeof(AeonToolBlockHeader) << " bytes).\n";
        return 1;
    }

    auto header_ptr = std::make_unique<AeonToolBlockHeader>();
    file.read(reinterpret_cast<char*>(header_ptr.get()), sizeof(AeonToolBlockHeader));
    if (!file) {
        std::cerr << "Error: Failed to read ATB header.\n";
        return 1;
    }
    const AeonToolBlockHeader& header = *header_ptr;

    // 1. Check Magic
    if (std::memcmp(header.magic, "ATB1", 4) != 0) {
        std::cerr << "Error: Invalid magic bytes. Expected 'ATB1', got: '"
                  << header.magic[0] << header.magic[1] << header.magic[2] << header.magic[3] << "'\n";
        return 1;
    }

    // 2. Check Version
    if (header.version != 1) {
        std::cerr << "Error: Unsupported version: " << header.version << "\n";
        return 1;
    }

    // 3. Compute expected tensor sizes
    uint64_t expected_layer_bytes = static_cast<uint64_t>(header.n_head_kv) *
                                    header.seq_len *
                                    header.d_head * 2; // F16 is 2 bytes
    uint64_t expected_total_bytes = header.n_layer * expected_layer_bytes;

    std::cout << "--- AeonToolBlock Verification Summary ---\n"
              << "File:             " << path << "\n"
              << "File Size:        " << file_size << " bytes\n"
              << "Header Size:      " << sizeof(AeonToolBlockHeader) << " bytes (aligned to 2097152)\n"
              << "Model Hash:       0x" << std::hex << header.model_hash << std::dec << "\n"
              << "Layers:           " << header.n_layer << "\n"
              << "KV Heads:         " << header.n_head_kv << "\n"
              << "d_head:           " << header.d_head << "\n"
              << "Sequence Length:  " << header.seq_len << "\n"
              << "Base Position:    " << header.base_pos << "\n"
              << "RoPE Freq Base:   " << header.rope_freq_base << "\n"
              << "RoPE Freq Scale:  " << header.rope_freq_scale << "\n"
              << "RoPE Scaling Type:" << header.rope_scaling_type << "\n"
              << "ggml_type_k:      " << header.ggml_type_k << " (F16=1)\n"
              << "ggml_type_v:      " << header.ggml_type_v << " (F16=1)\n"
              << "K offset:         " << header.k_tensor_offset << " bytes\n"
              << "V offset:         " << header.v_tensor_offset << " bytes\n"
              << "K size:           " << header.k_total_bytes << " bytes\n"
              << "V size:           " << header.v_total_bytes << " bytes\n";

    // 4. Assert size alignments and values
    if (header.ggml_type_k != 1 || header.ggml_type_v != 1) {
        std::cerr << "Error: Only GGML_TYPE_F16 (1) KV cache type is supported.\n";
        return 1;
    }

    if (header.k_total_bytes != expected_total_bytes) {
        std::cerr << "Error: k_total_bytes mismatch. Expected " << expected_total_bytes
                  << ", got " << header.k_total_bytes << "\n";
        return 1;
    }

    if (header.v_total_bytes != expected_total_bytes) {
        std::cerr << "Error: v_total_bytes mismatch. Expected " << expected_total_bytes
                  << ", got " << header.v_total_bytes << "\n";
        return 1;
    }

    uint64_t k_aligned = ((header.k_total_bytes + 2097151) / 2097152) * 2097152;
    if (header.v_tensor_offset != header.k_tensor_offset + k_aligned) {
        std::cerr << "Error: v_tensor_offset alignment mismatch. Expected "
                  << header.k_tensor_offset + k_aligned << ", got " << header.v_tensor_offset << "\n";
        return 1;
    }

    uint64_t v_aligned = ((header.v_total_bytes + 2097151) / 2097152) * 2097152;
    std::streamsize expected_file_size = header.v_tensor_offset + v_aligned;
    if (file_size != expected_file_size) {
        std::cerr << "Error: File size mismatch. Expected exactly " << expected_file_size
                  << " bytes, got " << file_size << " bytes.\n";
        return 1;
    }

    std::cout << "Verification Status: SUCCESS\n";
    return 0;
}
