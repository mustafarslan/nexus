#pragma once
#include <cstdint>
#include <bit>
#include <cstddef>

static_assert(std::endian::native == std::endian::little, "Project Nexus requires little-endian architectures.");

// Global physical file alignment constant (2MB) for Huge Pages / Direct I/O
inline constexpr size_t NEXUS_PAGE_ALIGNMENT = 2097152;

// Strictly 128-byte packed metadata struct
#pragma pack(push, 1)
struct alignas(128) AeonToolBlockHeader {
    char     magic[4];          // "ATB1"
    uint32_t version;           // 1
    
    uint64_t model_hash;        // FNV-1a hash of model topology
    uint32_t n_layer;           // Total transformer layers
    uint32_t n_head_kv;         // GQA-aware KV head count
    uint32_t d_head;            // Dimension per head
    
    uint32_t seq_len;           // Active token count in this block
    uint32_t base_pos;          // The RoPE starting position (e.g., 0 or 256)
    float    rope_freq_base;    // RoPE frequency base
    float    rope_freq_scale;   // RoPE frequency scale (default 1.0)
    uint32_t rope_scaling_type; // RoPE scaling type (default 0)
    
    uint32_t ggml_type_k;       // Must be GGML_TYPE_F16 (value: 1)
    uint32_t ggml_type_v;       // Must be GGML_TYPE_F16 (value: 1)
    
    uint64_t k_tensor_offset;   // Byte offset to contiguous K data
    uint64_t v_tensor_offset;   // Byte offset to contiguous V data
    uint64_t k_total_bytes;     // seq_len * n_layer * n_head_kv * d_head * sizeof(F16)
    uint64_t v_total_bytes;     // seq_len * n_layer * n_head_kv * d_head * sizeof(F16)
    
    uint8_t  padding[128 - 88]; // Pad strictly to 128 bytes
};
#pragma pack(pop)

static_assert(sizeof(AeonToolBlockHeader) == 128, "Header must be exactly 128 bytes");
static_assert(offsetof(AeonToolBlockHeader, magic) == 0, "magic offset must be 0");
static_assert(offsetof(AeonToolBlockHeader, version) == 4, "version offset must be 4");
static_assert(offsetof(AeonToolBlockHeader, model_hash) == 8, "model_hash offset must be 8");
static_assert(offsetof(AeonToolBlockHeader, n_layer) == 16, "n_layer offset must be 16");
static_assert(offsetof(AeonToolBlockHeader, n_head_kv) == 20, "n_head_kv offset must be 20");
static_assert(offsetof(AeonToolBlockHeader, d_head) == 24, "d_head offset must be 24");
static_assert(offsetof(AeonToolBlockHeader, seq_len) == 28, "seq_len offset must be 28");
static_assert(offsetof(AeonToolBlockHeader, base_pos) == 32, "base_pos offset must be 32");
static_assert(offsetof(AeonToolBlockHeader, rope_freq_base) == 36, "rope_freq_base offset must be 36");
static_assert(offsetof(AeonToolBlockHeader, rope_freq_scale) == 40, "rope_freq_scale offset must be 40");
static_assert(offsetof(AeonToolBlockHeader, rope_scaling_type) == 44, "rope_scaling_type offset must be 44");
static_assert(offsetof(AeonToolBlockHeader, ggml_type_k) == 48, "ggml_type_k offset must be 48");
static_assert(offsetof(AeonToolBlockHeader, ggml_type_v) == 52, "ggml_type_v offset must be 52");
static_assert(offsetof(AeonToolBlockHeader, k_tensor_offset) == 56, "k_tensor_offset offset must be 56");
static_assert(offsetof(AeonToolBlockHeader, v_tensor_offset) == 64, "v_tensor_offset offset must be 64");
static_assert(offsetof(AeonToolBlockHeader, k_total_bytes) == 72, "k_total_bytes offset must be 72");
static_assert(offsetof(AeonToolBlockHeader, v_total_bytes) == 80, "v_total_bytes offset must be 80");
static_assert(offsetof(AeonToolBlockHeader, padding) == 88, "padding offset must be 88");


