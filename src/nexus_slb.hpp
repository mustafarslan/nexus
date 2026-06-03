#pragma once
#include <cstdint>
#include <vector>
#include <span>

constexpr size_t SCENT_TOKENS = 5;

// Metadata kept completely separate from the dense vectors
struct SLBMeta {
    uint32_t tool_id;
    float scale;
    int32_t scent[SCENT_TOKENS];
};

class NexusSemanticSLB {
private:
    size_t dim_;             // Padded dimension (multiple of 32 or 16)
    size_t original_dim_;    // User-provided original dimension
    // SoA Layout:
    std::vector<SLBMeta> metadata_;          // Size N
    std::vector<int8_t> quantized_vectors_;  // Contiguous flat array, Size N * dim_

public:
    explicit NexusSemanticSLB(size_t dim);
    
    // Disable copy for memory efficiency
    NexusSemanticSLB(const NexusSemanticSLB&) = delete;
    NexusSemanticSLB& operator=(const NexusSemanticSLB&) = delete;

    NexusSemanticSLB(NexusSemanticSLB&&) noexcept = default;
    NexusSemanticSLB& operator=(NexusSemanticSLB&&) noexcept = default;

    // Quantizes FP32 vector and registers it (Offline/Init phase)
    void register_tool(uint32_t tool_id, std::span<const float> fp32_vector, std::span<const int32_t, SCENT_TOKENS> scent);
    
    struct SearchResult {
        uint32_t tool_id;
        float score;
        const int32_t* scent_tokens; // Pointer to SCENT_TOKENS of length SCENT_TOKENS
    };

    // Hot-path SIMD search. Returns Top-K metadata.
    std::vector<SearchResult> search(std::span<const float> query_vector, size_t top_k = 3) const;

    size_t get_dim() const { return original_dim_; }
    size_t get_padded_dim() const { return dim_; }
    size_t get_count() const { return metadata_.size(); }
};
