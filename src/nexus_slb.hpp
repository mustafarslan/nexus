#pragma once
#include <cstdint>
#include <vector>
#include <span>

struct SLBMeta {
    uint32_t tool_id;
    float scale;
    uint32_t digest_offset;
    uint32_t digest_len;
    int32_t tool_sum;
    uint32_t lexical_offset;
    uint32_t lexical_len;
};

class NexusSemanticSLB {
private:
    size_t dim_;
    size_t original_dim_;
    size_t max_tool_tokens_ = 32;
    std::vector<SLBMeta> metadata_;
    std::vector<int8_t> quantized_vectors_;
    std::vector<int32_t> digest_arena_;
    std::vector<uint32_t> lexical_arena_;

public:
    explicit NexusSemanticSLB(size_t dim, size_t max_tool_tokens = 32);

    NexusSemanticSLB(const NexusSemanticSLB&) = delete;
    NexusSemanticSLB& operator=(const NexusSemanticSLB&) = delete;

    NexusSemanticSLB(NexusSemanticSLB&&) noexcept = default;
    NexusSemanticSLB& operator=(NexusSemanticSLB&&) noexcept = default;

    struct SearchResult {
        uint32_t tool_id;
        float score;
        uint32_t digest_offset;
        uint32_t digest_len;
        std::vector<int32_t> scent_tokens;
    };

    void register_tool(uint32_t tool_id, std::span<const float> fp32_vector, std::span<const int32_t> digest_tokens,
                       std::span<const uint32_t> lexical_hashes = {});

    std::vector<SearchResult> search(std::span<const float> query_vector, size_t top_k = 3) const;

    std::vector<SearchResult> search_hybrid(std::span<const float> query_vector,
                                            std::span<const uint32_t> query_lexical_hashes,
                                            size_t top_k = 3) const;

    size_t get_dim() const { return original_dim_; }
    size_t get_padded_dim() const { return dim_; }
    size_t get_count() const { return metadata_.size(); }
    size_t get_max_tool_tokens() const { return max_tool_tokens_; }

    std::span<const int32_t> get_digest_tokens(uint32_t offset, uint32_t len) const {
        if (offset + len > digest_arena_.size()) {
            return {};
        }
        return std::span<const int32_t>(digest_arena_.data() + offset, len);
    }
};
