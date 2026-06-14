#include "nexus_slb.hpp"
#include <cmath>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <ggml.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

template <typename T, std::size_t Alignment>
struct AlignedAllocator {
    using value_type = T;
    
    AlignedAllocator() noexcept = default;
    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}
    
    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };
    
    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        void* ptr = nullptr;
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }
    
    void deallocate(T* p, std::size_t) noexcept {
        free(p);
    }
};

template <typename T, typename U, std::size_t Alignment>
bool operator==(const AlignedAllocator<T, Alignment>&, const AlignedAllocator<U, Alignment>&) { return true; }

template <typename T, typename U, std::size_t Alignment>
bool operator!=(const AlignedAllocator<T, Alignment>&, const AlignedAllocator<U, Alignment>&) { return false; }

NexusSemanticSLB::NexusSemanticSLB(size_t dim, size_t max_tool_tokens)
    : original_dim_(dim), max_tool_tokens_(max_tool_tokens) {
    dim_ = ((dim + 31) / 32) * 32;
}

void NexusSemanticSLB::register_tool(uint32_t tool_id, std::span<const float> fp32_vector, std::span<const int32_t> digest_tokens,
                                       std::span<const uint32_t> lexical_hashes) {
    if (fp32_vector.size() != original_dim_) {
        throw std::invalid_argument("NexusSemanticSLB::register_tool: Input vector size does not match SLB original dimension.");
    }
    
    // Calculate scaling factor with machine epsilon bounding to prevent divide-by-zero (NaN poisoning)
    float max_val = 0.0f;
    for (float val : fp32_vector) {
        max_val = std::max(max_val, std::abs(val));
    }
    max_val = std::max(max_val, 1e-4f);
    float scale = max_val / 127.0f;
    
    // Quantize the vector to INT8 and append to flat array, zero-padded to dim_
    size_t prev_size = quantized_vectors_.size();
    quantized_vectors_.resize(prev_size + dim_, 0);
    
    int32_t tool_sum = 0;
    for (size_t i = 0; i < original_dim_; ++i) {
        float scaled = fp32_vector[i] / scale;
        float rounded = std::round(scaled);
        int8_t quantized_val = static_cast<int8_t>(std::clamp(static_cast<int>(rounded), -127, 127));
        quantized_vectors_[prev_size + i] = quantized_val;
        tool_sum += quantized_val;
    }

    uint32_t digest_offset = static_cast<uint32_t>(digest_arena_.size());
    uint32_t digest_len = static_cast<uint32_t>(digest_tokens.size());
    digest_arena_.insert(digest_arena_.end(), digest_tokens.begin(), digest_tokens.end());

    uint32_t lexical_offset = static_cast<uint32_t>(lexical_arena_.size());
    uint32_t lexical_len = static_cast<uint32_t>(lexical_hashes.size());
    lexical_arena_.insert(lexical_arena_.end(), lexical_hashes.begin(), lexical_hashes.end());

    // Populate metadata SoA element
    SLBMeta meta;
    meta.tool_id = tool_id;
    meta.scale = scale;
    meta.tool_sum = tool_sum;
    meta.digest_offset = digest_offset;
    meta.digest_len = digest_len;
    meta.lexical_offset = lexical_offset;
    meta.lexical_len = lexical_len;
    metadata_.push_back(meta);
}

static float lexical_overlap_score(const std::vector<uint32_t>& tool_hashes,
                                   std::span<const uint32_t> query_hashes) {
    if (tool_hashes.empty() || query_hashes.empty()) {
        return 0.0f;
    }
    float score = 0.0f;
    for (uint32_t qh : query_hashes) {
        for (uint32_t th : tool_hashes) {
            if (qh == th) {
                score += 1.0f;
                break;
            }
        }
    }
    return score;
}

// 100% unrolled, branchless SIMD sequence without loop remainders
static int32_t dot_product_int8(const int8_t* a, const int8_t* b, size_t dim, int32_t tool_sum) {
#if defined(__ARM_NEON)
    (void)tool_sum;
    int32x4_t accum = vdupq_n_s32(0);
    
    #if defined(__ARM_FEATURE_DOTPROD)
        for (size_t i = 0; i < dim; i += 16) {
            int8x16_t va = vld1q_s8(a + i);
            int8x16_t vb = vld1q_s8(b + i);
            accum = vdotq_s32(accum, va, vb);
        }
    #else
        for (size_t i = 0; i < dim; i += 16) {
            int8x16_t va = vld1q_s8(a + i);
            int8x16_t vb = vld1q_s8(b + i);
            int16x8_t va16_lo = vmovl_s8(vget_low_s8(va));
            int16x8_t va16_hi = vmovl_s8(vget_high_s8(va));
            int16x8_t vb16_lo = vmovl_s8(vget_low_s8(vb));
            int16x8_t vb16_hi = vmovl_s8(vget_high_s8(vb));
            accum = vmlal_s16(accum, vget_low_s16(va16_lo), vget_low_s16(vb16_lo));
            accum = vmlal_s16(accum, vget_high_s16(va16_lo), vget_high_s16(vb16_lo));
            accum = vmlal_s16(accum, vget_low_s16(va16_hi), vget_low_s16(vb16_hi));
            accum = vmlal_s16(accum, vget_high_s16(va16_hi), vget_high_s16(vb16_hi));
        }
    #endif

    int32_t total = 0;
    #if defined(__aarch64__)
        total = vaddvq_s32(accum);
    #else
        total = vgetq_lane_s32(accum, 0) + vgetq_lane_s32(accum, 1) + 
                vgetq_lane_s32(accum, 2) + vgetq_lane_s32(accum, 3);
    #endif
    return total;

#elif defined(__AVX512VNNI__) || defined(__AVXVNNI__)
    __m256i accum = _mm256_setzero_si256();
    __m256i shift = _mm256_set1_epi8(128);
    for (size_t i = 0; i < dim; i += 32) {
        __m256i q_s = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i t_s = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        __m256i q_u = _mm256_add_epi8(q_s, shift);
        accum = _mm256_dpbusd_epi32(accum, q_u, t_s);
    }
    
    __m128i lo = _mm256_castsi256_si128(accum);
    __m128i hi = _mm256_extracti128_si256(accum, 1);
    __m128i sum128 = _mm_add_epi32(lo, hi);
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(0, 1, 0, 1)));
    int32_t final_dot_product = _mm_cvtsi128_si32(sum128);
    final_dot_product -= (128 * tool_sum);
    return final_dot_product;

#elif defined(__AVX2__)
    (void)tool_sum;
    __m256i accum = _mm256_setzero_si256();
    size_t i = 0;
    for (; i < dim; i += 32) {
        __m256i va_lo = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i)));
        __m256i vb_lo = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i)));
        __m256i prod_lo = _mm256_madd_epi16(va_lo, vb_lo);
        
        __m256i va_hi = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i + 16)));
        __m256i vb_hi = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i + 16)));
        __m256i prod_hi = _mm256_madd_epi16(va_hi, vb_hi);
        
        accum = _mm256_add_epi32(accum, _mm256_add_epi32(prod_lo, prod_hi));
    }
    
    __m128i lo = _mm256_castsi256_si128(accum);
    __m128i hi = _mm256_extracti128_si256(accum, 1);
    __m128i sum128 = _mm_add_epi32(lo, hi);
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(0, 1, 0, 1)));
    return _mm_cvtsi128_si32(sum128);

#else
    (void)tool_sum;
    int32_t total = 0;
    for (size_t i = 0; i < dim; ++i) {
        total += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    }
    return total;
#endif
}

std::vector<NexusSemanticSLB::SearchResult> NexusSemanticSLB::search(std::span<const float> query_vector, size_t top_k) const {
    if (query_vector.size() != original_dim_) {
        throw std::invalid_argument("NexusSemanticSLB::search: Query vector size does not match SLB original dimension.");
    }
    if (metadata_.empty() || top_k == 0) {
        return {};
    }
    
    // Quantize query vector with machine epsilon bounding to prevent divide-by-zero (NaN poisoning)
    float max_val = 0.0f;
    for (float val : query_vector) {
        max_val = std::max(max_val, std::abs(val));
    }
    max_val = std::max(max_val, 1e-4f); // Upgraded from 1e-8f to avoid FP16 subnormals
    float query_scale = max_val / 127.0f;
    
    static thread_local std::vector<int8_t, AlignedAllocator<int8_t, 32>> tls_scratch_buffer;
    static thread_local uint32_t tls_underutilization_ticks = 0;
    if (tls_scratch_buffer.size() < dim_) {
        tls_scratch_buffer.resize(dim_, 0);
    }
    for (size_t i = 0; i < original_dim_; ++i) {
        float scaled = query_vector[i] / query_scale;
        float rounded = std::round(scaled);
        tls_scratch_buffer[i] = static_cast<int8_t>(std::clamp(static_cast<int>(rounded), -127, 127));
    }
    if (dim_ > original_dim_) {
        std::memset(tls_scratch_buffer.data() + original_dim_, 0, dim_ - original_dim_);
    }
    
    // Internal struct for zero-allocation sorting
    struct TempResult {
        uint32_t tool_id = 0;
        float score = -1e20f;
        uint32_t digest_offset = 0;
        uint32_t digest_len = 0;
    };
    
    constexpr size_t MAX_K = 16;
    size_t k = std::min(std::min(top_k, MAX_K), metadata_.size());
    
    TempResult best_results[MAX_K];
    
    for (size_t idx = 0; idx < metadata_.size(); ++idx) {
        const auto& meta = metadata_[idx];
        const int8_t* tool_vec = quantized_vectors_.data() + idx * dim_;
        
        int32_t int_dot = dot_product_int8(tls_scratch_buffer.data(), tool_vec, dim_, meta.tool_sum);
        
        // Dequantize score = int_dot * s_tool * s_query
        float score = static_cast<float>(int_dot) * meta.scale * query_scale;
        
        size_t insert_pos = MAX_K;
        for (size_t j = 0; j < k; ++j) {
            if (score > best_results[j].score) {
                insert_pos = j;
                break;
            }
        }
        
        if (insert_pos < k) {
            for (size_t j = k - 1; j > insert_pos; --j) {
                best_results[j] = best_results[j - 1];
            }
            best_results[insert_pos].tool_id = meta.tool_id;
            best_results[insert_pos].score = score;
            best_results[insert_pos].digest_offset = meta.digest_offset;
            best_results[insert_pos].digest_len = meta.digest_len;
        }
    }
    
    std::vector<SearchResult> results;
    results.reserve(k);
    for (size_t i = 0; i < k; ++i) {
        auto span = get_digest_tokens(best_results[i].digest_offset, best_results[i].digest_len);
        std::vector<int32_t> tokens(span.begin(), span.end());
        results.push_back({
            best_results[i].tool_id,
            best_results[i].score,
            best_results[i].digest_offset,
            best_results[i].digest_len,
            std::move(tokens)
        });
    }
    
    // Phase 53: TLS High-Watermark Protection.
    // If a prior search on this thread inflated the buffer beyond 64KB
    // (e.g., a rogue SLB with a massive dim_), release the backing memory
    // to prevent long-term TLS exhaustion in multi-tenant deployments.
    // Phase 63: Hysteresis Memory Management (True Anti-Thrashing)
    // Track underutilization over time rather than punishing a single small request.
    constexpr size_t TLS_HIGH_WATERMARK = 65536;
    if (tls_scratch_buffer.capacity() > TLS_HIGH_WATERMARK && 
        tls_scratch_buffer.capacity() > (dim_ * 2)) {
        tls_underutilization_ticks++;
    } else {
        tls_underutilization_ticks = 0; // Reset immediately on a large payload
    }

    if (tls_underutilization_ticks > 100) [[unlikely]] {
        decltype(tls_scratch_buffer) empty;
        tls_scratch_buffer.swap(empty);
        tls_underutilization_ticks = 0;
    }
    
    return results;
}
std::vector<NexusSemanticSLB::SearchResult> NexusSemanticSLB::search_hybrid(
    std::span<const float> query_vector,
    std::span<const uint32_t> query_lexical_hashes,
    size_t top_k) const {
    constexpr float RRF_K = 60.0f;
    auto dense = search(query_vector, std::max(top_k, size_t{8}));

    struct Ranked {
        uint32_t tool_id;
        float rrf;
        uint32_t digest_offset;
        uint32_t digest_len;
    };
    std::unordered_map<uint32_t, Ranked> fused;

    for (size_t rank = 0; rank < dense.size(); ++rank) {
        const auto& r = dense[rank];
        float contrib = 1.0f / (RRF_K + static_cast<float>(rank) + 1.0f);
        fused[r.tool_id] = Ranked{r.tool_id, contrib, r.digest_offset, r.digest_len};
    }

    std::vector<std::pair<float, size_t>> lexical_scores;
    lexical_scores.reserve(metadata_.size());
    for (size_t idx = 0; idx < metadata_.size(); ++idx) {
        const auto& meta = metadata_[idx];
        std::vector<uint32_t> tool_hashes(
            lexical_arena_.begin() + meta.lexical_offset,
            lexical_arena_.begin() + meta.lexical_offset + meta.lexical_len);
        float lex = lexical_overlap_score(tool_hashes, query_lexical_hashes);
        lexical_scores.emplace_back(lex, idx);
    }
    std::sort(lexical_scores.begin(), lexical_scores.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    size_t lex_k = std::min(metadata_.size(), std::max(top_k, size_t{8}));
    for (size_t rank = 0; rank < lex_k; ++rank) {
        const auto& meta = metadata_[lexical_scores[rank].second];
        float contrib = 1.0f / (RRF_K + static_cast<float>(rank) + 1.0f);
        auto it = fused.find(meta.tool_id);
        if (it == fused.end()) {
            fused[meta.tool_id] = Ranked{meta.tool_id, contrib, meta.digest_offset, meta.digest_len};
        } else {
            it->second.rrf += contrib;
        }
    }

    std::vector<Ranked> ranked;
    ranked.reserve(fused.size());
    for (const auto& kv : fused) {
        ranked.push_back(kv.second);
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const Ranked& a, const Ranked& b) { return a.rrf > b.rrf; });

    size_t k = std::min(top_k, ranked.size());
    std::vector<SearchResult> results;
    results.reserve(k);
    for (size_t i = 0; i < k; ++i) {
        auto span = get_digest_tokens(ranked[i].digest_offset, ranked[i].digest_len);
        std::vector<int32_t> tokens(span.begin(), span.end());
        results.push_back({
            ranked[i].tool_id,
            ranked[i].rrf,
            ranked[i].digest_offset,
            ranked[i].digest_len,
            std::move(tokens),
        });
    }
    return results;
}
