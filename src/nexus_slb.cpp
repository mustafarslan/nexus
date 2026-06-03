#include "nexus_slb.hpp"
#include <cmath>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

NexusSemanticSLB::NexusSemanticSLB(size_t dim) : original_dim_(dim) {
#if defined(__AVX2__)
    dim_ = ((dim + 31) / 32) * 32;
#else
    dim_ = ((dim + 15) / 16) * 16;
#endif
}

void NexusSemanticSLB::register_tool(uint32_t tool_id, std::span<const float> fp32_vector, std::span<const int32_t, SCENT_TOKENS> scent) {
    if (fp32_vector.size() != original_dim_) {
        throw std::invalid_argument("NexusSemanticSLB::register_tool: Input vector size does not match SLB original dimension.");
    }
    
    // Calculate scaling factor with machine epsilon bounding to prevent divide-by-zero (NaN poisoning)
    float max_val = 0.0f;
    for (float val : fp32_vector) {
        max_val = std::max(max_val, std::abs(val));
    }
    max_val = std::max(max_val, 1e-8f);
    float scale = max_val / 127.0f;
    
    // Populate metadata SoA element
    SLBMeta meta;
    meta.tool_id = tool_id;
    meta.scale = scale;
    std::memcpy(meta.scent, scent.data(), sizeof(meta.scent));
    metadata_.push_back(meta);
    
    // Quantize the vector to INT8 and append to flat array, zero-padded to dim_
    size_t prev_size = quantized_vectors_.size();
    quantized_vectors_.resize(prev_size + dim_, 0);
    
    for (size_t i = 0; i < original_dim_; ++i) {
        float scaled = fp32_vector[i] / scale;
        float rounded = std::round(scaled);
        quantized_vectors_[prev_size + i] = static_cast<int8_t>(std::clamp(static_cast<int>(rounded), -127, 127));
    }
}

// 100% unrolled, branchless SIMD sequence without loop remainders
static int32_t dot_product_int8(const int8_t* a, const int8_t* b, size_t dim) {
#if defined(__ARM_NEON)
    int32x4_t accum = vdupq_n_s32(0);
    size_t i = 0;
    
    #if defined(__ARM_FEATURE_DOTPROD)
        for (; i < dim; i += 16) {
            int8x16_t va = vld1q_s8(a + i);
            int8x16_t vb = vld1q_s8(b + i);
            accum = vdotq_s32(accum, va, vb);
        }
    #else
        for (; i < dim; i += 8) {
            int8x8_t va = vld1_s8(a + i);
            int8x8_t vb = vld1_s8(b + i);
            int16x8_t va16 = vmovl_s8(va);
            int16x8_t vb16 = vmovl_s8(vb);
            accum = vmlal_s16(accum, vget_low_s16(va16), vget_low_s16(vb16));
            accum = vmlal_s16(accum, vget_high_s16(va16), vget_high_s16(vb16));
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

#elif defined(__AVX2__)
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
    max_val = std::max(max_val, 1e-8f);
    float query_scale = max_val / 127.0f;
    
    std::vector<int8_t> quantized_query(dim_, 0);
    for (size_t i = 0; i < original_dim_; ++i) {
        float scaled = query_vector[i] / query_scale;
        float rounded = std::round(scaled);
        quantized_query[i] = static_cast<int8_t>(std::clamp(static_cast<int>(rounded), -127, 127));
    }
    
    // Internal struct for zero-allocation sorting
    struct TempResult {
        uint32_t tool_id = 0;
        float score = -1e20f;
        const int32_t* scent = nullptr;
    };
    
    constexpr size_t MAX_K = 16;
    size_t k = std::min(std::min(top_k, MAX_K), metadata_.size());
    
    TempResult best_results[MAX_K];
    
    for (size_t idx = 0; idx < metadata_.size(); ++idx) {
        const auto& meta = metadata_[idx];
        const int8_t* tool_vec = quantized_vectors_.data() + idx * dim_;
        
        int32_t int_dot = dot_product_int8(quantized_query.data(), tool_vec, dim_);
        
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
            best_results[insert_pos].scent = meta.scent;
        }
    }
    
    std::vector<SearchResult> results;
    results.reserve(k);
    for (size_t i = 0; i < k; ++i) {
        results.push_back({
            best_results[i].tool_id,
            best_results[i].score,
            best_results[i].scent
        });
    }
    
    return results;
}
