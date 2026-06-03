#include "nexus_rope_math.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include "ggml.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline float get_rope_frequency(
    int64_t i0, int d_head, float freq_base, float freq_scale,
    uint32_t scaling_type, float ext_factor, float beta_fast, float beta_slow, int n_ctx_orig) {
    
    float theta_extrap = std::pow(freq_base, -2.0f * (float)(i0 / 2) / d_head);
    if (scaling_type == 2) { // YaRN scaling
        float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        if (ext_factor != 0.0f) {
            // YaRN Correction dims
            float start = std::floor((float)d_head * std::logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / (2.0f * std::logf(freq_base)));
            float end   = std::ceil((float)d_head * std::logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / (2.0f * std::logf(freq_base)));
            float corr_dims[2];
            corr_dims[0] = std::max(0.0f, start);
            corr_dims[1] = std::min((float)d_head - 1.0f, end);
            
            // ramp mix calculation
            float y = ((float)i0 - corr_dims[0]) / std::max(0.001f, corr_dims[1] - corr_dims[0]);
            float ramp_mix = (1.0f - std::min(1.0f, std::max(0.0f, y))) * ext_factor;
            
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        }
        return theta;
    } else {
        // Standard (scaling_type=0) or Linear scaling (scaling_type=1)
        return freq_scale * theta_extrap;
    }
}

void apply_relative_rope_shift(
    uint16_t* k_data,
    uint32_t seq_len,
    uint32_t n_head_kv,
    uint32_t d_head,
    int32_t delta_pos,
    float freq_base,
    float freq_scale,
    uint32_t scaling_type,
    float ext_factor,
    float beta_fast,
    float beta_slow,
    int n_ctx_orig
) {
    if (delta_pos == 0) return;

    // Precompute cos/sin values for all d_head/2 dimension pairs
    std::vector<float> cos_vals(d_head / 2);
    std::vector<float> sin_vals(d_head / 2);
    for (uint32_t i = 0; i < d_head / 2; ++i) {
        float theta = get_rope_frequency(2 * i, d_head, freq_base, freq_scale, scaling_type, ext_factor, beta_fast, beta_slow, n_ctx_orig);
        float phi = (float)delta_pos * theta;
        cos_vals[i] = std::cos(phi);
        sin_vals[i] = std::sin(phi);
    }

    // Auto-vectorization friendly contiguous inner loops
    for (uint32_t s = 0; s < seq_len; ++s) {
        for (uint32_t h = 0; h < n_head_kv; ++h) {
            uint16_t* head_ptr = k_data + (s * n_head_kv * d_head) + (h * d_head);
            for (uint32_t i = 0; i < d_head / 2; ++i) {
                float x0 = ggml_fp16_to_fp32(head_ptr[2 * i]);
                float x1 = ggml_fp16_to_fp32(head_ptr[2 * i + 1]);

                float r0 = x0 * cos_vals[i] - x1 * sin_vals[i];
                float r1 = x0 * sin_vals[i] + x1 * cos_vals[i];

                head_ptr[2 * i]     = ggml_fp32_to_fp16(r0);
                head_ptr[2 * i + 1] = ggml_fp32_to_fp16(r1);
            }
        }
    }
}
