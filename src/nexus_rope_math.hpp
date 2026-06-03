#pragma once
#include <cstdint>

/**
 * Apply relative RoPE shift to a contiguous Key tensor slice of shape [seq_len, n_head_kv, d_head]
 * in FP16 representation (uint16_t).
 */
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
);
