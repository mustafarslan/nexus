#pragma once
#include <cstddef>
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

/** LegoLink re-anchor: rotate K rows from compiled absolute positions to target absolute positions.
 *  For linear RoPE this matches apply_relative_rope_shift with delta = target_base - compiled_base;
 *  per-token loop supports future per-position YaRN extensions. */
void apply_absolute_rope_reanchor(
    uint16_t* k_data,
    uint32_t seq_len,
    uint32_t n_head_kv,
    uint32_t d_head,
    uint32_t compiled_base_pos,
    uint32_t target_base_pos,
    float freq_base,
    float freq_scale,
    uint32_t scaling_type,
    float ext_factor,
    float beta_fast,
    float beta_slow,
    int n_ctx_orig);

/** CacheBlend-inspired log-denominator deficit estimate over prefix K vs spliced block K.
 *  Writes per-KV-cell additive biases into out_bias[kv_len]. */
void apply_denominator_calibration(
    const float* prefix_k_norms,
    size_t prefix_len,
    const float* splice_k_norms,
    size_t splice_len,
    float* out_bias,
    size_t kv_len,
    float n_past_scale);

/** Extended calibration: KQ additive bias + per-position V multiplicative scales. */
void apply_denominator_calibration_v2(
    const float* prefix_k_norms,
    size_t prefix_len,
    const float* splice_k_norms,
    size_t splice_len,
    float* out_bias,
    float* out_v_scales,
    size_t kv_len,
    float n_past_scale,
    float target_p,
    float compiled_p);
