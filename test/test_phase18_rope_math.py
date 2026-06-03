import sys
import os
import torch
import numpy as np
import pytest

# Add the build directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))

import nexus_fsm_ext

def pytorch_apply_rope(k, positions, freq_base=10000.0, freq_scale=1.0, scaling_type=0):
    """
    Applies absolute RoPE rotation to a Key tensor K of shape [seq_len, n_head_kv, d_head].
    """
    seq_len, n_head_kv, d_head = k.shape
    dim = d_head // 2
    
    # theta_i = base^(-2i/d)
    theta = 1.0 / (freq_base ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head))
    if scaling_type == 1:  # Linear scaling
        theta = theta * freq_scale
        
    t = torch.tensor(positions, dtype=torch.float32).unsqueeze(1) # [seq_len, 1]
    freqs = t @ theta.unsqueeze(0) # [seq_len, dim]
    
    cos = torch.cos(freqs).unsqueeze(1) # [seq_len, 1, dim]
    sin = torch.sin(freqs).unsqueeze(1) # [seq_len, 1, dim]
    
    k0 = k[..., 0::2]
    k1 = k[..., 1::2]
    
    k_rot = torch.zeros_like(k)
    k_rot[..., 0::2] = k0 * cos - k1 * sin
    k_rot[..., 1::2] = k0 * sin + k1 * cos
    return k_rot

def test_relative_rope_shift_lossless():
    print("\n[Rope Math Verification] Running test_relative_rope_shift_lossless...")
    
    seq_len = 16
    n_head_kv = 8
    d_head = 128
    
    freq_base = 10000.0
    freq_scale = 1.0
    scaling_type = 0 # Standard
    
    # Start position A and Target position B
    pos_a = 5
    pos_b = 12
    delta_pos = pos_b - pos_a
    
    # 1. Generate random raw key values
    torch.manual_seed(42)
    k_raw = torch.randn(seq_len, n_head_kv, d_head, dtype=torch.float32)
    
    # 2. Compute absolute RoPE at Position A (representing offline compilation start)
    positions_a = [pos_a + i for i in range(seq_len)]
    k_pos_a = pytorch_apply_rope(k_raw, positions_a, freq_base, freq_scale, scaling_type)
    
    # 3. Compute absolute RoPE natively at Position B (representing live context placement)
    positions_b = [pos_b + i for i in range(seq_len)]
    k_pos_b_native = pytorch_apply_rope(k_raw, positions_b, freq_base, freq_scale, scaling_type)
    
    # 4. Pass the Position A tensor to the C++ relative RoPE shift kernel
    # C++ kernel expects FP16 representation (uint16_t)
    k_pos_a_np = k_pos_a.numpy().astype(np.float16)
    k_pos_a_uint16 = k_pos_a_np.view(np.uint16)
    
    # Call the C++ shift FFI kernel
    nexus_fsm_ext.apply_relative_rope_shift(
        k_pos_a_uint16,
        seq_len,
        n_head_kv,
        d_head,
        delta_pos,
        freq_base,
        freq_scale,
        scaling_type,
        1.0,  # ext_factor
        32.0, # beta_fast
        1.0,  # beta_slow
        4096  # n_ctx_orig
    )
    
    # Convert back to float32 PyTorch tensor
    k_pos_b_shifted_np = k_pos_a_uint16.view(np.float16).astype(np.float32)
    k_pos_b_shifted = torch.from_numpy(k_pos_b_shifted_np)
    
    # 5. Assert Cosine Similarity > 0.9999
    cos_sim = torch.nn.functional.cosine_similarity(
        k_pos_b_native.flatten(),
        k_pos_b_shifted.flatten(),
        dim=0
    ).item()
    
    print(f"  Cosine Similarity between C++ Shifted and PyTorch Native: {cos_sim:.7f}")
    assert cos_sim > 0.9999, f"RoPE shift mathematical degradation detected! Cosine similarity: {cos_sim}"
    print("test_relative_rope_shift_lossless passed!")

if __name__ == "__main__":
    test_relative_rope_shift_lossless()
