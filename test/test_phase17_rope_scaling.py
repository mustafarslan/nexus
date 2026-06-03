import subprocess
import struct
import os
import pytest

def test_atb_rope_scaling_fields():
    # Define structural layout matching AeonToolBlockHeader in C++
    # magic (4s), version (I), model_hash (Q), n_layer (I), n_head_kv (I), d_head (I),
    # seq_len (I), base_pos (I), rope_freq_base (f), rope_freq_scale (f), rope_scaling_type (I),
    # ggml_type_k (I), ggml_type_v (I), k_tensor_offset (Q), v_tensor_offset (Q),
    # k_total_bytes (Q), v_total_bytes (Q), padding (40s)
    
    # 16296s (16296) = 16384 bytes total.
    
    fmt = "<4sI Q III IIffI II QQQQ 40s"
    assert struct.calcsize(fmt) == 128
    
    # Configure tool block settings
    magic = b"ATB1"
    version = 1
    model_hash = 0xabcdef123456
    n_layer = 2
    n_head_kv = 4
    d_head = 8
    seq_len = 16
    base_pos = 128
    rope_freq_base = 10000.0
    rope_freq_scale = 0.5
    rope_scaling_type = 2 # YaRN
    ggml_type_k = 1 # GGML_TYPE_F16
    ggml_type_v = 1
    
    k_total_bytes = n_layer * n_head_kv * seq_len * d_head * 2
    v_total_bytes = k_total_bytes
    k_tensor_offset = 2097152
    k_total_bytes_aligned = ((k_total_bytes + 2097151) // 2097152) * 2097152
    v_tensor_offset = k_tensor_offset + k_total_bytes_aligned
    
    padding = b"\x00" * 40
    
    header = struct.pack(
        fmt,
        magic, version, model_hash, n_layer, n_head_kv, d_head,
        seq_len, base_pos, rope_freq_base, rope_freq_scale, rope_scaling_type,
        ggml_type_k, ggml_type_v, k_tensor_offset, v_tensor_offset,
        k_total_bytes, v_total_bytes, padding
    )
    
    # Write mock ATB file
    atb_path = "build/mock_test_rope.atb"
    os.makedirs("build", exist_ok=True)
    with open(atb_path, "wb") as f:
        f.write(header)
        f.write(b"\x00" * (2097152 - 128)) # Pad header to 2MB boundary
        # Write dummy K tensor data
        f.write(b"\x00" * k_total_bytes)
        f.write(b"\x00" * (k_total_bytes_aligned - k_total_bytes))
        # Write dummy V tensor data
        f.write(b"\x00" * v_total_bytes)
        v_total_bytes_aligned = ((v_total_bytes + 2097151) // 2097152) * 2097152
        f.write(b"\x00" * (v_total_bytes_aligned - v_total_bytes))
        
    try:
        # Run verify_atb
        binary_path = "./build/verify_atb"
        result = subprocess.run([binary_path, atb_path], capture_output=True, text=True, check=True)
        
        # Verify output prints the correct parameters
        output = result.stdout
        print(output)
        
        assert "Verification Status: SUCCESS" in output
        assert "RoPE Freq Scale:  0.5" in output
        assert "RoPE Scaling Type:2" in output
        assert "Base Position:    128" in output
        assert "Model Hash:       0xabcdef123456" in output
        
    finally:
        if os.path.exists(atb_path):
            os.remove(atb_path)

if __name__ == "__main__":
    test_atb_rope_scaling_fields()
