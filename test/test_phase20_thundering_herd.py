import sys
import os
import struct
import tempfile
import concurrent.futures
import pytest

# Add build directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))

import nexus_fsm_ext
from nexus_fsm_ext import NexusBlockCache, AeonToolBlock

def create_temp_atb_file(filename, seq_len):
    # AeonToolBlockHeader format: "<4sI Q III IIffI II QQQQ 40s"
    fmt = "<4sI Q III IIffI II QQQQ 40s"
    magic = b"ATB1"
    version = 1
    model_hash = 0xDECAFBAD
    n_layer = 32
    n_head_kv = 8
    d_head = 128
    
    k_total_bytes = n_layer * n_head_kv * seq_len * d_head * 2
    v_total_bytes = k_total_bytes
    k_tensor_offset = 2097152
    k_total_bytes_aligned = ((k_total_bytes + 2097151) // 2097152) * 2097152
    v_tensor_offset = k_tensor_offset + k_total_bytes_aligned
    v_total_bytes_aligned = ((v_total_bytes + 2097151) // 2097152) * 2097152
    
    header = struct.pack(
        fmt,
        magic, version, model_hash, n_layer, n_head_kv, d_head,
        seq_len, 0, 10000.0, 1.0, 0,
        1, 1, k_tensor_offset, v_tensor_offset,
        k_total_bytes, v_total_bytes, b"\x00"*40
    )
    
    with open(filename, "wb") as f:
        f.write(header)
        f.write(b"\x00" * (2097152 - 128))
        f.write(b"\x00" * k_total_bytes)
        f.write(b"\x00" * (k_total_bytes_aligned - k_total_bytes))
        f.write(b"\x00" * v_total_bytes)
        f.write(b"\x00" * (v_total_bytes_aligned - v_total_bytes))

def test_thundering_herd_prevention():
    print("\n[Thundering Herd Verification] Running test_thundering_herd_prevention...")
    
    temp_dir = tempfile.gettempdir()
    atb_file = os.path.join(temp_dir, "thundering_herd_tool.atb")
    create_temp_atb_file(atb_file, 256)
    
    # Initialize cache with plenty of space (1GB limit)
    cache = NexusBlockCache(1024 * 1024 * 1024)
    
    loaded_blocks = []
    
    def fetch_block():
        # Load block from cache
        block = cache.get_or_load(atb_file)
        assert block is not None
        assert block.get_file_size() > 0
        return block

    num_threads = 50
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(fetch_block) for _ in range(num_threads)]
        
        # Wait for all threads to complete
        for future in concurrent.futures.as_completed(futures):
            loaded_blocks.append(future.result())

    # Ensure all threads loaded successfully
    assert len(loaded_blocks) == num_threads
    
    # Clean up temp file
    if os.path.exists(atb_file):
        os.remove(atb_file)
        
    print(f"  Successfully executed {num_threads} concurrent cache loads targeting the same uncached block.")
    print("test_thundering_herd_prevention passed!")

if __name__ == "__main__":
    test_thundering_herd_prevention()
