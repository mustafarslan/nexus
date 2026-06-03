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

def test_lru_cache_concurrency():
    print("\n[Concurrency Verification] Running test_lru_cache_concurrency...")
    
    # Create 5 temporary ATB files of varying sizes
    temp_dir = tempfile.gettempdir()
    atb_files = [os.path.join(temp_dir, f"temp_tool_{i}.atb") for i in range(5)]
    for i, path in enumerate(atb_files):
        create_temp_atb_file(path, 128 + i * 32)

    # Initialize a cache with capacity 2 (forces frequent evictions)
    cache = NexusBlockCache(2)

    def hammer_cache(file_path):
        # 1. Load block
        block = cache.get_or_load(file_path)
        assert block is not None
        assert block.get_file_size() > 0
        
        # 2. Asynchronously prefetch
        cache.prefetch(file_path)
        
        # 3. Retrieve some stats
        seq_len = block.seq_len
        assert seq_len > 0

    # Spawn 100 threads hammering the LRU cache concurrently
    num_threads = 100
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        # Submit 500 tasks (each of 5 files fetched 100 times)
        for _ in range(100):
            for path in atb_files:
                futures.append(executor.submit(hammer_cache, path))
                
        # Wait for all threads to complete
        for future in concurrent.futures.as_completed(futures):
            future.result() # raises exception if thread failed

    # Clean up temp files
    for path in atb_files:
        if os.path.exists(path):
            os.remove(path)
            
    print("  Successfully executed 500 concurrent cache operations across 100 threads with zero crashes.")
    print("test_lru_cache_concurrency passed!")

if __name__ == "__main__":
    test_lru_cache_concurrency()
