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

def test_cache_poisoning_recovery():
    print("\n[Cache Poisoning stress test] Running test_cache_poisoning_recovery...")
    
    temp_dir = tempfile.gettempdir()
    poison_file = os.path.join(temp_dir, "poison_stress_tool.atb")
    
    # Ensure file does not exist initially (triggering open() exception)
    if os.path.exists(poison_file):
        os.remove(poison_file)
        
    cache = NexusBlockCache(1024 * 1024 * 1024)
    
    exceptions_caught = 0
    
    def attempt_load():
        try:
            block = cache.get_or_load(poison_file)
            # Access tensors to force wait_until_resident which might throw asynchronously
            block.get_file_size()
            return "SUCCESS"
        except RuntimeError as e:
            return f"EXCEPTION: {str(e)}"
        except Exception as e:
            return f"UNKNOWN EXCEPTION: {str(e)}"

    # Stress test: spawn 50 threads trying to load the non-existent file
    num_threads = 50
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(attempt_load) for _ in range(num_threads)]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    # Verify that all threads caught the exception and no deadlock occurred
    for res in results:
        assert "EXCEPTION" in res or "RuntimeError" in res
        
    print(f"  All {num_threads} threads cleanly caught loading exceptions without deadlocking.")
    
    # Recovery check: write a VALID file now and see if the cache recovers
    print("  Creating valid ATB file to test recovery...")
    create_temp_atb_file(poison_file, 128)
    
    # Try loading again. If the poisoned future was successfully deleted, this should succeed.
    try:
        block = cache.get_or_load(poison_file)
        assert block is not None
        assert block.get_file_size() > 0
        assert block.seq_len == 128
        print("  Successfully loaded valid ATB file after recovery.")
    except Exception as e:
        pytest.fail(f"Cache failed to recover from poisoning: {e}")
        
    # Clean up
    if os.path.exists(poison_file):
        os.remove(poison_file)
        
    print("test_cache_poisoning_recovery passed!")

if __name__ == "__main__":
    test_cache_poisoning_recovery()
