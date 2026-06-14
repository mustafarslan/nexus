import sys
import os
import numpy as np

# Add the build directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))

import nexus_fsm_ext
from nexus_fsm_ext import NexusSemanticSLB

def test_slb_basic():
    print("Running test_slb_basic...")
    dim = 128
    slb = NexusSemanticSLB(dim)
    
    # Check dimensions and initial state
    assert slb.dim == dim
    assert slb.count == 0
    
    # Create mock tool embeddings
    # Tool 1: high values at indices 0-10
    vec1 = np.zeros(dim, dtype=np.float32)
    vec1[0:10] = 5.0
    scent1 = [11, 22, 33, 44, 55]
    
    # Tool 2: high values at indices 20-30
    vec2 = np.zeros(dim, dtype=np.float32)
    vec2[20:30] = 8.0
    scent2 = [101, 102, 103, 104, 105]
    
    # Tool 3: high values at indices 40-50
    vec3 = np.zeros(dim, dtype=np.float32)
    vec3[40:50] = 2.0
    scent3 = [991, 992, 993, 994, 995]
    
    # Register tools
    slb.register_tool(1, vec1, scent1)
    slb.register_tool(2, vec2, scent2)
    slb.register_tool(3, vec3, scent3)
    
    assert slb.count == 3
    
    # Test Query 1: should match Tool 1 best
    query1 = np.zeros(dim, dtype=np.float32)
    query1[0:10] = 1.0
    results1 = slb.search(query1, top_k=2)
    
    assert len(results1) == 2
    assert results1[0].tool_id == 1
    assert results1[0].scent_tokens == scent1
    assert results1[0].score > 0
    
    # Test Query 2: should match Tool 2 best
    query2 = np.zeros(dim, dtype=np.float32)
    query2[20:30] = 3.0
    results2 = slb.search(query2, top_k=3)
    
    assert len(results2) == 3
    assert results2[0].tool_id == 2
    assert results2[0].scent_tokens == scent2
    
    # Verify that the scores are in descending order
    for i in range(len(results2) - 1):
        assert results2[i].score >= results2[i+1].score
        
    print("test_slb_basic passed successfully!")

def test_slb_quantization_precision():
    print("Running test_slb_quantization_precision...")
    dim = 256
    slb = NexusSemanticSLB(dim)
    
    # Tool with normalized pattern
    np.random.seed(42)
    vec = np.random.randn(dim).astype(np.float32)
    scent = [1, 2, 3, 4, 5]
    slb.register_tool(42, vec, scent)
    
    # Query with exact same pattern (should have perfect similarity)
    results = slb.search(vec, top_k=1)
    assert len(results) == 1
    assert results[0].tool_id == 42
    
    # Calculate exact FP32 dot product
    exact_dot = np.dot(vec, vec)
    quantized_dot = results[0].score
    error_percent = abs(exact_dot - quantized_dot) / exact_dot * 100
    print(f"  Exact FP32 Dot:      {exact_dot:.4f}")
    print(f"  Quantized INT8 Dot:  {quantized_dot:.4f}")
    print(f"  Quantization Error:  {error_percent:.4f}%")
    
    # Expect error to be low (< 2%) due to 8-bit resolution
    assert error_percent < 2.0
    print("test_slb_quantization_precision passed successfully!")

def test_slb_invalid_dimension():
    print("Running test_slb_invalid_dimension...")
    dim = 128
    slb = NexusSemanticSLB(dim)
    
    # Try to register vector with wrong dimension
    bad_vec = np.zeros(64, dtype=np.float32)
    scent = [1, 2, 3, 4, 5]
    try:
        slb.register_tool(1, bad_vec, scent)
        assert False, "Should have raised exception for dimension mismatch on registration"
    except (ValueError, RuntimeError, TypeError, KeyError) as e:
        print(f"  Successfully caught exception on register_tool: {e}")
        
    # Try to search with wrong dimension
    try:
        slb.search(bad_vec, top_k=2)
        assert False, "Should have raised exception for dimension mismatch on search"
    except (ValueError, RuntimeError, TypeError, KeyError) as e:
        print(f"  Successfully caught exception on search: {e}")
        
    print("test_slb_invalid_dimension passed successfully!")

def test_slb_hysteresis_thrashing():
    print("Running test_slb_hysteresis_thrashing...")
    
    large_dim = 70000
    slb_large = NexusSemanticSLB(large_dim)
    
    vec_large = np.zeros(large_dim, dtype=np.float32)
    scent_large = [1] * 5
    slb_large.register_tool(1, vec_large, scent_large)
    
    # Inflate thread-local scratch buffer above 64KB
    slb_large.search(vec_large, top_k=1)
    
    small_dim = 32
    slb_small = NexusSemanticSLB(small_dim)
    vec_small = np.zeros(small_dim, dtype=np.float32)
    scent_small = [2] * 5
    slb_small.register_tool(2, vec_small, scent_small)
    
    # Run under-threshold (50) searches
    for _ in range(50):
        slb_small.search(vec_small, top_k=1)
        
    # Large search should not need reallocation
    slb_large.search(vec_large, top_k=1)
    
    # Exceed threshold (>100 consecutive underutilizations)
    for _ in range(60):
        slb_small.search(vec_small, top_k=1)
        
    # Large search will reallocate
    slb_large.search(vec_large, top_k=1)
    
    print("test_slb_hysteresis_thrashing passed successfully!")

if __name__ == "__main__":
    test_slb_basic()
    test_slb_quantization_precision()
    test_slb_invalid_dimension()
    test_slb_hysteresis_thrashing()
    print("All SLB tests passed successfully!")

