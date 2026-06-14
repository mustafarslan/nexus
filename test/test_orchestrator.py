import sys
import os
import numpy as np
try:
    import pytest
except ImportError:
    pytest = None


# Add the build directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))

import nexus_fsm_ext
from nexus_fsm_ext import NexusSemanticSLB, NexusRadixFSM, NexusOrchestrator

def test_orchestrator_bindings():
    print("Running test_orchestrator_bindings...")
    
    # Instantiate dependencies
    dim = 64
    slb = NexusSemanticSLB(dim)
    fsm = NexusRadixFSM()
    
    # Test registration of tools in FSM
    # Tool 1: [10, 20]
    fsm.add_route(1, [10, 20])
    
    # Tool 1 embedding and scent
    vec = np.random.randn(dim).astype(np.float32)
    scent = [1, 2, 3, 4, 5]
    slb.register_tool(1, vec, scent)
    
    # We pass a dummy integer pointer (e.g. 0x12345678) for the llama_context* since we don't load a GGUF model here.
    # In a full run, we would pass a valid llama_context address.
    ctx_addr = 0x12345678
    
    orchestrator = NexusOrchestrator(ctx_addr, slb, fsm, base_pos=256)
    
    # Test registering tool path
    path = "build/tool_1.atb"
    orchestrator.register_tool_path(1, path)
    
    print("  Orchestrator successfully instantiated.")
    print("  Tool path registered successfully.")
    print("test_orchestrator_bindings passed!")

def test_orchestrator_null_context():
    print("Running test_orchestrator_null_context...")
    dim = 64
    slb = NexusSemanticSLB(dim)
    fsm = NexusRadixFSM()
    
    # Test that passing None raises TypeError
    try:
        orchestrator = NexusOrchestrator(None, slb, fsm, base_pos=256)
        assert False, "Should have failed with TypeError for None context"
    except TypeError as e:
        print(f"  Successfully caught exception for None context: {e}")
        
    # Test that passing 0 raises ValueError
    try:
        orchestrator = NexusOrchestrator(0, slb, fsm, base_pos=256)
        assert False, "Should have failed with ValueError for 0 context"
    except ValueError as e:
        print(f"  Successfully caught exception for nullptr context: {e}")
        
    print("test_orchestrator_null_context passed!")

if __name__ == "__main__":
    test_orchestrator_bindings()
    test_orchestrator_null_context()


