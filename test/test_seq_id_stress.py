import sys
import os
import concurrent.futures
import numpy as np

# Add build directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))

from nexus_fsm_ext import NexusSemanticSLB, NexusRadixFSM, NexusOrchestrator

def stress_test_seq_id():
    print("Initializing test dependencies...")
    dim = 64
    slb = NexusSemanticSLB(dim)
    fsm = NexusRadixFSM()
    
    # Instantiate with dummy context pointer (0x12345678)
    ctx_addr = 0x12345678
    orchestrator = NexusOrchestrator(ctx_addr, slb, fsm, base_pos=256)
    
    print("Hammering orchestrator release_hazard concurrently across 20,000 sequence IDs...")
    
    def worker(seq_id):
        # Trigger hazard releasing
        orchestrator.release_hazard(seq_id)
        
    num_threads = 64
    num_seqs = 20000
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, seq_id) for seq_id in range(num_seqs)]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    print("Successfully ran monotonic seq_id stress test with no crashes or segfaults!")

if __name__ == "__main__":
    stress_test_seq_id()
