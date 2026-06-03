import sys
import os
import numpy as np
import pytest

# Add the build directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))

import nexus_fsm_ext
from nexus_fsm_ext import NexusRadixFSM, RoutingState

def test_concurrent_routing():
    print("Running test_concurrent_routing...")
    fsm = NexusRadixFSM()
    
    # Tool 1: Route [100, 200, 300]
    # Tool 2: Route [400, 500, 600]
    fsm.add_route(1, [100, 200, 300])
    fsm.add_route(2, [400, 500, 600])
    
    # Initialize both sequences
    fsm.reset(seq_id=0)
    fsm.reset(seq_id=1)
    
    fsm.begin_routing(seq_id=0)
    fsm.begin_routing(seq_id=1)
    
    assert fsm.get_state(seq_id=0) == RoutingState.NAVIGATING
    assert fsm.get_state(seq_id=1) == RoutingState.NAVIGATING
    
    vocab_size = 1000
    logits_0 = np.arange(vocab_size, dtype=np.float32)
    logits_1 = np.arange(vocab_size, dtype=np.float32)
    
    # Interleaved logit masking and advancing
    fsm.apply_logit_mask(logits_0, seq_id=0)
    fsm.apply_logit_mask(logits_1, seq_id=1)
    
    # Verify Sequence 0 allows both starting tokens (100 and 400) but masks others
    assert logits_0[100] == 100.0
    assert logits_0[400] == 400.0
    assert logits_0[200] == -np.inf
    
    # Verify Sequence 1 allows both starting tokens (100 and 400) but masks others
    assert logits_1[100] == 100.0
    assert logits_1[400] == 400.0
    assert logits_1[500] == -np.inf
    
    # Advance both along different routes
    assert fsm.advance(100, seq_id=0) == RoutingState.NAVIGATING
    assert fsm.advance(400, seq_id=1) == RoutingState.NAVIGATING
    
    # Interleaved Step 2 (Divergence should occur here)
    logits_0 = np.arange(vocab_size, dtype=np.float32)
    logits_1 = np.arange(vocab_size, dtype=np.float32)
    
    fsm.apply_logit_mask(logits_0, seq_id=0)
    fsm.apply_logit_mask(logits_1, seq_id=1)
    
    # Sequence 0 is at [100], so only child 200 is valid
    assert logits_0[200] == 200.0
    assert logits_0[500] == -np.inf
    
    # Sequence 1 is at [400], so only child 500 is valid
    assert logits_1[500] == 500.0
    assert logits_1[200] == -np.inf
    
    assert fsm.advance(200, seq_id=0) == RoutingState.NAVIGATING
    assert fsm.advance(500, seq_id=1) == RoutingState.NAVIGATING
    
    # Interleaved Step 3 (Leaf resolution)
    logits_0 = np.arange(vocab_size, dtype=np.float32)
    logits_1 = np.arange(vocab_size, dtype=np.float32)
    
    fsm.apply_logit_mask(logits_0, seq_id=0)
    fsm.apply_logit_mask(logits_1, seq_id=1)
    
    # Sequence 0 is at [100, 200], so only child 300 is valid
    assert logits_0[300] == 300.0
    assert logits_0[600] == -np.inf
    
    # Sequence 1 is at [400, 500], so only child 600 is valid
    assert logits_1[600] == 600.0
    assert logits_1[300] == -np.inf
    
    assert fsm.advance(300, seq_id=0) == RoutingState.LEAF_REACHED
    assert fsm.advance(600, seq_id=1) == RoutingState.LEAF_REACHED
    
    assert fsm.get_resolved_tool_id(seq_id=0) == 1
    assert fsm.get_resolved_tool_id(seq_id=1) == 2
    
    print("test_concurrent_routing passed!")

if __name__ == "__main__":
    test_concurrent_routing()
