import sys
import os
import numpy as np

# Add the build directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))

import nexus_fsm_ext
from nexus_fsm_ext import NexusRadixFSM, RoutingState

def test_fsm_basic():
    fsm = NexusRadixFSM()
    assert fsm.state == RoutingState.IDLE
    
    # Define tool routes
    # Tool 1: [100, 200, 300]
    # Tool 2: [100, 200, 400]
    # Tool 3: [500, 600]
    fsm.add_route(1, [100, 200, 300])
    fsm.add_route(2, [100, 200, 400])
    fsm.add_route(3, [500, 600])
    
    # Initially IDLE
    fsm.reset()
    assert fsm.state == RoutingState.IDLE
    
    # Start routing
    fsm.begin_routing()
    assert fsm.state == RoutingState.NAVIGATING
    
    # Initial logits array (vocab size = 1000)
    vocab_size = 1000
    logits = np.arange(vocab_size, dtype=np.float32)
    
    # At root, valid first tokens are 100 and 500
    fsm.apply_logit_mask(logits)
    
    # Verify that only indices 100 and 500 are preserved, and everything else is -inf
    assert logits[100] == 100.0
    assert logits[500] == 500.0
    assert logits[0] == -np.inf
    assert logits[999] == -np.inf
    
    # Advance FSM with token 100
    state = fsm.advance(100)
    assert state == RoutingState.NAVIGATING
    assert fsm.state == RoutingState.NAVIGATING
    
    # At node [100], valid next token is 200
    logits = np.arange(vocab_size, dtype=np.float32)
    fsm.apply_logit_mask(logits)
    assert logits[200] == 200.0
    assert logits[100] == -np.inf
    assert logits[500] == -np.inf
    
    # Advance with token 200
    state = fsm.advance(200)
    assert state == RoutingState.NAVIGATING
    
    # At node [100, 200], valid next tokens are 300 and 400
    logits = np.arange(vocab_size, dtype=np.float32)
    fsm.apply_logit_mask(logits)
    assert logits[300] == 300.0
    assert logits[400] == 400.0
    assert logits[200] == -np.inf
    
    # Advance with token 300 (resolves tool 1)
    state = fsm.advance(300)
    assert state == RoutingState.LEAF_REACHED
    assert fsm.resolved_tool_id == 1
    
    print("test_fsm_basic passed!")

def test_fsm_invalid_advance():
    fsm = NexusRadixFSM()
    fsm.add_route(1, [100, 200])
    
    fsm.begin_routing()
    # Advance with an invalid token
    state = fsm.advance(999)
    assert state == RoutingState.INVALID
    print("test_fsm_invalid_advance passed!")

if __name__ == "__main__":
    test_fsm_basic()
    test_fsm_invalid_advance()
    print("All tests passed successfully!")
