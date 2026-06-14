import sys
import os
import ctypes

# Add build directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))

import nexus_fsm_ext
from nexus_fsm_ext import QuantizedBitmapAllocator

def test_buddy_allocator():
    print("Running Buddy Allocator tests...")
    
    # 2MB alignment size
    PAGE_SIZE = 2 * 1024 * 1024
    
    # Allocate a 16MB buffer (8 blocks of 2MB)
    num_pages = 8
    arena_size = num_pages * PAGE_SIZE
    
    # Allocate a buffer using ctypes to act as our simulated HugeTLB memory arena
    buffer = ctypes.create_string_buffer(arena_size)
    base_addr = ctypes.addressof(buffer)
    
    allocator = QuantizedBitmapAllocator(base_addr, arena_size)
    
    # Allocation 1: 2MB (Block 0)
    ptr1 = allocator.allocate(2 * 1024 * 1024)
    assert ptr1 == base_addr, f"Expected {base_addr}, got {ptr1}"
    
    # Allocation 2: 2MB (Block 1) - Buddy of Block 0
    ptr2 = allocator.allocate(2 * 1024 * 1024)
    assert ptr2 == base_addr + PAGE_SIZE, f"Expected {base_addr + PAGE_SIZE}, got {ptr2}"
    
    # Allocation 3: 4MB (Blocks 2-3)
    ptr3 = allocator.allocate(4 * 1024 * 1024)
    assert ptr3 == base_addr + 2 * PAGE_SIZE, f"Expected {base_addr + 2 * PAGE_SIZE}, got {ptr3}"
    
    # Allocation 4: 8MB (Blocks 4-7)
    ptr4 = allocator.allocate(8 * 1024 * 1024)
    assert ptr4 == base_addr + 4 * PAGE_SIZE, f"Expected {base_addr + 4 * PAGE_SIZE}, got {ptr4}"
    
    # Arena is now fully allocated. Further allocations should return 0.
    ptr_fail = allocator.allocate(2 * 1024 * 1024)
    assert ptr_fail == 0, f"Expected 0, got {ptr_fail}"
    
    # Free allocation 1 (Block 0) and 2 (Block 1). They should coalesce to 4MB.
    allocator.free(ptr1, 2 * 1024 * 1024)
    allocator.free(ptr2, 2 * 1024 * 1024)
    
    # Free allocation 3 (Blocks 2-3). Blocks 2-3 coalesce with 0-1, forming a contiguous 8MB block.
    allocator.free(ptr3, 4 * 1024 * 1024)
    
    # Allocate 8MB. It should now successfully allocate at the beginning of the arena.
    ptr5 = allocator.allocate(8 * 1024 * 1024)
    assert ptr5 == base_addr, f"Expected {base_addr}, got {ptr5}"
    
    # Free ptr5 and ptr4 to return everything back to the arena.
    allocator.free(ptr5, 8 * 1024 * 1024)
    allocator.free(ptr4, 8 * 1024 * 1024)
    
    # Allocate 16MB. Coalescing should have reconstituted the full 16MB block.
    ptr6 = allocator.allocate(16 * 1024 * 1024)
    assert ptr6 == base_addr, f"Expected {base_addr}, got {ptr6}"
    
    print("All Buddy Allocator assertions passed!")

if __name__ == "__main__":
    test_buddy_allocator()
