#pragma once
#include <cstddef>
#include <memory>
#include <string>
#include <thread>
#include <vector>

// Worst-case physical cache line size representing both x86_64 (64 bytes) and ARM64 (128 bytes)
inline constexpr size_t NEXUS_CACHE_LINE = 128;

/**
 * OS Memory Abstraction Layer (MAL) Interface
 */
class INexusMemoryManager {
public:
    virtual ~INexusMemoryManager() = default;
    
    // Maps a file descriptor/handle with a specific NUMA node affinity and Transparent Huge Pages (THP) hints.
    virtual void* map_file(int fd, size_t size, int numa_node) = 0;
    
    // Releases mapped region.
    virtual void unmap_file(void* addr, size_t size) = 0;
    
    // Advises the OS to load pages asynchronously.
    virtual void prefetch(void* addr, size_t size) = 0;
    
    // Allocates virtual pages pinned to a local NUMA node.
    virtual void* allocate_numa(size_t size, int numa_node) = 0;
    
    // Frees NUMA allocated virtual pages.
    virtual void free_numa(void* addr, size_t size) = 0;

    // Pre-allocates a static, continuous chunk of memory explicitly mapped with Huge Pages.
    virtual void* allocate_huge_arena(size_t size, int numa_node) = 0;

    // Frees Huge Page allocated virtual arena.
    virtual void free_huge_arena(void* addr, size_t size) = 0;
};

// Retrieve singleton/shared instance of the OS memory manager
std::shared_ptr<INexusMemoryManager> get_memory_manager();

// Dynamically discover CPU NUMA node mapped to GPU physical complex.
int discover_gpu_numa_node();

// Retrieve total number of NUMA nodes present in the topology.
int get_numa_node_count();

// Retrieve CPU core list mapping for a designated NUMA node (intersecting cgroup affinity on Linux).
std::vector<int> get_numa_node_cores(int numa_node);

// Pin target thread to specific CPU core.
void pin_thread_to_core(std::thread::native_handle_type handle, int core_id);

// High-performance Direct I/O reader bypassing OS page cache completely.
bool read_file_direct(const std::string& filepath, void* dest_addr, size_t size);

// Legacy wrappers for backward compatibility
void* map_file(int fd, size_t size);
void unmap_file(void* addr, size_t size);
void prefetch_memory(void* addr, size_t size);
size_t get_page_size();
void apply_numa_hint(void* addr, size_t size);
