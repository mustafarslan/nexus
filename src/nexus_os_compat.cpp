#include "nexus_os_compat.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <sstream>

#include <sys/mman.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <pthread.h>
#include <fcntl.h>
#include <sys/stat.h>

#if defined(__APPLE__)
#include <mach/thread_policy.h>
#include <mach/thread_act.h>
#include <pthread/qos.h>
#endif

#if defined(__linux__)
#ifndef SYS_mbind
#define SYS_mbind 237
#endif
#endif

class PosixMemoryManager : public INexusMemoryManager {
public:
    void* map_file(int fd, size_t size, int numa_node) override {
        if (fd < 0 || size == 0) return nullptr;
        
        int flags = MAP_SHARED;
        #if defined(__linux__)
        flags |= MAP_POPULATE;
        #endif
        
        void* addr = mmap(nullptr, size, PROT_READ, flags, fd, 0);
        if (addr == MAP_FAILED) return nullptr;
        
        #if defined(__linux__) && defined(MADV_HUGEPAGE)
        madvise(addr, size, MADV_HUGEPAGE);
        #endif
        
        #if defined(__linux__) && defined(SYS_mbind)
        if (numa_node >= 0) {
            long page_size = sysconf(_SC_PAGESIZE);
            uintptr_t start = reinterpret_cast<uintptr_t>(addr);
            uintptr_t end = start + size;
            uintptr_t aligned_start = start & ~(page_size - 1);
            uintptr_t aligned_end = (end + page_size - 1) & ~(page_size - 1);
            void* aligned_addr = reinterpret_cast<void*>(aligned_start);
            unsigned long aligned_len = aligned_end - aligned_start;
            
            unsigned long nodemask = (1UL << numa_node);
            syscall(SYS_mbind, aligned_addr, aligned_len, 1 /* MPOL_BIND */, &nodemask, sizeof(nodemask) * 8, 0);
        }
        #endif

        return addr;
    }
    
    void unmap_file(void* addr, size_t size) override {
        if (!addr) return;
        munmap(addr, size);
    }
    
    void prefetch(void* addr, size_t size) override {
        if (!addr || size == 0) return;
        #if defined(__linux__)
            #if defined(MADV_POPULATE_READ)
                madvise(addr, size, MADV_POPULATE_READ);
            #else
                posix_madvise(addr, size, POSIX_MADV_WILLNEED);
            #endif
        #elif defined(__APPLE__)
            posix_madvise(addr, size, POSIX_MADV_WILLNEED);
        #endif
    }
    
    void* allocate_numa(size_t size, int numa_node) override {
        if (size == 0) return nullptr;
        long page_size = sysconf(_SC_PAGESIZE);
        size_t aligned_size = ((size + page_size - 1) / page_size) * page_size;
        
        void* addr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (addr == MAP_FAILED) return nullptr;
        
        #if defined(__linux__) && defined(SYS_mbind)
        if (numa_node >= 0) {
            unsigned long nodemask = (1UL << numa_node);
            syscall(SYS_mbind, addr, aligned_size, 1 /* MPOL_BIND */, &nodemask, sizeof(nodemask) * 8, 0);
        }
        #endif
        return addr;
    }
    
    void free_numa(void* addr, size_t size) override {
        if (!addr) return;
        long page_size = sysconf(_SC_PAGESIZE);
        size_t aligned_size = ((size + page_size - 1) / page_size) * page_size;
        munmap(addr, aligned_size);
    }

    void* allocate_huge_arena(size_t size, int numa_node) override {
        if (size == 0) return nullptr;
        size_t page_size = 2097152; // 2MB
        size_t aligned_size = ((size + page_size - 1) / page_size) * page_size;
        
        void* addr = MAP_FAILED;
#if defined(__linux__) && defined(MAP_HUGETLB)
        addr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (addr != MAP_FAILED) {
#if defined(SYS_mbind)
            if (numa_node >= 0) {
                unsigned long nodemask = (1UL << numa_node);
                syscall(SYS_mbind, addr, aligned_size, 1 /* MPOL_BIND */, &nodemask, sizeof(nodemask) * 8, 0);
            }
#endif
            return addr;
        }
#endif
        size_t alloc_size = aligned_size + page_size;
        addr = mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (addr == MAP_FAILED) return nullptr;
        
        uintptr_t raw_addr = reinterpret_cast<uintptr_t>(addr);
        uintptr_t aligned_addr = ((raw_addr + page_size - 1) / page_size) * page_size;
        void* final_addr = reinterpret_cast<void*>(aligned_addr);
        
        if (aligned_addr > raw_addr) {
            munmap(addr, aligned_addr - raw_addr);
        }
        size_t prefix_len = aligned_addr - raw_addr;
        size_t suffix_len = page_size - prefix_len;
        if (suffix_len > 0) {
            munmap(reinterpret_cast<void*>(aligned_addr + aligned_size), suffix_len);
        }
        
#if defined(__linux__) && defined(SYS_mbind)
        if (numa_node >= 0) {
            unsigned long nodemask = (1UL << numa_node);
            syscall(SYS_mbind, final_addr, aligned_size, 1 /* MPOL_BIND */, &nodemask, sizeof(nodemask) * 8, 0);
        }
#endif
        return final_addr;
    }

    void free_huge_arena(void* addr, size_t size) override {
        if (!addr) return;
        size_t aligned_size = ((size + 2097151) / 2097152) * 2097152;
        munmap(addr, aligned_size);
    }
};

std::shared_ptr<INexusMemoryManager> get_memory_manager() {
    static std::shared_ptr<INexusMemoryManager> manager = []() -> std::shared_ptr<INexusMemoryManager> {
        return std::make_shared<PosixMemoryManager>();
    }();
    return manager;
}

int discover_gpu_numa_node() {
#if defined(__linux__)
    for (int i = 0; i < 8; ++i) {
        std::string path = "/sys/class/drm/card" + std::to_string(i) + "/device/numa_node";
        std::ifstream f(path);
        if (f.is_open()) {
            int node = -1;
            if (f >> node && node >= 0) {
                return node;
            }
        }
    }
    try {
        if (std::filesystem::exists("/sys/bus/pci/devices")) {
            for (const auto& entry : std::filesystem::directory_iterator("/sys/bus/pci/devices")) {
                std::ifstream class_file(entry.path() / "class");
                uint32_t device_class = 0;
                if (class_file >> std::hex >> device_class) {
                    if ((device_class & 0xFFFF00) == 0x030000 || (device_class & 0xFFFF00) == 0x030200) {
                        std::ifstream numa_file(entry.path() / "numa_node");
                        int node = -1;
                        if (numa_file >> node && node >= 0) {
                            return node;
                        }
                    }
                }
            }
        }
    } catch (...) {}
    return 0;
#else
    return 0;
#endif
}

int get_numa_node_count() {
    int count = 1;
#if defined(__linux__)
    try {
        if (std::filesystem::exists("/sys/devices/system/node")) {
            int max_node = 0;
            for (const auto& entry : std::filesystem::directory_iterator("/sys/devices/system/node")) {
                std::string name = entry.path().filename().string();
                if (name.rfind("node", 0) == 0) {
                    try {
                        int num = std::stoi(name.substr(4));
                        if (num > max_node) {
                            max_node = num;
                        }
                    } catch (...) {}
                }
            }
            count = max_node + 1;
        }
    } catch (...) {}
#endif
    return count > 0 ? count : 1;
}

std::vector<int> get_numa_node_cores(int numa_node) {
    std::vector<int> physical_cores;
#if defined(__linux__)
    std::string path = "/sys/devices/system/node/node" + std::to_string(numa_node) + "/cpulist";
    std::ifstream f(path);
    if (f.is_open()) {
        std::string line;
        if (std::getline(f, line)) {
            std::stringstream ss(line);
            std::string part;
            while (std::getline(ss, part, ',')) {
                size_t dash = part.find('-');
                if (dash == std::string::npos) {
                    physical_cores.push_back(std::stoi(part));
                } else {
                    int start = std::stoi(part.substr(0, dash));
                    int end = std::stoi(part.substr(dash + 1));
                    for (int c = start; c <= end; ++c) {
                        physical_cores.push_back(c);
                    }
                }
            }
        }
    }
#endif

    std::vector<int> allowed_cores;
#if defined(__linux__)
    cpu_set_t allowed_cpus;
    CPU_ZERO(&allowed_cpus);
    if (sched_getaffinity(0, sizeof(cpu_set_t), &allowed_cpus) == 0) {
        for (int i = 0; i < CPU_SETSIZE; ++i) {
            if (CPU_ISSET(i, &allowed_cpus)) {
                allowed_cores.push_back(i);
            }
        }
    }
#endif

    std::vector<int> intersected_cores;
    if (!allowed_cores.empty()) {
        for (int pc : physical_cores) {
            if (std::find(allowed_cores.begin(), allowed_cores.end(), pc) != allowed_cores.end()) {
                intersected_cores.push_back(pc);
            }
        }
        if (intersected_cores.empty()) {
            intersected_cores = allowed_cores;
        }
    } else {
        intersected_cores = physical_cores;
    }

    if (intersected_cores.empty()) {
        int n = std::thread::hardware_concurrency();
        if (n <= 0) n = 4;
        for (int i = 0; i < n; ++i) {
            intersected_cores.push_back(i);
        }
    }
    return intersected_cores;
}

void pin_thread_to_core(std::thread::native_handle_type handle, int core_id) {
#if defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(handle, sizeof(cpu_set_t), &cpuset);
#elif defined(__APPLE__)
    thread_affinity_policy_data_t policy = { core_id };
    thread_port_t mach_thread = pthread_mach_thread_np(handle);
    thread_policy_set(mach_thread, THREAD_AFFINITY_POLICY, (thread_policy_t)&policy, THREAD_AFFINITY_POLICY_COUNT);
    // Enforce high QoS class on Apple Silicon to prevent demotion to E-cores
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#else
    (void)handle;
    (void)core_id;
#endif
}

bool read_file_direct(const std::string& filepath, void* dest_addr, size_t size) {
    int flags = O_RDONLY;
#if defined(__linux__)
    flags |= O_DIRECT;
#endif
    int fd = open(filepath.c_str(), flags);
#if defined(__linux__)
    if (fd < 0 && (flags & O_DIRECT)) {
        fd = open(filepath.c_str(), O_RDONLY);
    }
#endif
    if (fd < 0) return false;
    
    size_t total_read = 0;
    char* ptr = static_cast<char*>(dest_addr);
    bool success = true;
    while (total_read < size) {
        ssize_t bytes = pread(fd, ptr + total_read, size - total_read, total_read);
        if (bytes < 0) {
            if (errno == EINTR) continue;
            success = false;
            break;
        }
        if (bytes == 0) break; // EOF
        total_read += bytes;
    }
    close(fd);
    return success && (total_read == size);
}

// Legacy wrappers
void* map_file(int fd, size_t size) {
    return get_memory_manager()->map_file(fd, size, discover_gpu_numa_node());
}

void unmap_file(void* addr, size_t size) {
    get_memory_manager()->unmap_file(addr, size);
}

void prefetch_memory(void* addr, size_t size) {
    get_memory_manager()->prefetch(addr, size);
}

size_t get_page_size() {
    long page_size = sysconf(_SC_PAGESIZE);
    return page_size > 0 ? static_cast<size_t>(page_size) : 4096;
}

void apply_numa_hint(void* addr, size_t size) {
#if defined(__linux__) && defined(SYS_mbind)
    if (!addr || size == 0) return;
    int node = discover_gpu_numa_node();
    if (node >= 0) {
        long page_size = sysconf(_SC_PAGESIZE);
        uintptr_t start = reinterpret_cast<uintptr_t>(addr);
        uintptr_t end = start + size;
        uintptr_t aligned_start = start & ~(page_size - 1);
        uintptr_t aligned_end = (end + page_size - 1) & ~(page_size - 1);
        void* aligned_addr = reinterpret_cast<void*>(aligned_start);
        unsigned long aligned_len = aligned_end - aligned_start;
        
        unsigned long nodemask = (1UL << node);
        syscall(SYS_mbind, aligned_addr, aligned_len, 1 /* MPOL_BIND */, &nodemask, sizeof(nodemask) * 8, 0);
    }
#else
    (void)addr;
    (void)size;
#endif
}
