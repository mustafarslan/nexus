#pragma once
#include <string>
#include <span>
#include <cstdint>
#include <cstddef>
#include <unordered_map>
#include <unordered_set>
#include <shared_mutex>
#include <future>
#include <atomic>
#include <memory>
#include <array>
#include <vector>
#include <expected>
#include <functional>
#include <random>
#include <new>
#include <string_view>
#include "aeon_tool_block.hpp"
#include "nexus_os_compat.hpp"
#include "nexus_bitmap_allocator.hpp"
#include "llama.h"

// FNV-1a 64-bit fast hashing helper for filepaths
inline uint64_t get_filepath_hash(std::string_view filepath) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    for (char c : filepath) {
        hash ^= static_cast<uint8_t>(c);
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

// C++20 Heterogeneous lookup hash with nested is_transparent type
struct StringViewHash {
    using is_transparent = void;
    [[nodiscard]] size_t operator()(std::string_view txt) const {
        return std::hash<std::string_view>{}(txt);
    }
    [[nodiscard]] size_t operator()(const std::string& txt) const {
        return std::hash<std::string>{}(txt);
    }
    [[nodiscard]] size_t operator()(const char* txt) const {
        return std::hash<std::string_view>{}(txt);
    }
};



enum class NexusErrorCode : uint8_t;

// Phase 58: Aligned shared control block for telemetry lifetimes
struct alignas(NEXUS_CACHE_LINE) NexusTelemetry {
    alignas(NEXUS_CACHE_LINE) std::atomic<uint64_t> load_errors{0};
    alignas(NEXUS_CACHE_LINE) std::atomic<uint64_t> submit_rejections{0};
    alignas(NEXUS_CACHE_LINE) std::atomic<uint64_t> blocks_freed{0};
};

class AeonToolBlock {
private:
    void* mapped_data_ = nullptr;
    size_t file_size_ = 0;
    AeonToolBlockHeader header_{};
    bool has_header_ = false;
    std::shared_future<void> residency_future_;
    uint64_t generation_id_ = 0;
    std::function<void(void*)> free_callback_;
    int numa_node_ = 0;
    std::shared_ptr<NexusTelemetry> telemetry_;

public:
    explicit AeonToolBlock(const std::string& atb_filepath, void* mapped_data, size_t file_size, 
                           std::function<void(void*)> free_callback, std::function<void()> on_failure = nullptr, uint64_t generation_id = 0, int numa_node = 0, std::shared_ptr<NexusTelemetry> telemetry = nullptr);
    ~AeonToolBlock();

    AeonToolBlock(const AeonToolBlock&) = delete;
    AeonToolBlock& operator=(const AeonToolBlock&) = delete;

    const AeonToolBlockHeader* get_header() const { return has_header_ ? &header_ : nullptr; }
    void* get_mapped_data() const { return mapped_data_; }
    size_t get_file_size() const { return file_size_; }
    uint64_t get_generation_id() const { return generation_id_; }
    int get_numa_node() const { return numa_node_; }

    void wait_until_resident() const;

    std::span<const uint8_t> get_k_tensors() const;
    std::span<const uint8_t> get_v_tensors() const;

    void set_header(const AeonToolBlockHeader& header) {
        header_ = header;
        has_header_ = true;
    }
};

namespace nexus {
    int get_thread_slot_index();
}

struct alignas(NEXUS_CACHE_LINE) CacheEntry {
    // === HOT zone ===
    std::atomic<bool> referenced{false}; // CLOCK style access flag
    std::atomic<bool> evicted{false}; // Marked for eviction/removal

    // === COLD zone (eviction-mutex-protected or immutable after init) ===
    std::atomic<uint64_t> last_access_tick{0};
    size_t byte_size{0};
    uint64_t generation_id{0};
    int numa_node{0}; // Phase 34: NUMA node this block was allocated on
    bool is_protected{false}; // Scan-Resistant cache segment (false = Probation, true = Protected)

    // Intrusive doubly-linked list pointers (eviction-mutex-protected)
    CacheEntry* prev_lru = nullptr;
    CacheEntry* next_lru = nullptr;

    std::shared_future<std::expected<std::shared_ptr<AeonToolBlock>, NexusErrorCode>> future;
    std::string filepath; // Retain filepath for cleanup & reference
};

struct alignas(128) HazardSlot {
    std::atomic<CacheEntry*> entry{nullptr};
};

struct HazardGuard {
    CacheEntry* entry = nullptr;

    HazardGuard() = default;
    explicit HazardGuard(CacheEntry* e);
    ~HazardGuard();

    void release();

    // Disable copy
    HazardGuard(const HazardGuard&) = delete;
    HazardGuard& operator=(const HazardGuard&) = delete;

    // Enable move
    HazardGuard(HazardGuard&& other) noexcept;
    HazardGuard& operator=(HazardGuard&& other) noexcept;
};

enum class NexusErrorCode : uint8_t {
    OK = 0,
    RESOURCE_EXHAUSTED = 1,
    FILE_NOT_FOUND = 2,
    INVALID_HEADER = 3,
    INTERNAL_ERROR = 4,
};

class ResourceExhaustedException : public std::runtime_error {
public:
    explicit ResourceExhaustedException(const std::string& msg)
        : std::runtime_error(msg) {}
};

// Phase 54: Lock-free telemetry counters for hot-path observability (no fprintf)
struct NexusCacheTelemetry {
    uint64_t load_errors;
    uint64_t submit_rejections;
    uint64_t blocks_freed;
};

struct alignas(NEXUS_CACHE_LINE) CacheShard {
    std::shared_mutex mutex;
    std::unordered_map<std::string, CacheEntry*, StringViewHash, std::equal_to<>> cache_map;
};

class NexusBlockCache {
private:
    size_t max_pinned_bytes_;
    std::atomic<size_t> current_pinned_bytes_{0};
    std::atomic<uint64_t> global_tick_{0};
    std::atomic<uint64_t> global_generation_{0};
    
    // Allocate shards on heap to avoid class-level over-alignment restrictions in bindings (nanobind)
    std::unique_ptr<std::array<CacheShard, 64>> shards_;

    // HugeTLB Memory Arena members per NUMA node
    std::vector<void*> arena_bases_;
    std::vector<size_t> arena_sizes_;
    std::vector<std::unique_ptr<nexus::QuantizedBitmapAllocator>> allocators_;

    // Per-page generation and entry tracking for O(1) checking
    std::vector<std::vector<std::atomic<uint64_t>>> page_generations_;
    std::vector<std::vector<std::atomic<CacheEntry*>>> page_entries_;

    // Map tool_id to path for context manager lookups
    std::unordered_map<uint32_t, std::string> tool_id_to_path_;
    mutable std::shared_mutex tool_id_map_mutex_;
    std::unordered_set<uint32_t> pinned_tool_ids_;
    mutable std::shared_mutex pinned_mutex_;

    // Phase 34: Per-NUMA 2Q Intrusive Eviction Queues
    // Each NUMA node gets its own probation/protected lists and mutex,
    // preventing cross-socket UPI traffic during eviction and parallelizing
    // cache misses across NUMA nodes.
    struct alignas(NEXUS_CACHE_LINE) NUMAEvictionQueue {
        std::mutex mtx;
        CacheEntry* probation_head = nullptr;
        CacheEntry* probation_tail = nullptr;
        CacheEntry* protected_head = nullptr;
        CacheEntry* protected_tail = nullptr;
    };
    std::unique_ptr<NUMAEvictionQueue[]> eviction_queues_;
    size_t num_numa_nodes_ = 0;

    std::shared_ptr<NexusTelemetry> telemetry_;

    static void list_push_front(CacheEntry*& head, CacheEntry*& tail, CacheEntry* entry) {
        entry->next_lru = head;
        entry->prev_lru = nullptr;
        if (head) {
            head->prev_lru = entry;
        } else {
            tail = entry;
        }
        head = entry;
    }

    static void list_remove(CacheEntry*& head, CacheEntry*& tail, CacheEntry* entry) {
        if (entry->prev_lru) {
            entry->prev_lru->next_lru = entry->next_lru;
        } else {
            head = entry->next_lru;
        }
        if (entry->next_lru) {
            entry->next_lru->prev_lru = entry->prev_lru;
        } else {
            tail = entry->prev_lru;
        }
        entry->prev_lru = nullptr;
        entry->next_lru = nullptr;
    }

    std::mutex retired_mutex_;
    std::vector<CacheEntry*> retired_entries_;

    void retire_entry(CacheEntry* entry);
    void reclaim_retired_entries();

    void evict_to_limit(size_t required_bytes, int target_node);
    bool is_path_pinned(const std::string& path) const;
    void initialize_numa_arena(int numa_node);
    CacheEntry* get_entry_from_ptr(void* mapped_data) const;
    int get_node_for_ptr(const void* ptr) const;

public:
    static constexpr size_t MAX_THREADS = 128;
    static HazardSlot hazard_pointers[MAX_THREADS];

    explicit NexusBlockCache(size_t max_pinned_bytes);
    ~NexusBlockCache();

    static void release_entry(CacheEntry* entry);

    NexusBlockCache(const NexusBlockCache&) = delete;
    NexusBlockCache& operator=(const NexusBlockCache&) = delete;

    std::expected<std::shared_ptr<AeonToolBlock>, NexusErrorCode> get_or_load(std::string_view atb_filepath);
    void prefetch(std::string_view atb_filepath);

    // Dynamic failure handler callback interface for ABA mitigation
    void on_load_failure(std::string_view atb_filepath, uint64_t failed_gen);
    
    uint64_t get_current_generation(void* mapped_data) const;

    size_t get_current_pinned_bytes() const { return current_pinned_bytes_.load(std::memory_order_relaxed); }
    size_t get_max_pinned_bytes() const { return max_pinned_bytes_; }

    // Lock-free Read Hazard Management
    HazardGuard acquire_hazard(void* mapped_data);
    void release_hazard(void* mapped_data);

    void register_tool_path(uint32_t tool_id, const std::string& path);
    void pin_tool(uint32_t tool_id);
    void unpin_tool(uint32_t tool_id);
    bool is_tool_pinned(uint32_t tool_id) const;
    HazardGuard hazard_guard(uint32_t tool_id);

    NexusCacheTelemetry get_cache_telemetry() const;
};

// static_assert(offsetof(CacheEntry, active_readers) == 0, "Hot atomics must be at offset 0");

