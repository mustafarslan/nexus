#include "nexus_block_cache.hpp"
#include "nexus_os_compat.hpp"
#include "nexus_thread_pool.hpp"
#include "nexus_io_async.hpp"
#include <stdexcept>
#include <cstring>
#include <limits>
#include <chrono>
#include <filesystem>
#include <latch>

alignas(128) HazardSlot NexusBlockCache::hazard_pointers[NexusBlockCache::MAX_THREADS];

static bool is_entry_protected(CacheEntry* entry) {
    if (!entry) return false;
    for (size_t i = 0; i < NexusBlockCache::MAX_THREADS; ++i) {
        if (NexusBlockCache::hazard_pointers[i].entry.load(std::memory_order_acquire) == entry) {
            return true;
        }
    }
    return false;
}

// Phase 54/56: Lock-free atomic telemetry counters as NexusBlockCache members
NexusCacheTelemetry NexusBlockCache::get_cache_telemetry() const {
    if (!telemetry_) return {0, 0, 0};
    return {
        .load_errors = telemetry_->load_errors.load(std::memory_order_relaxed),
        .submit_rejections = telemetry_->submit_rejections.load(std::memory_order_relaxed),
        .blocks_freed = telemetry_->blocks_freed.load(std::memory_order_relaxed),
    };
}

// Phase 58: Aligned shared control block context for telemetry lifetimes
struct LoadContext {
    NexusBlockCache* cache;
    std::string filepath;
    void* mapped_data;
    size_t size;
    uint64_t gen;
    int target_node;
    std::shared_ptr<std::promise<std::expected<std::shared_ptr<AeonToolBlock>, NexusErrorCode>>> promise;
    std::atomic<bool> promise_fulfilled{false};
    std::shared_ptr<NexusTelemetry> telemetry;

    ~LoadContext() noexcept {
        if (!promise_fulfilled.load(std::memory_order_acquire) && promise) {
            try {
                promise->set_value(std::unexpected(NexusErrorCode::RESOURCE_EXHAUSTED));
            } catch (...) {
                // Swallow exception to prevent std::terminate
            }
        }
    }

    // Non-copyable, non-movable
    LoadContext(const LoadContext&) = delete;
    LoadContext& operator=(const LoadContext&) = delete;
    LoadContext(LoadContext&&) = delete;
    LoadContext& operator=(LoadContext&&) = delete;

    LoadContext(NexusBlockCache* cache_, std::string filepath_, void* mapped_data_,
                size_t size_, uint64_t gen_, int target_node_,
                std::shared_ptr<std::promise<std::expected<std::shared_ptr<AeonToolBlock>, NexusErrorCode>>> promise_,
                std::shared_ptr<NexusTelemetry> telemetry_)
        : cache(cache_), filepath(std::move(filepath_)), mapped_data(mapped_data_),
          size(size_), gen(gen_), target_node(target_node_), promise(std::move(promise_)),
          telemetry(std::move(telemetry_)) {}
};

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/mman.h>

extern "C" {
    void llama_unregister_host_memory(void * buffer);
    bool llama_register_host_memory(void * buffer, size_t size);
}

AeonToolBlock::AeonToolBlock(const std::string& atb_filepath, void* mapped_data, size_t file_size, 
                             std::function<void(void*)> free_callback, std::function<void()> on_failure, uint64_t generation_id, int numa_node, std::shared_ptr<NexusTelemetry> telemetry)
    : mapped_data_(mapped_data), file_size_(file_size), free_callback_(free_callback), generation_id_(generation_id), numa_node_(numa_node), telemetry_(std::move(telemetry)) {
    if (!mapped_data_) {
        throw std::runtime_error("AeonToolBlock: mapped_data is null");
    }
    if (file_size_ < sizeof(AeonToolBlockHeader)) {
        throw std::runtime_error("AeonToolBlock: file_size is smaller than header size");
    }

    std::promise<void> promise;
    residency_future_ = promise.get_future().share();

    nexus::submit_async_read_direct(atb_filepath, mapped_data_, file_size_, std::move(promise), this, on_failure, numa_node_);
}

AeonToolBlock::~AeonToolBlock() {
    // Phase 58: Shared telemetry block to prevent UAF
    if (telemetry_) {
        telemetry_->blocks_freed.fetch_add(1, std::memory_order_relaxed);
    }
    if (residency_future_.valid()) {
        try {
            residency_future_.wait();
        } catch (...) {}
    }
    if (mapped_data_) {
        if (free_callback_) {
            free_callback_(mapped_data_);
        }
    }
}

void AeonToolBlock::wait_until_resident() const {
    if (residency_future_.valid()) {
        residency_future_.get();
    }
}

std::span<const uint8_t> AeonToolBlock::get_k_tensors() const {
    wait_until_resident();
    if (!mapped_data_ || !has_header_) return {};
    const uint8_t* ptr = static_cast<const uint8_t*>(mapped_data_) + header_.k_tensor_offset;
    return { ptr, static_cast<size_t>(header_.k_total_bytes) };
}

std::span<const uint8_t> AeonToolBlock::get_v_tensors() const {
    wait_until_resident();
    if (!mapped_data_ || !has_header_) return {};
    const uint8_t* ptr = static_cast<const uint8_t*>(mapped_data_) + header_.v_tensor_offset;
    return { ptr, static_cast<size_t>(header_.v_total_bytes) };
}

int NexusBlockCache::get_node_for_ptr(const void* ptr) const {
    for (size_t i = 0; i < arena_bases_.size(); ++i) {
        if (ptr >= arena_bases_[i] && static_cast<const char*>(ptr) < static_cast<const char*>(arena_bases_[i]) + arena_sizes_[i]) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

struct NumaInitBarrier {
    std::mutex mtx;
    std::condition_variable cv;
    int active_tasks = 0;
};

struct NumaInitTaskCtx {
    NexusBlockCache* cache;
    int numa_node;
    std::shared_ptr<NumaInitBarrier> barrier;
    std::atomic<bool> success{true};
    std::exception_ptr exc_ptr;
};

NexusBlockCache::NexusBlockCache(size_t max_pinned_bytes) 
    : telemetry_(std::make_shared<NexusTelemetry>()),
      shards_(std::make_unique<std::array<CacheShard, 64>>()) {
    if (max_pinned_bytes == 0) {
        max_pinned_bytes = 16ULL * 1024 * 1024 * 1024; // Default to 16GB limit
    }
    max_pinned_bytes_ = max_pinned_bytes;
    
    // Ensure we physically allocate at least 256MB to fit test blocks while respecting logical limits
    size_t min_arena = 256ULL * 1024 * 1024;
    size_t arena_size = std::max<size_t>(max_pinned_bytes_, min_arena);
    // Over-provision the physical arena by 25% to absorb page alignment fragmentation
    arena_size = arena_size + (arena_size * 25 / 100);
    arena_size = ((arena_size + 2097151) / 2097152) * 2097152;

    int node_count = get_numa_node_count();
    if (node_count <= 0) node_count = 1;

    eviction_queues_ = std::make_unique<NUMAEvictionQueue[]>(node_count);
    arena_bases_.reserve(node_count);
    arena_sizes_.reserve(node_count);
    allocators_.reserve(node_count);
    page_generations_.resize(node_count);
    page_entries_.resize(node_count);
    num_numa_nodes_ = node_count;

    for (int i = 0; i < node_count; ++i) {
        void* base = get_memory_manager()->allocate_huge_arena(arena_size, i);
        if (!base) {
            // Rollback already allocated arenas
            for (size_t j = 0; j < arena_bases_.size(); ++j) {
                llama_unregister_host_memory(arena_bases_[j]);
                get_memory_manager()->free_huge_arena(arena_bases_[j], arena_sizes_[j]);
            }
            throw std::runtime_error("NexusBlockCache: Failed to allocate HugeTLB arena of size " + 
                                     std::to_string(arena_size) + " on NUMA node " + std::to_string(i));
        }
        arena_bases_.push_back(base);
        arena_sizes_.push_back(arena_size);
        allocators_.push_back(std::make_unique<nexus::QuantizedBitmapAllocator>(base, arena_size));

        size_t num_pages = arena_size / NEXUS_PAGE_ALIGNMENT;
        page_generations_[i] = std::vector<std::atomic<uint64_t>>(num_pages);
        page_entries_[i] = std::vector<std::atomic<CacheEntry*>>(num_pages);
        for (size_t p = 0; p < num_pages; ++p) {
            page_generations_[i][p].store(0, std::memory_order_relaxed);
            page_entries_[i][p].store(nullptr, std::memory_order_relaxed);
        }
    }
    
    // Force initialization of NUMA thread pools to start background worker polling
    (void)get_numa_pool_manager();

    auto barrier = std::make_shared<NumaInitBarrier>();
    barrier->active_tasks = node_count;

    std::vector<std::shared_ptr<NumaInitTaskCtx>> tasks;
    tasks.reserve(node_count);

    for (int i = 0; i < node_count; ++i) {
        auto task = std::make_shared<NumaInitTaskCtx>();
        task->cache = this;
        task->numa_node = i;
        task->barrier = barrier;
        tasks.push_back(task);

        if (!get_numa_pool_manager().get_pool(i).submit([task]() {
            try {
                task->cache->initialize_numa_arena(task->numa_node);
            } catch (...) {
                task->success.store(false, std::memory_order_release);
                task->exc_ptr = std::current_exception();
            }

            std::shared_ptr<NumaInitBarrier> barr = task->barrier;
            {
                std::lock_guard<std::mutex> lock(barr->mtx);
                barr->active_tasks--;
                if (barr->active_tasks == 0) {
                    barr->cv.notify_one();
                }
            }
        })) {
            task->success.store(false, std::memory_order_release);
            std::lock_guard<std::mutex> lock(barrier->mtx);
            barrier->active_tasks--;
            if (barrier->active_tasks == 0) {
                barrier->cv.notify_one();
            }
        }
    }

    {
        std::unique_lock<std::mutex> lock(barrier->mtx);
        barrier->cv.wait(lock, [&]() { return barrier->active_tasks == 0; });
    }

    // Validate initialization success
    for (int i = 0; i < node_count; ++i) {
        if (!tasks[i]->success.load(std::memory_order_acquire)) {
            // Rollback already initialized and registered host memory
            for (size_t j = 0; j < arena_bases_.size(); ++j) {
                llama_unregister_host_memory(arena_bases_[j]);
                get_memory_manager()->free_huge_arena(arena_bases_[j], arena_sizes_[j]);
            }
            if (tasks[i]->exc_ptr) {
                std::rethrow_exception(tasks[i]->exc_ptr);
            } else {
                throw std::runtime_error("NexusBlockCache: NUMA Node " + std::to_string(i) + " initialization failed.");
            }
        }
    }
}

void NexusBlockCache::initialize_numa_arena(int numa_node) {
    if (numa_node < 0 || numa_node >= static_cast<int>(arena_bases_.size())) {
        return;
    }
    void* base = arena_bases_[numa_node];
    size_t size = arena_sizes_[numa_node];
    if (!base || size == 0) return;

    // Force First-Touch page faults to physical RAM on the local NUMA node
    char* char_base = static_cast<char*>(base);
    for (size_t offset = 0; offset < size; offset += 4096) {
        volatile char* p = reinterpret_cast<volatile char*>(char_base + offset);
        *p = 0;
    }

    // Pre-pin the host memory local to the thread's CPU socket
    if (!llama_register_host_memory(base, size)) {
        throw std::runtime_error("Failed to register host memory for HugeTLB arena on NUMA node " + std::to_string(numa_node));
    }
}

NexusBlockCache::~NexusBlockCache() {
    for (size_t i = 0; i < 64; ++i) {
        auto& shard = (*shards_)[i];
        for (auto& pair : shard.cache_map) {
            delete pair.second;
        }
    }
    // Clean up retired entries
    {
        std::lock_guard<std::mutex> lock(retired_mutex_);
        for (auto* entry : retired_entries_) {
            delete entry;
        }
        retired_entries_.clear();
    }
    // Phase 34: eviction_queues_ list pointers become dangling after entries are deleted
    // above. Clear them to prevent accidental double-free in destructor order variations.
    for (size_t i = 0; i < num_numa_nodes_; ++i) {
        auto& eq = eviction_queues_[i];
        eq.probation_head = nullptr;
        eq.probation_tail = nullptr;
        eq.protected_head = nullptr;
        eq.protected_tail = nullptr;
    }
    shards_.reset();
    for (size_t i = 0; i < arena_bases_.size(); ++i) {
        if (arena_bases_[i]) {
            llama_unregister_host_memory(arena_bases_[i]);
            get_memory_manager()->free_huge_arena(arena_bases_[i], arena_sizes_[i]);
        }
    }
}

uint64_t NexusBlockCache::get_current_generation(void* mapped_data) const {
    int node_idx = get_node_for_ptr(mapped_data);
    if (node_idx != -1) {
        size_t page_idx = (static_cast<char*>(mapped_data) - static_cast<char*>(arena_bases_[node_idx])) / NEXUS_PAGE_ALIGNMENT;
        if (page_idx < page_generations_[node_idx].size()) {
            return page_generations_[node_idx][page_idx].load(std::memory_order_acquire);
        }
    }
    return 0;
}

std::expected<std::shared_ptr<AeonToolBlock>, NexusErrorCode> NexusBlockCache::get_or_load(std::string_view atb_filepath) {
    uint64_t hash = get_filepath_hash(atb_filepath);
    size_t shard_idx = hash % 64;
    auto& shard = (*shards_)[shard_idx];

    CacheEntry* entry = nullptr;
    {
        std::shared_lock<std::shared_mutex> read_lock(shard.mutex);
        auto it = shard.cache_map.find(atb_filepath);
        if (it != shard.cache_map.end()) {
            entry = it->second;
        }
    }

    if (entry) {
        HazardGuard guard(entry);
        if (guard.entry) {
            entry->referenced.store(true, std::memory_order_relaxed);
            auto block_or_err = entry->future.get();
            return block_or_err;
        }
    }

    std::error_code ec;
    std::string filepath_str(atb_filepath);
    size_t actual_file_size = std::filesystem::file_size(filepath_str, ec);
    if (ec || actual_file_size < sizeof(AeonToolBlockHeader)) {
        throw std::runtime_error("NexusBlockCache: Failed to stat ATB file or file too small: " + filepath_str);
    }

    // Allocate memory block from target GPU NUMA node
    int target_node = discover_gpu_numa_node();
    if (target_node < 0 || target_node >= static_cast<int>(allocators_.size())) {
        target_node = 0;
    }

    // 1. Evict to make room (locks and unlocks target NUMA eviction queue internally)
    evict_to_limit(actual_file_size, target_node);

    // 2. Lock both NUMA eviction queue and shard mutex in strict order to avoid deadlock:
    //    eviction_queues_[target_node].mtx  →  shard.mutex
    std::unique_lock<std::mutex> evict_lock(eviction_queues_[target_node].mtx);
    std::unique_lock<std::shared_mutex> write_lock(shard.mutex);

    // Recheck under lock
    auto it = shard.cache_map.find(atb_filepath);
    if (it != shard.cache_map.end()) {
        entry = it->second;
        HazardGuard guard(entry);
        write_lock.unlock();
        evict_lock.unlock();
        current_pinned_bytes_.fetch_sub(actual_file_size, std::memory_order_relaxed);
        if (guard.entry) {
            entry->referenced.store(true, std::memory_order_relaxed);
            auto block_or_err = entry->future.get();
            return block_or_err;
        }
    }

    void* mapped_data = allocators_[target_node]->allocate(actual_file_size);
    if (mapped_data) {
        uint64_t gen = global_generation_.fetch_add(1, std::memory_order_relaxed);
        auto promise = std::make_shared<std::promise<std::expected<std::shared_ptr<AeonToolBlock>, NexusErrorCode>>>();
        std::shared_future<std::expected<std::shared_ptr<AeonToolBlock>, NexusErrorCode>> future = promise->get_future().share();

        entry = new CacheEntry();
        entry->future = future;
        entry->generation_id = gen;
        entry->filepath = filepath_str;
        entry->byte_size = actual_file_size;
        entry->is_protected = false;
        entry->referenced.store(false, std::memory_order_relaxed);
        entry->evicted = false;

        HazardGuard guard(entry); // Protect during load

        // Set page generations and entries
        size_t start_page = (static_cast<char*>(mapped_data) - static_cast<char*>(arena_bases_[target_node])) / NEXUS_PAGE_ALIGNMENT;
        size_t num_pages = (actual_file_size + NEXUS_PAGE_ALIGNMENT - 1) / NEXUS_PAGE_ALIGNMENT;
        for (size_t i = 0; i < num_pages; ++i) {
            page_generations_[target_node][start_page + i].store(gen, std::memory_order_release);
            page_entries_[target_node][start_page + i].store(entry, std::memory_order_release);
        }

        // Push new entry to this NUMA node's probation list and register in shard
        entry->numa_node = target_node;
        auto& eq = eviction_queues_[target_node];
        list_push_front(eq.probation_head, eq.probation_tail, entry);
        shard.cache_map.emplace(filepath_str, entry);

        write_lock.unlock();
        evict_lock.unlock();

        auto safe_ctx = std::make_unique<LoadContext>(
            this, filepath_str, mapped_data, actual_file_size, gen, target_node, promise, telemetry_
        );

        bool submit_success = get_numa_pool_manager().get_pool(target_node).submit([ctx = std::move(safe_ctx)]() mutable {
            try {
                auto block = std::make_shared<AeonToolBlock>(
                    ctx->filepath, ctx->mapped_data, ctx->size,
                    [cache = ctx->cache, size = ctx->size, telemetry = ctx->telemetry](void* ptr) {
                        // Phase 58: Shared telemetry block to avoid UAF
                        telemetry->blocks_freed.fetch_add(1, std::memory_order_relaxed);
#if defined(__APPLE__)
                        std::memset(ptr, 0, 128);
#endif
                        int node_idx = cache->get_node_for_ptr(ptr);
                        if (node_idx != -1) {
                            size_t start_page = (static_cast<char*>(ptr) - static_cast<char*>(cache->arena_bases_[node_idx])) / NEXUS_PAGE_ALIGNMENT;
                            size_t num_pages = (size + NEXUS_PAGE_ALIGNMENT - 1) / NEXUS_PAGE_ALIGNMENT;
                            for (size_t i = 0; i < num_pages; ++i) {
                                cache->page_generations_[node_idx][start_page + i].store(0, std::memory_order_release);
                                cache->page_entries_[node_idx][start_page + i].store(nullptr, std::memory_order_release);
                            }
                            cache->allocators_[node_idx]->free(ptr, size);
                        } else {
                            telemetry->load_errors.fetch_add(1, std::memory_order_relaxed);
                        }
                    },
                    [cache = ctx->cache, filepath = ctx->filepath, gen = ctx->gen]() {
                        cache->on_load_failure(filepath, gen);
                    },
                    ctx->gen, ctx->target_node,
                    ctx->telemetry
                );
                
                block->wait_until_resident();
                ctx->promise->set_value(block);
                ctx->promise_fulfilled.store(true, std::memory_order_release);
            } catch (const std::bad_alloc&) {
                ctx->telemetry->load_errors.fetch_add(1, std::memory_order_relaxed);
                ctx->cache->on_load_failure(ctx->filepath, ctx->gen);
                ctx->promise->set_value(std::unexpected(NexusErrorCode::RESOURCE_EXHAUSTED));
                ctx->promise_fulfilled.store(true, std::memory_order_release);
            } catch (const std::exception&) {
                ctx->telemetry->load_errors.fetch_add(1, std::memory_order_relaxed);
                ctx->cache->on_load_failure(ctx->filepath, ctx->gen);
                ctx->promise->set_value(std::unexpected(NexusErrorCode::INTERNAL_ERROR));
                ctx->promise_fulfilled.store(true, std::memory_order_release);
            } catch (...) {
                ctx->telemetry->load_errors.fetch_add(1, std::memory_order_relaxed);
                ctx->cache->on_load_failure(ctx->filepath, ctx->gen);
                ctx->promise->set_value(std::unexpected(NexusErrorCode::INTERNAL_ERROR));
                ctx->promise_fulfilled.store(true, std::memory_order_release);
            }
        });

        if (!submit_success) {
            telemetry_->submit_rejections.fetch_add(1, std::memory_order_relaxed);
            on_load_failure(filepath_str, gen);
            // safe_ctx goes out of scope and handles promise fulfillment.
        }

        auto block_or_err = entry->future.get();
        return block_or_err;
    }

    write_lock.unlock();
    evict_lock.unlock();
    current_pinned_bytes_.fetch_sub(actual_file_size, std::memory_order_relaxed);

    return std::unexpected(NexusErrorCode::RESOURCE_EXHAUSTED);
}

CacheEntry* NexusBlockCache::get_entry_from_ptr(void* mapped_data) const {
    int node_idx = get_node_for_ptr(mapped_data);
    if (node_idx != -1) {
        size_t page_idx = (static_cast<char*>(mapped_data) - static_cast<char*>(arena_bases_[node_idx])) / NEXUS_PAGE_ALIGNMENT;
        if (page_idx < page_entries_[node_idx].size()) {
            return page_entries_[node_idx][page_idx].load(std::memory_order_acquire);
        }
    }
    return nullptr;
}

void NexusBlockCache::retire_entry(CacheEntry* entry) {
    if (!entry) return;
    std::lock_guard<std::mutex> lock(retired_mutex_);
    retired_entries_.push_back(entry);
}

void NexusBlockCache::reclaim_retired_entries() {
    std::lock_guard<std::mutex> lock(retired_mutex_);
    auto it = retired_entries_.begin();
    while (it != retired_entries_.end()) {
        CacheEntry* entry = *it;
        if (!is_entry_protected(entry)) {
            delete entry;
            it = retired_entries_.erase(it);
        } else {
            ++it;
        }
    }
}

namespace nexus {
    int get_thread_slot_index() {
        static std::atomic<int> next_slot{0};
        thread_local int slot = next_slot.fetch_add(1, std::memory_order_relaxed) % NexusBlockCache::MAX_THREADS;
        return slot;
    }
}

void NexusBlockCache::release_entry(CacheEntry* entry) {
    // No-op under Hazard Pointer scheme
}

HazardGuard::HazardGuard(CacheEntry* e) : entry(e) {
    if (entry) {
        int slot = nexus::get_thread_slot_index();
        NexusBlockCache::hazard_pointers[slot].entry.store(entry, std::memory_order_release);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (entry->evicted.load(std::memory_order_acquire)) {
            NexusBlockCache::hazard_pointers[slot].entry.store(nullptr, std::memory_order_release);
            entry = nullptr;
        }
    }
}

HazardGuard::~HazardGuard() {
    release();
}

void HazardGuard::release() {
    if (entry) {
        int slot = nexus::get_thread_slot_index();
        NexusBlockCache::hazard_pointers[slot].entry.store(nullptr, std::memory_order_release);
        entry = nullptr;
    }
}

HazardGuard::HazardGuard(HazardGuard&& other) noexcept : entry(other.entry) {
    other.entry = nullptr;
}

HazardGuard& HazardGuard::operator=(HazardGuard&& other) noexcept {
    if (this != &other) {
        release();
        entry = other.entry;
        other.entry = nullptr;
    }
    return *this;
}

HazardGuard NexusBlockCache::acquire_hazard(void* mapped_data) {
    CacheEntry* entry = get_entry_from_ptr(mapped_data);
    return HazardGuard(entry);
}

void NexusBlockCache::release_hazard(void* mapped_data) {
    CacheEntry* entry = get_entry_from_ptr(mapped_data);
    if (entry) {
        int slot = nexus::get_thread_slot_index();
        NexusBlockCache::hazard_pointers[slot].entry.store(nullptr, std::memory_order_release);
    }
}

void NexusBlockCache::register_tool_path(uint32_t tool_id, const std::string& path) {
    std::unique_lock<std::shared_mutex> lock(tool_id_map_mutex_);
    tool_id_to_path_[tool_id] = path;
}

void NexusBlockCache::pin_tool(uint32_t tool_id) {
    std::unique_lock<std::shared_mutex> lock(pinned_mutex_);
    pinned_tool_ids_.insert(tool_id);
}

void NexusBlockCache::unpin_tool(uint32_t tool_id) {
    std::unique_lock<std::shared_mutex> lock(pinned_mutex_);
    pinned_tool_ids_.erase(tool_id);
}

bool NexusBlockCache::is_tool_pinned(uint32_t tool_id) const {
    std::shared_lock<std::shared_mutex> lock(pinned_mutex_);
    return pinned_tool_ids_.count(tool_id) > 0;
}

bool NexusBlockCache::is_path_pinned(const std::string& path) const {
    std::shared_lock<std::shared_mutex> pin_lock(pinned_mutex_);
    if (pinned_tool_ids_.empty()) {
        return false;
    }
    std::shared_lock<std::shared_mutex> map_lock(tool_id_map_mutex_);
    for (uint32_t tool_id : pinned_tool_ids_) {
        auto it = tool_id_to_path_.find(tool_id);
        if (it != tool_id_to_path_.end() && it->second == path) {
            return true;
        }
    }
    return false;
}

HazardGuard NexusBlockCache::hazard_guard(uint32_t tool_id) {
    std::shared_lock<std::shared_mutex> lock(tool_id_map_mutex_);
    auto it = tool_id_to_path_.find(tool_id);
    if (it == tool_id_to_path_.end()) {
        throw std::runtime_error("Tool ID not registered in cache: " + std::to_string(tool_id));
    }
    std::string path = it->second;
    lock.unlock();

    uint64_t hash = get_filepath_hash(path);
    size_t shard_idx = hash % 64;
    auto& shard = (*shards_)[shard_idx];

    std::shared_lock<std::shared_mutex> shard_lock(shard.mutex);
    auto entry_it = shard.cache_map.find(path);
    if (entry_it == shard.cache_map.end()) {
        throw std::runtime_error("Tool block not found in cache shards for path: " + path);
    }
    return HazardGuard(entry_it->second);
}

void NexusBlockCache::prefetch(std::string_view atb_filepath) {
    uint64_t hash = get_filepath_hash(atb_filepath);
    size_t shard_idx = hash % 64;
    auto& shard = (*shards_)[shard_idx];

    CacheEntry* entry = nullptr;
    {
        std::shared_lock<std::shared_mutex> read_lock(shard.mutex);
        auto it = shard.cache_map.find(atb_filepath);
        if (it != shard.cache_map.end()) {
            entry = it->second;
        }
    }

    std::shared_ptr<AeonToolBlock> block = nullptr;
    if (entry) {
        HazardGuard guard(entry);
        if (guard.entry) {
            entry->referenced.store(true, std::memory_order_relaxed);
            auto res = entry->future.get();
            if (res.has_value()) [[likely]] {
                block = res.value();
            } else [[unlikely]] {
                // handle error
            }
        }
    } else {
        try {
            auto res = get_or_load(atb_filepath);
            if (res.has_value()) [[likely]] {
                block = res.value();
            } else [[unlikely]] {
                // handle error
            }
        } catch (...) {
            return;
        }
    }

    if (block) {
        prefetch_memory(block->get_mapped_data(), block->get_file_size());
    }
}

void NexusBlockCache::on_load_failure(std::string_view atb_filepath, uint64_t failed_gen) {
    uint64_t hash = get_filepath_hash(atb_filepath);
    size_t shard_idx = hash % 64;
    auto& shard = (*shards_)[shard_idx];

    // Phase 34: We need the entry to find which NUMA queue it belongs to.
    // Lock shard first to find the entry's numa_node, then lock the eviction queue.
    // Since we might not know the node yet, peek under shard lock first.
    int entry_node = 0;
    {
        std::shared_lock<std::shared_mutex> peek_lock(shard.mutex);
        auto it = shard.cache_map.find(atb_filepath);
        if (it != shard.cache_map.end() && it->second->generation_id == failed_gen) {
            entry_node = it->second->numa_node;
        } else {
            return; // Entry already removed or generation mismatch
        }
    }

    std::unique_lock<std::mutex> evict_lock(eviction_queues_[entry_node].mtx);
    std::unique_lock<std::shared_mutex> cleanup_lock(shard.mutex);

    auto it = shard.cache_map.find(atb_filepath);
    if (it != shard.cache_map.end() && it->second->generation_id == failed_gen) {
        CacheEntry* entry = it->second;
        auto& eq = eviction_queues_[entry->numa_node];
        if (entry->is_protected) {
            list_remove(eq.protected_head, eq.protected_tail, entry);
        } else {
            list_remove(eq.probation_head, eq.probation_tail, entry);
        }
        size_t evicted_size = entry->byte_size;
        shard.cache_map.erase(it);
        entry->evicted.store(true, std::memory_order_release);
        current_pinned_bytes_.fetch_sub(evicted_size, std::memory_order_relaxed);

        cleanup_lock.unlock();
        evict_lock.unlock();

        if (!is_entry_protected(entry)) {
            delete entry;
        } else {
            retire_entry(entry);
        }
    }
}

void NexusBlockCache::evict_to_limit(size_t required_bytes, int target_node) {
    reclaim_retired_entries();
    auto try_evict_from_node = [&](int node_idx) -> bool {
        std::unique_lock<std::mutex> evict_lock(eviction_queues_[node_idx].mtx);
        auto& eq = eviction_queues_[node_idx];
        CacheEntry* candidate = nullptr;

        // Pass 1: Scan probation tailward for an unreferenced, inactive entry
        CacheEntry* curr = eq.probation_tail;
        while (curr) {
            if (is_entry_protected(curr)) {
                curr = curr->prev_lru;
                continue;
            }
            if (curr->referenced.load(std::memory_order_relaxed)) {
                // Promote probation block to protected queue
                curr->referenced.store(false, std::memory_order_relaxed);
                curr->is_protected = true;
                CacheEntry* prev = curr->prev_lru;
                list_remove(eq.probation_head, eq.probation_tail, curr);
                list_push_front(eq.protected_head, eq.protected_tail, curr);
                curr = prev;
            } else {
                if (!is_path_pinned(curr->filepath)) {
                    candidate = curr;
                    break;
                }
                curr = curr->prev_lru;
            }
        }

        // Pass 2: Sweep protected list tailward to demote inactive entries
        if (!candidate) {
            curr = eq.protected_tail;
            while (curr) {
                if (is_entry_protected(curr)) {
                    curr = curr->prev_lru;
                    continue;
                }
                if (curr->referenced.load(std::memory_order_relaxed)) {
                    curr->referenced.store(false, std::memory_order_relaxed);
                    CacheEntry* prev = curr->prev_lru;
                    list_remove(eq.protected_head, eq.protected_tail, curr);
                    list_push_front(eq.protected_head, eq.protected_tail, curr);
                    curr = prev;
                } else {
                    // Demote protected block to probation queue and mark as candidate
                    curr->is_protected = false;
                    CacheEntry* prev = curr->prev_lru;
                    list_remove(eq.protected_head, eq.protected_tail, curr);
                    list_push_front(eq.probation_head, eq.probation_tail, curr);
                    if (!is_path_pinned(curr->filepath)) {
                        candidate = curr;
                        break;
                    }
                    curr = prev;
                }
            }
        }

        if (!candidate) {
            return false; // No evictable candidates on this node
        }

        // Remove from list
        if (candidate->is_protected) {
            list_remove(eq.protected_head, eq.protected_tail, candidate);
        } else {
            list_remove(eq.probation_head, eq.probation_tail, candidate);
        }

        // Release eviction lock before acquiring shard lock (lock ordering)
        evict_lock.unlock();

        // Erase from shard map
        uint64_t hash = get_filepath_hash(candidate->filepath);
        size_t shard_idx = hash % 64;
        auto& shard = (*shards_)[shard_idx];

        {
            std::unique_lock<std::shared_mutex> shard_lock(shard.mutex);
            shard.cache_map.erase(candidate->filepath);
        }

        size_t evicted_size = candidate->byte_size;
        candidate->evicted.store(true, std::memory_order_release);
        if (!is_entry_protected(candidate)) {
            delete candidate;
        } else {
            retire_entry(candidate);
        }
        current_pinned_bytes_.fetch_sub(evicted_size, std::memory_order_relaxed);
        return true;
    };

    while (current_pinned_bytes_.load(std::memory_order_relaxed) + required_bytes > max_pinned_bytes_) {
        if (!try_evict_from_node(target_node)) {
            break; // No more evictable blocks on this node
        }
    }

    current_pinned_bytes_.fetch_add(required_bytes, std::memory_order_relaxed);
}

NexusNUMAThreadPoolManager& get_numa_pool_manager() {
    static NexusNUMAThreadPoolManager manager;
    return manager;
}

NexusThreadPool& get_global_thread_pool() {
    return get_numa_pool_manager().get_pool(0);
}
