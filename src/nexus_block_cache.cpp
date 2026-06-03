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

#if defined(_WIN32)
#include <io.h>
#include <fcntl.h>
#include <sys/stat.h>
#define open _open
#define fstat _fstat
#define close _close
#define stat _stat
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/mman.h>
#endif

extern "C" {
    void llama_unregister_host_memory(void * buffer);
    bool llama_register_host_memory(void * buffer, size_t size);
}

AeonToolBlock::AeonToolBlock(const std::string& atb_filepath, void* mapped_data, size_t file_size, 
                             std::function<void(void*)> free_callback, std::function<void()> on_failure, uint64_t generation_id, int numa_node)
    : mapped_data_(mapped_data), file_size_(file_size), free_callback_(free_callback), generation_id_(generation_id), numa_node_(numa_node) {
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

struct NumaInitTaskCtx {
    NexusBlockCache* cache;
    int numa_node;
    std::latch* latch;
    std::atomic<bool> success{true};
    std::exception_ptr exc_ptr;
};

NexusBlockCache::NexusBlockCache(size_t max_pinned_bytes) 
    : shards_(std::make_unique<std::array<CacheShard, 64>>()) {
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

    arena_bases_.reserve(node_count);
    arena_sizes_.reserve(node_count);
    allocators_.reserve(node_count);
    page_generations_.resize(node_count);
    page_entries_.resize(node_count);

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

    std::latch latch(node_count);
    std::vector<NumaInitTaskCtx> init_contexts(node_count);

    struct LatchGuard {
        std::latch& latch_ref;
        ~LatchGuard() {
            latch_ref.count_down();
        }
    };

    for (int i = 0; i < node_count; ++i) {
        init_contexts[i].cache = this;
        init_contexts[i].numa_node = i;
        init_contexts[i].latch = &latch;

        auto worker_fn = [](void* arg) {
            auto* ctx = static_cast<NumaInitTaskCtx*>(arg);
            LatchGuard guard(*(ctx->latch));
            try {
                ctx->cache->initialize_numa_arena(ctx->numa_node);
            } catch (...) {
                ctx->success.store(false, std::memory_order_release);
                ctx->exc_ptr = std::current_exception();
            }
        };

        if (!get_numa_pool_manager().get_pool(i).submit(worker_fn, &init_contexts[i])) {
            init_contexts[i].success.store(false, std::memory_order_release);
            latch.count_down();
        }
    }

    latch.wait();

    // Validate initialization success
    for (int i = 0; i < node_count; ++i) {
        if (!init_contexts[i].success.load(std::memory_order_acquire)) {
            // Rollback already initialized and registered host memory
            for (size_t j = 0; j < arena_bases_.size(); ++j) {
                llama_unregister_host_memory(arena_bases_[j]);
                get_memory_manager()->free_huge_arena(arena_bases_[j], arena_sizes_[j]);
            }
            if (init_contexts[i].exc_ptr) {
                std::rethrow_exception(init_contexts[i].exc_ptr);
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

std::shared_ptr<AeonToolBlock> NexusBlockCache::get_or_load(std::string_view atb_filepath) {
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
        entry->last_access_tick.store(global_tick_.fetch_add(1, std::memory_order_relaxed), std::memory_order_relaxed);
        return entry->future.get();
    }

    std::error_code ec;
    std::string filepath_str(atb_filepath);
    size_t actual_file_size = std::filesystem::file_size(filepath_str, ec);
    if (ec || actual_file_size < sizeof(AeonToolBlockHeader)) {
        throw std::runtime_error("NexusBlockCache: Failed to stat ATB file or file too small: " + filepath_str);
    }

    // Evict old entries BEFORE locking the shard to prevent deadlock/livelock
    evict_to_limit(actual_file_size);

    std::unique_lock<std::shared_mutex> write_lock(shard.mutex);
    auto it = shard.cache_map.find(atb_filepath);
    if (it != shard.cache_map.end()) {
        entry = it->second;
        write_lock.unlock();
        current_pinned_bytes_.fetch_sub(actual_file_size, std::memory_order_relaxed);
        entry->last_access_tick.store(global_tick_.fetch_add(1, std::memory_order_relaxed), std::memory_order_relaxed);
        return entry->future.get();
    }

    uint64_t gen = global_generation_.fetch_add(1, std::memory_order_relaxed);
    auto promise = std::make_shared<std::promise<std::shared_ptr<AeonToolBlock>>>();
    std::shared_future<std::shared_ptr<AeonToolBlock>> future = promise->get_future().share();

    // Allocate memory block from target GPU NUMA node
    int target_node = discover_gpu_numa_node();
    if (target_node < 0 || target_node >= static_cast<int>(allocators_.size())) {
        target_node = 0;
    }

    void* mapped_data = allocators_[target_node]->allocate(actual_file_size);
    if (!mapped_data) {
        // Under high concurrency, another thread might have stolen our evicted page slot.
        // Unlock, evict another block, and retry up to 5 times.
        for (int retry = 0; retry < 5; ++retry) {
            write_lock.unlock();
            evict_to_limit(actual_file_size);
            write_lock.lock();

            // Check if another thread loaded the exact same tool block in the meantime
            auto retry_it = shard.cache_map.find(atb_filepath);
            if (retry_it != shard.cache_map.end()) {
                entry = retry_it->second;
                write_lock.unlock();
                current_pinned_bytes_.fetch_sub(actual_file_size, std::memory_order_relaxed);
                entry->last_access_tick.store(global_tick_.fetch_add(1, std::memory_order_relaxed), std::memory_order_relaxed);
                return entry->future.get();
            }

            mapped_data = allocators_[target_node]->allocate(actual_file_size);
            if (mapped_data) {
                break;
            }
        }
    }

    if (!mapped_data) {
        write_lock.unlock();
        current_pinned_bytes_.fetch_sub(actual_file_size, std::memory_order_relaxed);
        throw std::runtime_error("NexusBlockCache: Out of HugeTLB memory pool for " + filepath_str);
    }

    entry = new CacheEntry();
    entry->future = future;
    entry->last_access_tick.store(global_tick_.fetch_add(1, std::memory_order_relaxed), std::memory_order_relaxed);
    entry->generation_id = gen;
    entry->filepath = filepath_str;
    entry->byte_size = actual_file_size;

    // Set page generations and entries
    size_t start_page = (static_cast<char*>(mapped_data) - static_cast<char*>(arena_bases_[target_node])) / NEXUS_PAGE_ALIGNMENT;
    size_t num_pages = (actual_file_size + NEXUS_PAGE_ALIGNMENT - 1) / NEXUS_PAGE_ALIGNMENT;
    for (size_t i = 0; i < num_pages; ++i) {
        page_generations_[target_node][start_page + i].store(gen, std::memory_order_release);
        page_entries_[target_node][start_page + i].store(entry, std::memory_order_release);
    }

    entry->vector_index = shard.keys.size();
    shard.keys.push_back(entry);
    shard.cache_map.emplace(filepath_str, entry);
    write_lock.unlock();

    std::shared_ptr<AeonToolBlock> block = nullptr;
    try {
        block = std::make_shared<AeonToolBlock>(filepath_str, mapped_data, actual_file_size, [this, actual_file_size](void* ptr) {
#if defined(__APPLE__)
            std::memset(ptr, 0, 128);
#endif
            int node_idx = this->get_node_for_ptr(ptr);
            if (node_idx != -1) {
                size_t start_page = (static_cast<char*>(ptr) - static_cast<char*>(this->arena_bases_[node_idx])) / NEXUS_PAGE_ALIGNMENT;
                size_t num_pages = (actual_file_size + NEXUS_PAGE_ALIGNMENT - 1) / NEXUS_PAGE_ALIGNMENT;
                for (size_t i = 0; i < num_pages; ++i) {
                    this->page_generations_[node_idx][start_page + i].store(0, std::memory_order_release);
                    this->page_entries_[node_idx][start_page + i].store(nullptr, std::memory_order_release);
                }
                this->allocators_[node_idx]->free(ptr, actual_file_size);
            }
        }, [this, filepath_str, gen]() {
            this->on_load_failure(filepath_str, gen);
        }, gen, target_node);
        promise->set_value(block);
    } catch (...) {
        promise->set_exception(std::current_exception());
        this->on_load_failure(filepath_str, gen);
        throw;
    }

    return block;
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

HazardGuard::HazardGuard(CacheEntry* e) : entry(e) {
    if (entry) {
        entry->active_readers.fetch_add(1, std::memory_order_acquire);
    }
}

HazardGuard::~HazardGuard() {
    if (entry) {
        entry->active_readers.fetch_sub(1, std::memory_order_release);
    }
}

HazardGuard::HazardGuard(HazardGuard&& other) noexcept : entry(other.entry) {
    other.entry = nullptr;
}

HazardGuard& HazardGuard::operator=(HazardGuard&& other) noexcept {
    if (this != &other) {
        if (entry) {
            entry->active_readers.fetch_sub(1, std::memory_order_release);
        }
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
        entry->active_readers.fetch_sub(1, std::memory_order_release);
    }
}

void NexusBlockCache::register_tool_path(uint32_t tool_id, const std::string& path) {
    std::unique_lock<std::shared_mutex> lock(tool_id_map_mutex_);
    tool_id_to_path_[tool_id] = path;
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
        entry->last_access_tick.store(global_tick_.fetch_add(1, std::memory_order_relaxed), std::memory_order_relaxed);
        try {
            block = entry->future.get();
        } catch (...) {
            return;
        }
    } else {
        try {
            block = get_or_load(atb_filepath);
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

    std::unique_lock<std::shared_mutex> cleanup_lock(shard.mutex);
    auto it = shard.cache_map.find(atb_filepath);
    if (it != shard.cache_map.end() && it->second->generation_id == failed_gen) {
        CacheEntry* entry = it->second;
        size_t v_idx = entry->vector_index;
        if (v_idx < shard.keys.size()) {
            if (v_idx != shard.keys.size() - 1) {
                CacheEntry* back_entry = shard.keys.back();
                shard.keys[v_idx] = back_entry;
                back_entry->vector_index = v_idx;
            }
            shard.keys.pop_back();
        }
        size_t evicted_size = entry->byte_size;
        shard.cache_map.erase(it);
        delete entry;
        current_pinned_bytes_.fetch_sub(evicted_size, std::memory_order_relaxed);
    }
}

void NexusBlockCache::evict_to_limit(size_t required_bytes) {
    constexpr size_t K = 5;

    while (current_pinned_bytes_.load(std::memory_order_relaxed) + required_bytes > max_pinned_bytes_) {
        CacheEntry* oldest_entry = nullptr;
        uint64_t lowest_tick = std::numeric_limits<uint64_t>::max();
        size_t oldest_shard_idx = 0;

        size_t sampled = 0;
        for (size_t s = 0; s < K; ++s) {
            size_t s_idx = nexus::get_random_value() % 64;
            auto& shard = (*shards_)[s_idx];

            std::shared_lock<std::shared_mutex> shard_lock(shard.mutex);
            if (!shard.keys.empty()) {
                size_t k_idx = nexus::get_random_value() % shard.keys.size();
                CacheEntry* entry = shard.keys[k_idx];
                if (entry) {
                    if (entry->active_readers.load(std::memory_order_relaxed) > 0) {
                        continue;
                    }
                    auto status = entry->future.wait_for(std::chrono::seconds(0));
                    if (status == std::future_status::ready) {
                        try {
                            auto block = entry->future.get();
                            if (block) {
                                uint64_t tick = entry->last_access_tick.load(std::memory_order_relaxed);
                                if (tick < lowest_tick) {
                                    lowest_tick = tick;
                                    oldest_shard_idx = s_idx;
                                    oldest_entry = entry;
                                }
                                sampled++;
                            }
                        } catch (...) {
                            oldest_shard_idx = s_idx;
                            oldest_entry = entry;
                            break;
                        }
                    }
                }
            }
        }

        if (!oldest_entry) {
            break;
        }

        auto& target_shard = (*shards_)[oldest_shard_idx];
        std::unique_lock<std::shared_mutex> write_lock(target_shard.mutex);
        auto it = target_shard.cache_map.find(oldest_entry->filepath);
        if (it != target_shard.cache_map.end() && it->second == oldest_entry) {
            size_t v_idx = oldest_entry->vector_index;
            if (v_idx < target_shard.keys.size()) {
                if (v_idx != target_shard.keys.size() - 1) {
                    CacheEntry* back_entry = target_shard.keys.back();
                    target_shard.keys[v_idx] = back_entry;
                    back_entry->vector_index = v_idx;
                }
                target_shard.keys.pop_back();
            }
            size_t evicted_size = oldest_entry->byte_size;
            target_shard.cache_map.erase(it);
            delete oldest_entry;
            current_pinned_bytes_.fetch_sub(evicted_size, std::memory_order_relaxed);
        }
    }

    current_pinned_bytes_.fetch_add(required_bytes, std::memory_order_relaxed);
}
