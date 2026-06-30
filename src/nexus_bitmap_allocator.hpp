#pragma once

#include <cstdint>
#include <vector>
#include <stdexcept>
#include <cstddef>
#include <algorithm>
#include <cassert>
#include <mutex>
#include <random>

#ifndef NEXUS_PAGE_ALIGNMENT
#define NEXUS_PAGE_ALIGNMENT (2 * 1024 * 1024) // 2MB HugeTLB alignment
#endif

#ifndef NEXUS_CACHE_LINE
#define NEXUS_CACHE_LINE 128
#endif

namespace nexus {

// Xoshiro256++ PRNG implementation aligned to cache line (used by block cache)
class alignas(NEXUS_CACHE_LINE) Xoshiro256PlusPlus {
private:
    uint64_t s[4];

    static inline uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

public:
    explicit Xoshiro256PlusPlus(uint64_t seed) {
        uint64_t x = seed;
        for (int i = 0; i < 4; ++i) {
            x += 0x9e3779b97f4a7c15ULL;
            uint64_t z = x;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            s[i] = z ^ (z >> 31);
        }
    }

    uint64_t operator()() {
        const uint64_t result = rotl(s[0] + s[3], 23) + s[0];
        const uint64_t t = s[1] << 17;

        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];

        s[2] ^= t;
        s[3] = rotl(s[3], 45);

        return result;
    }
};

inline uint64_t get_random_value() {
    thread_local Xoshiro256PlusPlus prng(std::random_device{}());
    return prng();
}

/**
 * Power-of-2 Buddy Allocator operating on 2MB-aligned HugeTLB arenas.
 *
 * MEMORY SAFETY NOTE (Phase 34):
 * The buddy calculation uses XOR on zero-based BLOCK INDICES, not on raw
 * absolute pointers. Given block index `i` = (ptr - arena_base) / PAGE_SIZE,
 * the buddy at order `k` is `i ^ (1 << k)`. Because `i` is a relative index
 * starting from 0, the XOR is mathematically correct regardless of the OS
 * mmap base address alignment. The arena base alignment only needs to satisfy
 * the page size (2MB for HugeTLB), not the maximum block size.
 *
 * The constructor enforces that base_ptr is 2MB-aligned to prevent offset
 * calculation errors from misaligned subtraction.
 */
class alignas(NEXUS_CACHE_LINE) QuantizedBitmapAllocator {
private:
    struct BuddyNode {
        size_t prev = -1;
        size_t next = -1;
        bool free = false;
        size_t order = 0;
    };

    // Read-mostly geometry: touched on every alloc/free for offset math, never written
    // after construction. Kept together so they share clean (non-bouncing) cache lines.
    void* base_ptr_;
    size_t total_size_;
    size_t num_blocks_;
    size_t max_order_;
    std::vector<BuddyNode> nodes_;
    std::vector<size_t> free_heads_;
    // Pin the contended lock to its own cache line so the atomic/lock traffic from one
    // allocator can never invalidate (RFO-storm) the read-mostly geometry above or a
    // neighbouring object's line. alignas on the class itself rounds sizeof up to a
    // 128B multiple, so adjacent allocators also start on clean boundaries.
    alignas(NEXUS_CACHE_LINE) std::mutex mutex_;

    void push_to_free_list(size_t i, size_t order) {
        nodes_[i].free = true;
        nodes_[i].order = order;
        nodes_[i].next = free_heads_[order];
        nodes_[i].prev = -1;
        if (free_heads_[order] != -1) {
            nodes_[free_heads_[order]].prev = i;
        }
        free_heads_[order] = i;
    }

    void remove_from_free_list(size_t i, size_t order) {
        size_t prev_idx = nodes_[i].prev;
        size_t next_idx = nodes_[i].next;
        if (prev_idx != -1) {
            nodes_[prev_idx].next = next_idx;
        } else {
            free_heads_[order] = next_idx;
        }
        if (next_idx != -1) {
            nodes_[next_idx].prev = prev_idx;
        }
        nodes_[i].free = false;
        nodes_[i].next = -1;
        nodes_[i].prev = -1;
    }

public:
    QuantizedBitmapAllocator(void* base_ptr, size_t total_size)
        : base_ptr_(base_ptr), total_size_(total_size) {
        // Phase 34: Validate arena base is page-aligned to prevent offset math errors
        if (reinterpret_cast<uintptr_t>(base_ptr) % NEXUS_PAGE_ALIGNMENT != 0) {
            throw std::invalid_argument(
                "Arena base must be aligned to NEXUS_PAGE_ALIGNMENT (2MB). "
                "Got address: " + std::to_string(reinterpret_cast<uintptr_t>(base_ptr)));
        }
        if (total_size % NEXUS_PAGE_ALIGNMENT != 0) {
            throw std::invalid_argument("Total size must be a multiple of 2MB");
        }
        num_blocks_ = total_size / NEXUS_PAGE_ALIGNMENT;
        if (num_blocks_ == 0) {
            throw std::invalid_argument("Total size must be at least 2MB");
        }

        max_order_ = 0;
        while ((1ULL << max_order_) < num_blocks_) {
            max_order_++;
        }

        nodes_.resize(num_blocks_);
        free_heads_.assign(max_order_ + 1, -1);

        // Binary decomposition of the range [0, num_blocks_ - 1] into buddy-aligned blocks
        size_t curr = 0;
        while (curr < num_blocks_) {
            size_t remaining = num_blocks_ - curr;
            size_t block_size = 1;
            size_t order = 0;
            while (block_size * 2 <= remaining && (curr % (block_size * 2) == 0)) {
                block_size *= 2;
                order++;
            }
            push_to_free_list(curr, order);
            curr += block_size;
        }
    }

    QuantizedBitmapAllocator(const QuantizedBitmapAllocator&) = delete;
    QuantizedBitmapAllocator& operator=(const QuantizedBitmapAllocator&) = delete;

    void* allocate(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t needed_blocks = (size + NEXUS_PAGE_ALIGNMENT - 1) / NEXUS_PAGE_ALIGNMENT;
        if (needed_blocks == 0) return nullptr;

        size_t order = 0;
        while ((1ULL << order) < needed_blocks) {
            order++;
        }

        if (order > max_order_) return nullptr;

        size_t found_order = order;
        while (found_order <= max_order_ && free_heads_[found_order] == -1) {
            found_order++;
        }

        if (found_order > max_order_) {
            return nullptr; // No block available
        }

        size_t i = free_heads_[found_order];
        remove_from_free_list(i, found_order);

        while (found_order > order) {
            found_order--;
            size_t buddy = i + (1ULL << found_order);
            push_to_free_list(buddy, found_order);
        }

        nodes_[i].order = order;
        nodes_[i].free = false;
        return static_cast<char*>(base_ptr_) + i * NEXUS_PAGE_ALIGNMENT;
    }

    void free(void* ptr, size_t size) {
        if (!ptr) return;
        std::lock_guard<std::mutex> lock(mutex_);
        size_t offset = static_cast<char*>(ptr) - static_cast<char*>(base_ptr_);
        size_t i = offset / NEXUS_PAGE_ALIGNMENT;
        size_t needed_blocks = (size + NEXUS_PAGE_ALIGNMENT - 1) / NEXUS_PAGE_ALIGNMENT;

        size_t order = 0;
        while ((1ULL << order) < needed_blocks) {
            order++;
        }

        while (order < max_order_) {
            // XOR on block index `i` (not raw pointer) to find buddy.
            // This is safe because `i` is a zero-based index: i = offset / PAGE_SIZE.
            size_t buddy = i ^ (1ULL << order);
            // Bounds guard: for non-power-of-2 arena sizes, XOR can exceed num_blocks_
            assert(buddy != i && "XOR buddy equals self — logic error in order tracking");
            if (buddy >= num_blocks_ || !nodes_[buddy].free || nodes_[buddy].order != order) {
                break;
            }
            remove_from_free_list(buddy, order);
            i = std::min(i, buddy);
            order++;
        }

        push_to_free_list(i, order);
    }

    size_t get_free_bytes() {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t free_blocks = 0;
        for (size_t order = 0; order <= max_order_; ++order) {
            size_t curr = free_heads_[order];
            while (curr != -1) {
                free_blocks += (1ULL << order);
                curr = nodes_[curr].next;
            }
        }
        return free_blocks * NEXUS_PAGE_ALIGNMENT;
    }
};

// Phase 2 alignment hardening: the allocator object starts on a cache line and occupies
// whole lines, so concurrent allocators cannot false-share. (NEXUS_CACHE_LINE = 128.)
static_assert(alignof(QuantizedBitmapAllocator) % NEXUS_CACHE_LINE == 0,
              "QuantizedBitmapAllocator must be cache-line aligned (false-sharing guard)");
static_assert(sizeof(QuantizedBitmapAllocator) % NEXUS_CACHE_LINE == 0,
              "QuantizedBitmapAllocator size must be a whole number of cache lines");

} // namespace nexus
