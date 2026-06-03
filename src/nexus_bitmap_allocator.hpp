#pragma once

#include <cstdint>
#include <vector>
#include <atomic>
#include <stdexcept>
#include <cstddef>
#include <algorithm>
#include <random>

#ifndef NEXUS_PAGE_ALIGNMENT
#define NEXUS_PAGE_ALIGNMENT (2 * 1024 * 1024) // 2MB HugeTLB alignment
#endif

namespace nexus {

#if defined(_WIN32)
#include <intrin.h>
#pragma intrinsic(_BitScanForward64)
inline int count_trailing_zeros_64(uint64_t mask) {
    unsigned long index;
    if (_BitScanForward64(&index, mask)) {
        return static_cast<int>(index);
    }
    return 64;
}
#else
inline int count_trailing_zeros_64(uint64_t mask) {
    return __builtin_ctzll(mask);
}
#endif

// Xoshiro256++ PRNG implementation aligned to cache line
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

class QuantizedBitmapAllocator {
private:
    void* base_ptr_;
    size_t total_size_;
    std::vector<std::atomic<uint64_t>> bitset_;

public:
    QuantizedBitmapAllocator(void* base_ptr, size_t total_size)
        : base_ptr_(base_ptr), total_size_(total_size),
          bitset_(((total_size / NEXUS_PAGE_ALIGNMENT) + 63) / 64) {
        if (total_size % NEXUS_PAGE_ALIGNMENT != 0) {
            throw std::invalid_argument("Total size must be a multiple of 2MB");
        }
        for (size_t i = 0; i < bitset_.size(); ++i) {
            bitset_[i].store(0ULL, std::memory_order_relaxed);
        }
    }

    QuantizedBitmapAllocator(const QuantizedBitmapAllocator&) = delete;
    QuantizedBitmapAllocator& operator=(const QuantizedBitmapAllocator&) = delete;

    void* allocate(size_t size) {
        size_t blocks_needed = (size + NEXUS_PAGE_ALIGNMENT - 1) / NEXUS_PAGE_ALIGNMENT;
        size_t num_blocks = total_size_ / NEXUS_PAGE_ALIGNMENT;
        size_t num_words = bitset_.size();

        thread_local size_t next_search_index = get_random_value() % num_words;
        size_t start_word = next_search_index;

        for (size_t step = 0; step < num_words; ++step) {
            size_t w = (start_word + step) % num_words;
            uint64_t val = bitset_[w].load(std::memory_order_relaxed);
            if (val == ~0ULL) {
                continue;
            }
            int bit = count_trailing_zeros_64(~val);
            size_t i = w * 64 + bit;
            if (i + blocks_needed > num_blocks) {
                continue;
            }

            // Check if the contiguous range starting at i is free
            bool range_free = true;
            size_t j = 0;
            for (; j < blocks_needed; ++j) {
                size_t check_word = (i + j) / 64;
                size_t check_bit = (i + j) % 64;
                uint64_t cv = bitset_[check_word].load(std::memory_order_relaxed);
                if (cv & (1ULL << check_bit)) {
                    range_free = false;
                    break;
                }
            }

            if (!range_free) {
                // Skip to the next possible starting point
                size_t next_pos = i + j + 1;
                size_t next_w = next_pos / 64;
                if (next_w > w) {
                    step += (next_w - w - 1);
                }
                continue;
            }

            // Try to reserve all blocks in the contiguous range using CAS-with-rollback
            bool success = true;
            size_t reserved_count = 0;
            for (size_t k = 0; k < blocks_needed; ++k) {
                size_t check_word = (i + k) / 64;
                size_t check_bit = (i + k) % 64;
                uint64_t mask = 1ULL << check_bit;

                uint64_t expected = bitset_[check_word].load(std::memory_order_relaxed);
                while (true) {
                    if (expected & mask) {
                        // Conflict: block already reserved by another thread
                        success = false;
                        break;
                    }
                    if (bitset_[check_word].compare_exchange_weak(expected, expected | mask, 
                                                                  std::memory_order_acquire, 
                                                                  std::memory_order_relaxed)) {
                        reserved_count++;
                        break;
                    }
                }
                if (!success) {
                    break;
                }
            }

            if (success) {
                next_search_index = (i + blocks_needed) / 64 % num_words;
                return static_cast<char*>(base_ptr_) + (i * NEXUS_PAGE_ALIGNMENT);
            }

            // Rollback already reserved blocks on failure
            for (size_t k = 0; k < reserved_count; ++k) {
                size_t check_word = (i + k) / 64;
                size_t check_bit = (i + k) % 64;
                uint64_t mask = 1ULL << check_bit;
                uint64_t expected = bitset_[check_word].load(std::memory_order_relaxed);
                while (true) {
                    uint64_t desired = expected & ~mask;
                    if (bitset_[check_word].compare_exchange_weak(expected, desired, 
                                                                  std::memory_order_release, 
                                                                  std::memory_order_relaxed)) {
                        break;
                    }
                }
            }
        }
        return nullptr;
    }

    void free(void* ptr, size_t size) {
        if (!ptr) return;
        size_t offset = static_cast<char*>(ptr) - static_cast<char*>(base_ptr_);
        size_t start_idx = offset / NEXUS_PAGE_ALIGNMENT;
        size_t blocks_needed = (size + NEXUS_PAGE_ALIGNMENT - 1) / NEXUS_PAGE_ALIGNMENT;

        for (size_t j = 0; j < blocks_needed; ++j) {
            size_t word_idx = (start_idx + j) / 64;
            size_t bit_idx = (start_idx + j) % 64;
            uint64_t mask = 1ULL << bit_idx;
            uint64_t expected = bitset_[word_idx].load(std::memory_order_relaxed);
            while (true) {
                uint64_t desired = expected & ~mask;
                if (bitset_[word_idx].compare_exchange_weak(expected, desired, 
                                                             std::memory_order_release, 
                                                             std::memory_order_relaxed)) {
                    break;
                }
            }
        }
    }
};

} // namespace nexus
