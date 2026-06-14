#pragma once
#include <atomic>
#include <vector>
#include <thread>
#include <cstddef>
#include <cstdlib>
#include <type_traits>
#include <new>
#include "nexus_os_compat.hpp"

namespace nexus {
    void poll_completions();
    void run_io_reactor(int numa_node, std::stop_token stop_tok);

    template <typename Signature>
    class InlineTask;

    template <typename ReturnType, typename... Args>
    class alignas(64) InlineTask<ReturnType(Args...)> {
    private:
        // ── VTable Trampoline (Phase 64.2) ──────────────────────────────────
        struct VTable {
            ReturnType (*invoke)(void*, Args...);
            void (*move)(void*, void*) noexcept;
            void (*destroy)(void*) noexcept;
        };

        // ── Per-type static constexpr VTable instances (Phase 64.4) ─────────
        template <typename F>
        static constexpr VTable vtable_for = {
            // invoke
            [](void* obj, Args... args) -> ReturnType {
                if constexpr (std::is_void_v<ReturnType>) {
                    (*std::launder(reinterpret_cast<F*>(obj)))(std::forward<Args>(args)...);
                } else {
                    return (*std::launder(reinterpret_cast<F*>(obj)))(std::forward<Args>(args)...);
                }
            },
            // move
            [](void* src, void* dst) noexcept {
                ::new (dst) F(std::move(*std::launder(reinterpret_cast<F*>(src))));
            },
            // destroy
            [](void* obj) noexcept {
                std::launder(reinterpret_cast<F*>(obj))->~F();
            }
        };

        // ── Phase 69: 64-Byte Cache-Line Perfection ─────────────────────────
        static constexpr size_t StorageSize = 64 - sizeof(void*); // 56 bytes
        const VTable* vtable_ = nullptr;
        alignas(8) std::byte storage_[StorageSize];

        void destroy() noexcept {
            if (vtable_) {
                vtable_->destroy(storage_);
                vtable_ = nullptr;
            }
        }

    public:
        InlineTask() noexcept = default;
        InlineTask(std::nullptr_t) noexcept {}

        // ── Phase 69.3: Compile-time check against 56-byte storage ──────────
        template <typename F>
        requires (!std::is_same_v<std::decay_t<F>, InlineTask> && std::is_invocable_r_v<ReturnType, F, Args...>)
        InlineTask(F&& f) {
            using DecayedF = std::decay_t<F>;
            static_assert(sizeof(DecayedF) <= StorageSize,
                "FATAL: Task capture size exceeds SOO buffer (56 bytes). Heap allocation is strictly forbidden.");
            static_assert(alignof(DecayedF) <= 8,
                "FATAL: Alignment requirement exceeds buffer alignment.");
            static_assert(std::is_nothrow_move_constructible_v<DecayedF>,
                "FATAL: Callable must be noexcept move constructible to prevent std::terminate.");

            ::new (storage_) DecayedF(std::forward<F>(f));
            vtable_ = &vtable_for<DecayedF>;
        }

        ~InlineTask() {
            destroy();
        }

        InlineTask(const InlineTask&) = delete;
        InlineTask& operator=(const InlineTask&) = delete;

        // ── Phase 70: Destructive Move Semantics ────────────────────────────
        InlineTask(InlineTask&& other) noexcept {
            if (other.vtable_) {
                other.vtable_->move(other.storage_, storage_);
                vtable_ = other.vtable_;
                // CRITICAL: Destroy the moved-from state to release captured resources!
                other.vtable_->destroy(other.storage_);
                other.vtable_ = nullptr;
            }
        }

        InlineTask& operator=(InlineTask&& other) noexcept {
            if (this != &other) {
                destroy();
                if (other.vtable_) {
                    other.vtable_->move(other.storage_, storage_);
                    vtable_ = other.vtable_;
                    // CRITICAL: Destroy the moved-from state to release captured resources!
                    other.vtable_->destroy(other.storage_);
                    other.vtable_ = nullptr;
                }
            }
            return *this;
        }

        InlineTask& operator=(std::nullptr_t) noexcept {
            destroy();
            return *this;
        }

        ReturnType operator()(Args... args) {
            if (!vtable_) [[unlikely]] {
                #if defined(__has_builtin) && __has_builtin(__builtin_trap)
                    __builtin_trap();
                #else
                    std::abort();
                #endif
            }
            if constexpr (std::is_void_v<ReturnType>) {
                vtable_->invoke(storage_, std::forward<Args>(args)...);
            } else {
                return vtable_->invoke(storage_, std::forward<Args>(args)...);
            }
        }

        explicit operator bool() const noexcept {
            return vtable_ != nullptr;
        }
    };

    // ── Phase 69.4: Class-level L1 cache line assertion ───────────────────
    static_assert(sizeof(InlineTask<void()>) == 64, "FATAL: InlineTask must pack exactly into one 64-byte L1 cache line.");
}

/**
 * Dmitry Vyukov's Bounded MPMC Queue
 * A lock-free, bounded, multi-producer multi-consumer queue.
 */
template<typename T, size_t BufferSize>
class LockFreeMPMCQueue {
private:
    struct Cell {
        std::atomic<size_t> sequence;
        T data;
    };

    static_assert((BufferSize & (BufferSize - 1)) == 0, "BufferSize must be a power of 2");
    
    alignas(NEXUS_CACHE_LINE) Cell buffer_[BufferSize];
    alignas(NEXUS_CACHE_LINE) std::atomic<size_t> enqueue_pos_{0};
    alignas(NEXUS_CACHE_LINE) std::atomic<size_t> dequeue_pos_{0};

public:
    LockFreeMPMCQueue() {
        for (size_t i = 0; i < BufferSize; ++i) {
            buffer_[i].sequence.store(i, std::memory_order_relaxed);
        }
    }

    bool enqueue(T&& data) {
        Cell* cell;
        size_t pos = enqueue_pos_.load(std::memory_order_relaxed);
        while (true) {
            cell = &buffer_[pos & (BufferSize - 1)];
            size_t seq = cell->sequence.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);
            if (diff == 0) {
                if (enqueue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return false; // Queue is full
            } else {
                pos = enqueue_pos_.load(std::memory_order_relaxed);
            }
        }
        cell->data = std::move(data);
        cell->sequence.store(pos + 1, std::memory_order_release);
        return true;
    }

    bool dequeue(T& data) {
        Cell* cell;
        size_t pos = dequeue_pos_.load(std::memory_order_relaxed);
        while (true) {
            cell = &buffer_[pos & (BufferSize - 1)];
            size_t seq = cell->sequence.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);
            if (diff == 0) {
                if (dequeue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return false; // Queue is empty
            } else {
                pos = dequeue_pos_.load(std::memory_order_relaxed);
            }
        }
        data = std::move(cell->data);
        cell->sequence.store(pos + BufferSize, std::memory_order_release);
        return true;
    }
};

class NexusThreadPool {
public:
    using Task = nexus::InlineTask<void()>;

private:
    static constexpr size_t QUEUE_SIZE = 4096;
    alignas(NEXUS_CACHE_LINE) LockFreeMPMCQueue<Task, QUEUE_SIZE> task_queue_;
    std::vector<std::jthread> workers_;
    std::atomic<bool> stop_{false};
    int numa_node_;

public:
    explicit NexusThreadPool(int numa_node, size_t num_threads = 0) : numa_node_(numa_node) {
        std::vector<int> cores = get_numa_node_cores(numa_node);
        if (num_threads == 0) {
            num_threads = cores.size();
            if (num_threads == 0) num_threads = 4;
        }

        workers_.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            int core_id = cores[i % cores.size()];
            workers_.emplace_back([this, core_id](std::stop_token stop_tok) {
                // Pin background worker thread to specific CPU core belonging to this NUMA node
                pin_thread_to_core(pthread_self(), core_id);

                while (!stop_tok.stop_requested() && !stop_.load(std::memory_order_relaxed)) {
                    Task task;
                    if (task_queue_.dequeue(task)) {
                        if (task) {
                            task();
                            task = nullptr; // CRITICAL: Destroys captured unique_ptr immediately!
                        }
                    } else {
                        std::this_thread::yield();
                    }
                }
            });
        }
    }

    ~NexusThreadPool() {
        stop_.store(true, std::memory_order_relaxed);
    }

    bool submit(Task func) {
        return task_queue_.enqueue(std::move(func));
    }

    int get_numa_node() const { return numa_node_; }
};

// Manager maintaining an array of NUMA-isolated thread pools
class NexusNUMAThreadPoolManager {
private:
    std::vector<std::unique_ptr<NexusThreadPool>> pools_;
    std::vector<std::jthread> reactor_threads_;

public:
    NexusNUMAThreadPoolManager() {
        int node_count = get_numa_node_count();
        if (node_count <= 0) node_count = 1;
        pools_.reserve(node_count);
        for (int i = 0; i < node_count; ++i) {
            // Pre-warm background worker threads pinned strictly to this NUMA node
            pools_.push_back(std::make_unique<NexusThreadPool>(i, 4));
        }

        reactor_threads_.reserve(node_count);
        for (int i = 0; i < node_count; ++i) {
            reactor_threads_.emplace_back([i](std::stop_token stop_tok) {
                // Pin reactor thread to the first core of this NUMA node
                std::vector<int> cores = get_numa_node_cores(i);
                if (!cores.empty()) {
                    pin_thread_to_core(pthread_self(), cores[0]);
                }
                nexus::run_io_reactor(i, stop_tok);
            });
        }
    }

    NexusThreadPool& get_pool(int numa_node) {
        if (numa_node < 0 || numa_node >= static_cast<int>(pools_.size())) {
            return *pools_[0];
        }
        return *pools_[numa_node];
    }
};

// Retrieve singleton instance of pool manager
NexusNUMAThreadPoolManager& get_numa_pool_manager();

// Retrieve backward-compatible global thread pool referring to NUMA node 0
NexusThreadPool& get_global_thread_pool();
