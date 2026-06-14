#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <system_error>
#include <future>
#include <stdexcept>
#include <cstring>
#include <mutex>
#include <vector>
#include <atomic>
#include <functional>
#include <condition_variable>
#include <ctime>
#include <algorithm>
#include <chrono>
#include <memory>
#include <filesystem>

#if defined(__linux__)
#include <liburing.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/syscall.h>
#elif defined(__APPLE__)
#include <fcntl.h>
#include <unistd.h>
#include <sys/sysctl.h>
#include <sys/mman.h>
#endif

#include "nexus_os_compat.hpp"

// Forward declarations
class AeonToolBlock;
struct AeonToolBlockHeader;

#if defined(__APPLE__)
constexpr size_t NVME_SECTOR_SIZE = 16384;
#endif

namespace nexus {

struct AsyncIORequest {
    std::string filepath;
    void* buffer;
    size_t size;
    std::promise<void> promise;
    AeonToolBlock* block;
    std::function<void()> on_failure;

    std::atomic<int> chunks_remaining{0};
    std::atomic<bool> has_error{false};
    std::atomic<int> error_code{0};

    int fd = -1;

#if defined(__linux__)
    struct LinuxSubRequest {
        AsyncIORequest* parent = nullptr;
        size_t expected_size = 0;
    };
    std::vector<std::unique_ptr<LinuxSubRequest>> sub_requests;
#endif

    int result_res = 0;
    bool result_success = false;
    uint32_t result_bytes = 0;
};

// LockFreeMPSCQueue definition
class LockFreeMPSCQueue {
private:
    struct Node {
        AsyncIORequest* req;
        std::atomic<Node*> next{nullptr};
    };
    alignas(NEXUS_CACHE_LINE) std::atomic<Node*> head_;
    alignas(NEXUS_CACHE_LINE) std::atomic<Node*> tail_;

public:
    LockFreeMPSCQueue() {
        Node* dummy = new Node{nullptr, nullptr};
        head_.store(dummy, std::memory_order_relaxed);
        tail_.store(dummy, std::memory_order_relaxed);
    }

    ~LockFreeMPSCQueue() {
        Node* curr = head_.load(std::memory_order_relaxed);
        while (curr) {
            Node* next = curr->next.load(std::memory_order_relaxed);
            delete curr;
            curr = next;
        }
    }

    void push(AsyncIORequest* req) {
        Node* new_node = new Node{req, nullptr};
        Node* prev_tail = tail_.exchange(new_node, std::memory_order_acq_rel);
        prev_tail->next.store(new_node, std::memory_order_release);
    }

    AsyncIORequest* pop() {
        Node* head = head_.load(std::memory_order_relaxed);
        Node* next = head->next.load(std::memory_order_acquire);
        if (next == nullptr) {
            return nullptr;
        }
        head_.store(next, std::memory_order_release);
        AsyncIORequest* req = next->req;
        delete head;
        return req;
    }

    bool empty() const {
        Node* head = head_.load(std::memory_order_relaxed);
        return head->next.load(std::memory_order_relaxed) == nullptr;
    }
};

#if defined(__linux__)
inline std::vector<std::unique_ptr<LockFreeMPSCQueue>>& get_numa_mpsc_queues() {
    static std::vector<std::unique_ptr<LockFreeMPSCQueue>> queues = []() {
        int count = get_numa_node_count();
        if (count <= 0) count = 1;
        std::vector<std::unique_ptr<LockFreeMPSCQueue>> v;
        v.reserve(count);
        for (int i = 0; i < count; ++i) {
            v.push_back(std::make_unique<LockFreeMPSCQueue>());
        }
        return v;
    }();
    return queues;
}

inline std::atomic<bool>& is_sqpoll_enabled() {
    static std::atomic<bool> enabled{true};
    return enabled;
}

inline struct io_uring& get_io_ring(int numa_node) {
    static std::vector<std::unique_ptr<struct io_uring>> rings = []() {
        int count = get_numa_node_count();
        if (count <= 0) count = 1;
        std::vector<std::unique_ptr<struct io_uring>> v;
        v.reserve(count);
        bool sqpoll = true;
        for (int i = 0; i < count; ++i) {
            auto ring = std::make_unique<struct io_uring>();
            struct io_uring_params params{};
            std::memset(&params, 0, sizeof(params));
            if (sqpoll) {
                params.flags |= IORING_SETUP_SQPOLL;
                params.sq_thread_idle = 10000;
            }
            int ret = io_uring_queue_init_params(256, ring.get(), &params);
            if (ret < 0) {
                if (sqpoll && ret == -EPERM) {
                    sqpoll = false;
                    is_sqpoll_enabled().store(false, std::memory_order_relaxed);
                    std::memset(&params, 0, sizeof(params));
                    ret = io_uring_queue_init_params(256, ring.get(), &params);
                }
                if (ret < 0) {
                    throw std::runtime_error("Failed to initialize io_uring for NUMA node " + std::to_string(i) + ": error " + std::to_string(ret));
                }
            }
            v.push_back(std::move(ring));
        }
        return v;
    }();
    return *rings[numa_node];
}
#endif



// Forward declaration of submit_async_read_direct
void submit_async_read_direct(const std::string& filepath, void* buffer, size_t size, 
                             std::promise<void>&& promise, AeonToolBlock* block, 
                             std::function<void()> on_failure, int numa_node);

} // namespace nexus

#include "nexus_block_cache.hpp"
#include "nexus_thread_pool.hpp"

namespace nexus {

}

namespace nexus {

inline void process_io_completion(AsyncIORequest* req) {
#if defined(__linux__)
    int res = req->result_res;
    bool success = req->result_success;
    if (!success) {
        int err = (res == 0) ? EIO : (res < 0 ? -res : res);
        req->promise.set_exception(std::make_exception_ptr(std::system_error(err, std::generic_category())));
        if (req->on_failure) req->on_failure();
    } else {
        try {
            apply_numa_hint(req->buffer, req->size);
            
            AeonToolBlockHeader local_header;
            std::memcpy(&local_header, req->buffer, sizeof(AeonToolBlockHeader));
            
            if (std::memcmp(local_header.magic, "ATB1", 4) != 0) {
                throw std::runtime_error("AeonToolBlock: Invalid ATB magic bytes");
            }
            if (local_header.version != 1) {
                throw std::runtime_error("AeonToolBlock: Unsupported ATB version");
            }
            if (local_header.k_tensor_offset + local_header.k_total_bytes > req->size ||
                local_header.v_tensor_offset + local_header.v_total_bytes > req->size) {
                throw std::runtime_error("AeonToolBlock: Corrupted ATB offsets exceed file size");
            }
            req->block->set_header(local_header);
            req->promise.set_value();
        } catch (...) {
            req->promise.set_exception(std::current_exception());
            if (req->on_failure) req->on_failure();
        }
    }
    if (req->fd >= 0) {
        close(req->fd);
    }
    req->sub_requests.clear();
    delete req;

#elif defined(__APPLE__)
    bool success = req->result_success;
    if (!success) {
        int err = req->result_res != 0 ? req->result_res : EIO;
        req->promise.set_exception(std::make_exception_ptr(std::system_error(err, std::generic_category())));
        if (req->on_failure) req->on_failure();
    } else {
        try {
            apply_numa_hint(req->buffer, req->size);
            
            AeonToolBlockHeader local_header;
            std::memcpy(&local_header, req->buffer, sizeof(AeonToolBlockHeader));
            
            if (std::memcmp(local_header.magic, "ATB1", 4) != 0) {
                throw std::runtime_error("AeonToolBlock: Invalid ATB magic bytes");
            }
            if (local_header.version != 1) {
                throw std::runtime_error("AeonToolBlock: Unsupported ATB version");
            }
            if (local_header.k_tensor_offset + local_header.k_total_bytes > req->size ||
                local_header.v_tensor_offset + local_header.v_total_bytes > req->size) {
                throw std::runtime_error("AeonToolBlock: Corrupted ATB offsets exceed file size");
            }
            req->block->set_header(local_header);
            req->promise.set_value();
        } catch (...) {
            req->promise.set_exception(std::current_exception());
            if (req->on_failure) req->on_failure();
        }
    }
    if (req->fd >= 0) {
        close(req->fd);
    }
    delete req;
#endif
}

inline void submit_async_read_direct(const std::string& filepath, void* buffer, size_t size, 
                                     std::promise<void>&& promise, AeonToolBlock* block, 
                                     std::function<void()> on_failure, int numa_node) {
#if defined(__linux__)
    int fd = open(filepath.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) {
        promise.set_exception(std::make_exception_ptr(std::system_error(errno, std::generic_category())));
        if (on_failure) on_failure();
        return;
    }

    AsyncIORequest* req = new AsyncIORequest{
        .filepath = filepath,
        .buffer = buffer,
        .size = size,
        .promise = std::move(promise),
        .block = block,
        .on_failure = on_failure,
        .fd = fd
    };

    constexpr size_t MAX_IO_CHUNK_SIZE = 1024 * 1024 * 1024; // 1GB
    size_t offset = 0;
    while (offset < size) {
        size_t chunk_size = std::min(size - offset, MAX_IO_CHUNK_SIZE);
        auto sub = std::make_unique<AsyncIORequest::LinuxSubRequest>();
        sub->parent = req;
        sub->expected_size = chunk_size;
        req->sub_requests.push_back(std::move(sub));
        offset += chunk_size;
    }

    req->chunks_remaining.store(static_cast<int>(req->sub_requests.size()), std::memory_order_relaxed);

    get_numa_mpsc_queues()[numa_node]->push(req);

#elif defined(__APPLE__)
    int fd = open(filepath.c_str(), O_RDONLY);
    if (fd < 0) {
        promise.set_exception(std::make_exception_ptr(std::system_error(errno, std::generic_category())));
        if (on_failure) on_failure();
        return;
    }

    // Disable OS Page Cache - Apple Silicon zero-copy DMA bypass
    fcntl(fd, F_NOCACHE, 1);

    // Enforce 16KB geometry alignment assertion
    if (reinterpret_cast<uintptr_t>(buffer) % NVME_SECTOR_SIZE != 0) {
        close(fd);
        promise.set_exception(std::make_exception_ptr(std::runtime_error("Direct I/O Error: Buffer not 16KB aligned")));
        if (on_failure) on_failure();
        return;
    }

    AsyncIORequest* req = new AsyncIORequest{
        .filepath = filepath,
        .buffer = buffer,
        .size = size,
        .promise = std::move(promise),
        .block = block,
        .on_failure = on_failure,
        .fd = fd
    };

    #include <thread>
    std::thread([req]() {
        constexpr size_t MAX_IO_CHUNK_SIZE = 1024 * 1024 * 1024; // 1GB
        size_t total_read = 0;
        bool has_err = false;
        int err_code = 0;

        while (total_read < req->size) {
            size_t chunk_size = std::min(req->size - total_read, MAX_IO_CHUNK_SIZE);
            char* dest = static_cast<char*>(req->buffer) + total_read;
            
            size_t chunk_read = 0;
            while (chunk_read < chunk_size) {
                ssize_t bytes_read = ::pread(req->fd, dest + chunk_read, chunk_size - chunk_read, total_read + chunk_read);
                if (bytes_read < 0) {
                    if (errno == EINTR) continue;
                    has_err = true;
                    err_code = errno;
                    break;
                } else if (bytes_read == 0) {
                    has_err = true;
                    err_code = EIO;
                    break;
                }
                chunk_read += bytes_read;
            }
            if (has_err) break;
            total_read += chunk_size;
        }

        req->result_success = !has_err;
        req->result_res = err_code;
        process_io_completion(req);
    }).detach();
#endif
}

inline void run_io_reactor(int numa_node, std::stop_token stop_tok) {
#if defined(__linux__)
    struct io_uring& ring = get_io_ring(numa_node);
    auto& queue = *get_numa_mpsc_queues()[numa_node];

    while (!stop_tok.stop_requested()) {
        bool submitted = false;
        while (AsyncIORequest* req = queue.pop()) {
            size_t offset = 0;
            size_t submitted_chunks = 0;
            bool allocation_failed = false;
            for (size_t i = 0; i < req->sub_requests.size(); ++i) {
                struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
                if (sqe) {
                    auto* sub = req->sub_requests[i].get();
                    void* chunk_buf = static_cast<char*>(req->buffer) + offset;
                    io_uring_prep_read(sqe, req->fd, chunk_buf, sub->expected_size, offset);
                    io_uring_sqe_set_data(sqe, sub);
                    submitted = true;
                    submitted_chunks++;
                    offset += sub->expected_size;
                } else {
                    allocation_failed = true;
                    break;
                }
            }

            if (allocation_failed) {
                req->has_error.store(true, std::memory_order_relaxed);
                req->error_code.store(ENOMEM, std::memory_order_relaxed);
                if (submitted_chunks == 0) {
                    req->promise.set_exception(std::make_exception_ptr(std::runtime_error("io_uring SQE allocation failed")));
                    if (req->on_failure) req->on_failure();
                    close(req->fd);
                    delete req;
                } else {
                    req->chunks_remaining.store(static_cast<int>(submitted_chunks), std::memory_order_relaxed);
                }
            }
        }

        if (submitted) {
            // Syscall-free SQ ring update with full sequential consistency fence to prevent Store-Load reordering deadlock
            std::atomic_store_explicit(reinterpret_cast<std::atomic<unsigned>*>(ring.sq.ktail), ring.sq.sqe_tail, std::memory_order_release);
            std::atomic_thread_fence(std::memory_order_seq_cst);

            bool sqpoll_enabled = is_sqpoll_enabled().load(std::memory_order_relaxed);
            bool need_wakeup = false;
            if (sqpoll_enabled) {
                unsigned flags = std::atomic_load_explicit(reinterpret_cast<std::atomic<unsigned>*>(ring.sq.kflags), std::memory_order_acquire);
                if (flags & IORING_SQ_NEED_WAKEUP) {
                    need_wakeup = true;
                }
            } else {
                need_wakeup = true;
            }

            if (need_wakeup) {
                syscall(__NR_io_uring_enter, ring.ring_fd, 0, 0, IORING_ENTER_SQ_WAKEUP, NULL);
            }
        }

        struct io_uring_cqe* cqe = nullptr;
        unsigned head;
        unsigned count = 0;
        io_uring_for_each_cqe(&ring, head, cqe) {
            AsyncIORequest::LinuxSubRequest* sub = static_cast<AsyncIORequest::LinuxSubRequest*>(io_uring_cqe_get_data(cqe));
            if (sub) {
                AsyncIORequest* req = sub->parent;
                int res = cqe->res;
                if (res < 0) {
                    req->has_error.store(true, std::memory_order_relaxed);
                    req->error_code.store(-res, std::memory_order_relaxed);
                } else if (static_cast<size_t>(res) != sub->expected_size) {
                    req->has_error.store(true, std::memory_order_relaxed);
                    req->error_code.store(EIO, std::memory_order_relaxed);
                }

                int remaining = req->chunks_remaining.fetch_sub(1, std::memory_order_acq_rel);
                if (remaining == 1) {
                    req->result_success = !req->has_error.load(std::memory_order_relaxed);
                    req->result_res = req->error_code.load(std::memory_order_relaxed);
                    get_numa_pool_manager().get_pool(numa_node).submit([req]() {
                        process_io_completion(req);
                    });
                }
            }
            count++;
        }

        if (count > 0) {
            io_uring_cq_advance(&ring, count);
        } else {
            std::this_thread::yield();
        }
    }

#elif defined(__APPLE__)
    while (!stop_tok.stop_requested()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
#endif
}

inline void poll_completions() {}

} // namespace nexus
