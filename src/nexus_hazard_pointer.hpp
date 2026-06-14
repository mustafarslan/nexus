#pragma once
#include <atomic>
#include <mutex>

class HazardPointerManager {
public:
    struct HazardPointerRecord {
        std::atomic<void*> hp{nullptr};
        std::atomic<bool> active{false};
        HazardPointerRecord* next = nullptr;
    };

private:
    std::atomic<HazardPointerRecord*> head_{nullptr};
    std::mutex mutex_;

    HazardPointerRecord* acquire_record() {
        // First try to find an inactive record
        for (HazardPointerRecord* curr = head_.load(std::memory_order_acquire); curr; curr = curr->next) {
            bool expected = false;
            if (curr->active.compare_exchange_strong(expected, true, std::memory_order_release, std::memory_order_relaxed)) {
                return curr;
            }
        }

        // If none is available, allocate a new one under a lock
        std::lock_guard<std::mutex> lock(mutex_);
        // Double check after lock
        for (HazardPointerRecord* curr = head_.load(std::memory_order_relaxed); curr; curr = curr->next) {
            bool expected = false;
            if (curr->active.compare_exchange_strong(expected, true, std::memory_order_release, std::memory_order_relaxed)) {
                return curr;
            }
        }

        auto* new_rec = new HazardPointerRecord();
        new_rec->active.store(true, std::memory_order_relaxed);
        
        HazardPointerRecord* old_head = head_.load(std::memory_order_relaxed);
        new_rec->next = old_head;
        while (!head_.compare_exchange_weak(old_head, new_rec, std::memory_order_release, std::memory_order_relaxed)) {
            new_rec->next = old_head;
        }
        return new_rec;
    }

    void release_record(HazardPointerRecord* rec) {
        if (rec) {
            rec->hp.store(nullptr, std::memory_order_release);
            rec->active.store(false, std::memory_order_release);
        }
    }

public:
    static HazardPointerManager& instance() {
        static HazardPointerManager manager;
        return manager;
    }

    struct ThreadLocalState {
        HazardPointerRecord* record = nullptr;

        ThreadLocalState() {
            record = HazardPointerManager::instance().acquire_record();
        }

        ~ThreadLocalState() {
            HazardPointerManager::instance().release_record(record);
        }
    };

    static HazardPointerRecord* get_thread_record() {
        thread_local ThreadLocalState tl_state;
        return tl_state.record;
    }

    // Protect a pointer
    static void protect(void* ptr) {
        auto* rec = get_thread_record();
        if (rec) {
            rec->hp.store(ptr, std::memory_order_seq_cst);
        }
    }

    // Clear protection
    static void unprotect() {
        auto* rec = get_thread_record();
        if (rec) {
            rec->hp.store(nullptr, std::memory_order_seq_cst);
        }
    }

    // Check if any active thread is protecting this pointer
    bool is_protected(void* ptr) {
        for (HazardPointerRecord* curr = head_.load(std::memory_order_acquire); curr; curr = curr->next) {
            if (curr->active.load(std::memory_order_acquire)) {
                if (curr->hp.load(std::memory_order_acquire) == ptr) {
                    return true;
                }
            }
        }
        return false;
    }

    ~HazardPointerManager() {
        HazardPointerRecord* curr = head_.load(std::memory_order_relaxed);
        while (curr) {
            HazardPointerRecord* next = curr->next;
            delete curr;
            curr = next;
        }
    }
};
