# Project Nexus: Low-Level Engine Internals

This document details the low-level systems mechanics, memory layouts, algorithms, and FFI safety structures implemented inside the Project Nexus core engine.

---

## 1. Token-Level Radix Trie & FSM Logit Masking

To perform constrained decoding at standard generation speeds, Nexus evaluates vocabulary token pathways directly in C++ using a cache-friendly Radix Trie.

### Memory Layout & Alignment
* **RadixNode Struct (`nexus_fsm.hpp`)**:
  ```cpp
  struct alignas(64) RadixNode {
      bool is_leaf = false;
      uint32_t tool_id = 0;
      std::vector<std::pair<llama_token, std::unique_ptr<RadixNode>>> children;
  };
  ```
  * `alignas(64)`: Aligns each node to standard CPU L1/L2 cache lines (64 bytes) to prevent false sharing and minimize memory latency during lookup hops.
  * Node children are stored in a contiguous `std::vector`. Since Trie mutation occurs only once during offline compilation/setup, contiguous vectors maximize spatial cache locality during sequential evaluation scans.

### $\mathcal{O}(1)$ Isolated Logit Masking Algorithm
When the LLM generates a token, `apply_logit_mask` intercepts the raw logits array of size $V$ (vocabulary size) and filters it. To prevent concurrency state bleed across threads recycled by asynchronous Python pools (e.g. FastAPI/Uvicorn), Nexus isolates execution buffers at the instance level:
1. **Scratch Buffer Allocation**: The FSM includes a private vector `scratch_buffer_` allocated as a class member:
   ```cpp
   std::vector<std::pair<llama_token, float>> scratch_buffer_;
   ```
   In the constructor, memory is pre-allocated: `scratch_buffer_.reserve(512);`.
2. **Buffer Original Values**: Original logit floats for valid child tokens are stored in this member:
   ```cpp
   scratch_buffer_.clear();
   scratch_buffer_.reserve(children.size());
   for (const auto& child : children) {
       scratch_buffer_.push_back({ child.first, logits[child.first] });
   }
   ```
   Calling `.clear()` resets the size pointer without freeing the underlying pre-allocated buffer storage. This guarantees zero dynamic allocations on the hot path.
3. **Logits Wipe**: The entire logits span is set to `-INFINITY` using a vectorized memory fill (`std::fill`).
4. **Restore Valid Values**: The saved float values are written back:
   ```cpp
   for (const auto& item : scratch_buffer_) {
       logits[item.first] = item.second;
   }
   ```
   This guarantees the LLM's vocabulary choice is strictly restricted to valid Radix transitions with zero allocation overhead and complete agent thread isolation.

---

## 2. Aeon Tool Block (.atb) Binary Layout

The `.atb` format stores compiled tool schema KV caches. It is structured to align with memory pages and hardware interfaces.

```
+--------------------------------------------------------+
| AeonToolBlockHeader (128 bytes, aligned to 128)        |
+--------------------------------------------------------+
| Contiguous K Tensors Data                              |
+--------------------------------------------------------+
| Padding (to 128-byte boundary)                         |
+--------------------------------------------------------+
| Contiguous V Tensors Data                              |
+--------------------------------------------------------+
```

### Header Alignment
* `alignas(128)` and `#pragma pack(push, 1)` are used to enforce a strict 128-byte header size:
  ```cpp
  struct alignas(128) AeonToolBlockHeader {
      char     magic[4];          // "ATB1"
      uint32_t version;           // 1
      uint64_t model_hash;        // FNV-1a topology hash
      uint32_t n_layer;           // Transformer layers
      ...
      uint8_t  padding[48];       // Pads strictly to 128 bytes
  };
  ```
* Contiguous key and value data sections are aligned to 128-byte boundaries to support optimal SIMD vector registers loading and DMA PCIe transfers.

---

## 3. Semantic Lookaside Buffer (SLB)

The L1 Semantic Lookaside Buffer stores tool descriptions embeddings and accelerates nearest-neighbor searches.

### Struct of Arrays (SoA) Memory Layout
To prevent pointer-chasing and cache pollution, metadata and dense vector features are split:
* `metadata_`: Stores `SLBMeta` arrays (`tool_id`, `scale`, and 5-token `scent`).
* `quantized_vectors_`: Stores INT8 embeddings in a single flat vector. During search, the CPU streams through this memory sequentially, yielding optimal memory prefetching.

### INT8 Symmetric Quantization with Machine Epsilon Bounds
To prevent divide-by-zero errors ($\text{NaN}$ poisoning) when processing zero-vectors or padded embedding layers, Nexus enforces a machine epsilon lower bound of $1e-8f$ on the scaling factor calculations:
* **Scale Factor**:
  ```cpp
  float max_val = std::max(actual_max, 1e-8f);
  float sq = max_val / 127.0f;
  ```
* **Quantized Value**: $q_i = \text{clamp}(\text{round}(\frac{v_i}{s_q}), -127, 127)$

The dequantized dot-product of query $Q$ and tool $T$ is computed as:
$$\text{Score} = \left(\sum_{i=1}^{D} q_{Q, i} \cdot q_{T, i}\right) \cdot s_Q \cdot s_T$$

### SIMD Acceleration & Horizontal Reduction
The integer dot-product is optimized for native CPU architectures and includes a **Scalar Remainder Loop** to process tail elements when the embedding dimension $D$ is not a perfect multiple of the SIMD register capacity (16 or 32 elements). This prevents unaligned memory accesses and segmentation faults. 

To avoid store-to-load forwarding stalls, the SIMD vector registers are collapsed down to a scalar using register-only horizontal reductions *before* the scalar remainder loop adds tail elements:

#### ARM NEON (Apple Silicon)
On ARM64, the kernel uses the dot-product extension `vdotq_s32` (or widen/multiply fallback) and collapses lanes using `vaddvq_s32`:
```cpp
for (; i + 16 <= dim; i += 16) {
    int8x16_t va = vld1q_s8(a + i);
    int8x16_t vb = vld1q_s8(b + i);
    accum = vdotq_s32(accum, va, vb);
}
// Horizontal Reduction
int32_t total = vaddvq_s32(accum);
// Scalar Remainder Loop
for (; i < dim; ++i) {
    total += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
}
```

#### Intel AVX2
On x86 architectures, AVX2 performs widening and multiply-accumulate operations, collapsing the registers via `_mm256_extracti128_si256` and `_mm_shuffle_epi32` shuffles:
```cpp
for (; i + 32 <= dim; i += 32) {
    __m256i va_lo = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i)));
    __m256i vb_lo = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i)));
    __m256i prod_lo = _mm256_madd_epi16(va_lo, vb_lo);
    ...
    accum = _mm256_add_epi32(accum, _mm256_add_epi32(prod_lo, prod_hi));
}
// Horizontal Collapse Sequence (Register-only)
__m128i lo = _mm256_castsi256_si128(accum);
__m128i hi = _mm256_extracti128_si256(accum, 1);
__m128i sum128 = _mm_add_epi32(lo, hi);
sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(0, 1, 0, 1)));
int32_t total = _mm_cvtsi128_si32(sum128);

// Scalar Remainder Loop
for (; i < dim; ++i) {
    total += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
}
```

### Stack-Allocated Top-K Insertion Sort
To maintain the top $K$ results (where $K \le 16$) without heap allocations or standard priority queues, the search path uses a stack-allocated tracker:
1. Initialize a fixed stack array of size $16$: `TempResult best_results[16]`.
2. For each evaluated tool, find its insertion index $j$ where $\text{score} > \text{best\_results}[j].\text{score}$.
3. Shift lower elements to the right and insert the new candidate.

---

## 4. Stride-Aware 2D KV Cache Splicing & Cell Synchronization

To hot-swap tool schemas into the active KV cache without the memory overhead and compute latency of copying the entire cache back and forth between GPU and CPU memory, the `NexusKVSplicer` performs targeted, stride-aware 2D blits.

### Stride-Aware Blitting
Instead of assuming a flat contiguous mapping, the splicer queries the `ggml_tensor` byte strides `nb` dynamically at runtime. For a given position `pos`, KV head `ih`, and head dimension `id`:
* **K Cache (Untransposed layout `[pos, ih, id]`)**:
  The byte offset in the destination tensor is:
  $$\text{Byte Offset} = \text{id} \cdot nb[0] + \text{ih} \cdot nb[1] + \text{pos} \cdot nb[2]$$
* **V Cache (Transposed layout `[ih, id, pos]`)**:
  If `v_trans` is true, the sequence position `pos` becomes the innermost dimension. The byte offset is:
  $$\text{Byte Offset} = \text{pos} \cdot nb[0] + \text{id} \cdot nb[1] + \text{ih} \cdot nb[2]$$

### Local Segment Buffering & Uploads
To avoid multiple round-trip overheads, the splicer packs the tool schema cache segment of length $L = \text{seq\_len}$ into a compact host memory buffer, then performs targeted uploads:
1. **Contiguous Optimization**: If the sequence dimension stride matches the expected contiguous size ($nb[2] = N_{\text{heads}} \cdot D_{\text{head}} \cdot \text{sizeof(uint16\_t)}$), the entire spliced segment is uploaded to the backend in a single transaction starting at offset $\text{base\_pos} \cdot nb[2]$:
   ```cpp
   ggml_backend_tensor_set(k_tensor, host_k_spliced.data(), base_pos * nb[2], seq_len * nb[2]);
   ```
2. **Strided Fallback**: If non-contiguous padding exists, it falls back to position-by-position uploads.
3. **Transposed V Splicing**: For transposed V caches, the splicer performs row-by-row uploads. For each row (head $\text{ih}$ and dimension $\text{id}$), a compact buffer of size $\text{seq\_len} \cdot nb[0]$ is uploaded directly to the destination offset:
   ```cpp
   ggml_backend_tensor_set(v_tensor, row_buf.data(), dst_offset, seq_len * nb[0]);
   ```

### Active Cell Metadata Synchronization
After splicing the raw tensors on the backend, the FSM state must be synchronized inside the `llama.cpp` runtime cell metadata block. The splicer maps sequence IDs and position offsets by calling:
```cpp
llama_kv_cache_set_cell(ctx, pos, static_cast<llama_pos>(pos), target_seq);
```
This updates `llama_kv_cell` parameters (clearing old paths, registering the new `target_seq` ID, and setting the active sequence coordinates) and recalculates the internal `used` cells count. This prevents `llama.cpp` from garbage-collecting or overwriting the hot-swapped schema.

---

## 5. Redis-Style $\mathcal{O}(1)$ Amortized Probabilistic Cache Eviction

Linear map scanning under global write locks causes severe queue delays under high concurrent tenant workloads. Nexus addresses this via probabilistic eviction.

```
       HASH MAP SLOTS
+------------------------------------+
| [Bucket 0] -> [CacheEntry 1]       |
| [Bucket 1] -> Empty                |
| [Bucket 2] -> [CacheEntry 2]  <-------- [Sample 1] (access_tick: 4802)
| [Bucket 3] -> [CacheEntry 3]  <-------- [Sample 2] (access_tick: 1209) ---> EVICTED
| [Bucket 4] -> [CacheEntry 4]  <-------- [Sample 3] (access_tick: 9811)
| [Bucket 5] -> [CacheEntry 5]  <-------- [Sample 4] (access_tick: 3410)
| [Bucket 6] -> [CacheEntry 6]  <-------- [Sample 5] (access_tick: 7602)
+------------------------------------+
  1. Pick K=5 random buckets.
  2. Compare access ticks among active entries.
  3. Evict lowest access tick. Complexity = O(K) constant time.
```

### Algorithm & Complexity Analysis
1. **Sampling Phase**: Under a `std::unique_lock`, Nexus queries the bucket count of `cache_map_` and selects a starting bucket index using `std::mt19937`:
   ```cpp
   size_t start_bucket = rng_() % bucket_count;
   ```
2. **Evaluation Phase**: The eviction loop scans downstream buckets sequentially until $K = 5$ valid ready blocks are sampled.
3. **Tick Comparison**: Access metadata is fetched:
   ```cpp
   uint64_t tick = entry->last_access_tick.load(std::memory_order_relaxed);
   ```
4. **Eviction Execution**: The item with the minimum access tick is popped from `cache_map_`, and `current_pinned_bytes_` is updated. 

This limits the eviction lookup phase to $K$ steps. Since $K \ll N$, complexity is capped at $\mathcal{O}(1)$ constant time, keeping lock duration independent of cache size.

---

## 6. Exception-Safe Poison Cleanup Lifecycle

Failed file reads, unaligned page sizes, or failed host memory registrations must be handled without leaving corrupted markers inside the global cache.

```
Main Thread                               Background Residency Thread
get_or_load(key)
    |
  [Lock] -> Check cache
    |-----> (Miss) -> Insert Promise
    |                 & Future
  [Unlock]
    |
  Construct AeonToolBlock ------(Async)---->  Apply NUMA Hint (SYS_mbind)
    |                                                |
    | (Sync Catch Throw)                      Touch Pages (Page fault)
    |-- Erase Key                                    |
    |-- Set Exception                         llama_register_host_memory
    |                                                |
    |                                         (Catch Async Throw)
    |                                         -- Fulfill Promise w/ Exception
    |                                         -- Fire on_failure Callback
    |                                            |
    |                                            +--> [Lock] -> Erase Key -> [Unlock]
```

* **Synchronous Safe Cleanup**: If the `AeonToolBlock` constructor throws during initialization on the orchestrator thread (e.g., failed `mmap` or unaligned header files), the exception is caught, `promise->set_exception()` is populated, and the cache map entry is immediately erased under a local write-lock before re-throwing the exception.
* **Asynchronous Safe Cleanup**: If the exception occurs inside the background worker thread (e.g., page touching or device pinning failures), the worker catches the throw, runs the registered `on_failure` lambda, and re-throws:
  ```cpp
  [this, atb_filepath]() {
      std::unique_lock<std::shared_mutex> cleanup_lock(mutex_);
      cache_map_.erase(atb_filepath);
  }
  ```
  This guarantees that regardless of which stage of the hardware mapping fails, the poisoned cache future is removed, allowing subsequent threads to retry cleanly.

---

## 7. Linux NUMA Memory Alignment System Calls

To maximize PCIe DMA bandwidth on Linux NUMA servers, physical pages must reside on nodes local to the target GPU socket.

### Address Page Alignment Math
Linux system calls like `mbind` require memory boundaries to align strictly to the system page size (typically 4096 bytes on x86_64, or 16384 bytes on ARM64). Nexus calculates these boundaries using bitwise operations:
```cpp
uintptr_t page_size = get_page_size(); // Wrapping sysconf(_SC_PAGESIZE)
uintptr_t start = reinterpret_cast<uintptr_t>(addr);
uintptr_t end = start + size;
uintptr_t aligned_start = start & ~(page_size - 1);
uintptr_t aligned_end = (end + page_size - 1) & ~(page_size - 1);
```

### Direct System Call Invocation
To avoid linking dependencies on local server `libnuma.so` binaries, Nexus issues direct system call interrupts:
```cpp
#include <sys/syscall.h>
#include <unistd.h>

unsigned long nodemask = 3; // Bind to CPU Node 0 and Node 1 (binary mask 0011)
// SYS_mbind (syscall 237 on x86_64)
long res = syscall(SYS_mbind, 
                   reinterpret_cast<void*>(aligned_start), 
                   aligned_end - aligned_start, 
                   3,          // MPOL_INTERLEAVE mode
                   &nodemask, 
                   2,          // maxnode
                   0           // flags
           );
```
This forces the OS page table manager to distribute physical memory allocations evenly across local CPU nodes.

---

## 8. Compiler Backend FFI Bridges

To adjust telemetry logging styles dynamically based on the underlying compute device, Nexus hooks directly into internal `llama.cpp` runtime libraries:

```cpp
extern "C" {
    bool llama_context_is_metal(struct llama_context * ctx);
    bool llama_context_is_cuda(struct llama_context * ctx);
}
```

These functions inspect active compute backends registered inside `llama_context` and return boolean values, allowing PyCapsule wrappers and benchmark binaries to detect topologies without hardcoded system parameters.
