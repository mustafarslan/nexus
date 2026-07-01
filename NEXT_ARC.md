# NEXT_ARC.md — Project Nexus System Topology & Cache-Injection Failure Audit

**Scope.** Forensic trace of the tool-routing engine (SLB) and the KV-cache
sidecar (`.atb` compile → splice → recompute) as actually implemented in this
tree. Every claim below is anchored to `file:line` with the exact identifier or
constant. The central question — *why does cache injection break when the live
conversation boundary `n_past` scales past 256 tokens or meets a non-linear
(transposed / grouped-query) attention layout* — is answered structurally, not
abstractly. The host runtime is the vendored `external/llama.cpp` submodule;
the sidecar is C++ (`src/nexus_*.{cpp,hpp}`) bridged to Python
(`src/nexus_agent.py`) via nanobind (`src/bindings.cpp`).

Bounds notation: the splice guard enforces `n_past ≤ M`, `M = 256`. A `.atb`
block is compiled over the absolute index window `[p_b, p_b + L)` where
`p_b = header->base_pos` and `L = header->seq_len`.

---

## Verified vs. Brief — provenance table

| # | Brief assertion | Code status | Anchor |
|---|---|---|---|
| 1 | `n_past ≤ 256` splice guard | **Confirmed** | `nexus_orchestrator.cpp:69`, `nexus_agent.py:182` |
| 2 | Path B linear-prefill fallback on `n_past > 256` | **Confirmed (split host)** — orchestrator declines (`return 0`); the linear prefill itself runs in the Python gateway | `nexus_orchestrator.cpp:69-91`, `nexus_agent.py:691` |
| 3 | `QuantizedBitmapAllocator` / buddy sub-structures | **Confirmed** | `nexus_bitmap_allocator.hpp:78` |
| 4 | Pointer provenance via `std::launder` across `void*` | **Confirmed, but located precisely** — `std::launder` is **not** in the allocator/splice path (that path uses offset arithmetic + alignment validation + placement-new); it **is** used for type-erased callable storage in the thread pool | `nexus_thread_pool.hpp:34,36,41,45`; `nexus_bitmap_allocator.hpp:123-130, 203-204`; `bindings.cpp:84` |
| 5 | `.atb` parsed & physically injected; sequential K/V layout | **Confirmed** | `aeon_tool_block.hpp:13-58`, `atb_io.py:19-51` |
| 6 | Anchored Turn-1 splice offset/slice math | **Confirmed** | `nexus_kv_splicer.cpp:250-258, 278` |
| 7 | Tail-suffix recompute + denominator calibration | **Confirmed** | `nexus_orchestrator.cpp:490-501`, `nexus_rope_math.cpp:139-205` |
| 8 | Frozen `[0,M]` positions vs runtime `n_past` contradiction | **Confirmed** | `aeon_tool_block.hpp:23`, `nexus_kv_splicer.cpp:278` |
| 9 | Deep-turn recovery experiments | **Confirmed** | `test/nexus_recompute.py:10-152` |
| 10 | Re-prefill compute amplification penalty (constant) | **Partial** — no single hardcoded amplification ratio; the penalty is a *formula* (`exp(-0.02·depth_ratio)`) plus an empirical bench, not a literal | `nexus_kv_splicer.cpp:517`, `bench_dynamic_context.py:63-65` |
| 11 | Soft-capping forces `v_trans = true` | **Confirmed (transitive)** — not a direct flag; soft-cap → FlashAttn force-off → `v_trans=true` → splicer throws | `llama.cpp:19032-19033 → 3026 → nexus_kv_splicer.cpp:243-244` |
| 12 | Fixed-stride INT8 SIMD router (`route_and_splice`) breaks on GQA/MQA | **Partial** — fault is *detected* (layer-0 stride validator throws), not silently corrupting | `nexus_kv_splicer.cpp:293-295, 310-312` |

---

## 1. Production Path Optimization Matrix

### 1.1 The `n_past ≤ 256` splice guard (the gate)

The cache-injection entry point is `NexusOrchestrator::route_and_splice(...)`
(`src/nexus_orchestrator.cpp:51`). The gate is a single conditional:

```cpp
// src/nexus_orchestrator.cpp:69
if (n_past > max_splice_pos_) {        // M = 256
```

Gating variable `max_splice_pos_` is declared `uint32_t`
(`src/nexus_orchestrator.hpp:48`), defaulted to `256` at three layers:

- C++ ctor default — `src/nexus_orchestrator.hpp:88` `uint32_t max_splice_pos = 256`
- Python constant — `src/nexus_agent.py:182` `MAX_SPLICE_POS = 256` (ctor arg `:211`, stored `:226`)
- FFI default — `src/bindings.cpp:433` `nb::arg("max_splice_pos") = 256`

It is runtime-mutable via `set_max_splice_pos(uint32_t)`
(`src/nexus_orchestrator.hpp:98`). The Python anchor budget is computed against
it with a safety margin: `budget = self.max_splice_pos - SAFETY - len(sys_toks)`
(`src/nexus_agent.py:499`).

**Path A (pass-through, `n_past ≤ M`).** When the gate is not taken, the splice
proceeds. Two injection sites, both calling the splicer:

- `src/nexus_orchestrator.cpp:206` — speculative block:
  `NexusKVSplicer::inject_tool_page(ctx_, speculative_block, n_past, seq_id)`
- `src/nexus_orchestrator.cpp:467` — resolved/correct block:
  `NexusKVSplicer::inject_tool_page(ctx_, correct_block, n_past, seq_id)`

### 1.2 Path B fallback (`n_past > M`)

`src/nexus_orchestrator.cpp:69-91`. The block:

1. increments `splice_guard_fallback_count_` (`hpp:49`, `cpp:79`),
2. performs a **read-only** L0 probe only —
   `seq_warm_cache_.probe_prefix(prefix_tokens)` (`cpp:82`) — explicitly *not*
   consumed this phase (the F0 deep-path instrumentation comment block,
   `hpp:51-61`),
3. increments `deep_path_text_fallback_` (`hpp:61`, `cpp:90`),
4. `return 0`.

The orchestrator therefore **declines** rather than performing the linear
prefill itself. The standard linear prefill pass is invoked one layer up, in the
Python gateway: `src/nexus_agent.py:691` — the `B3pc-style text prefill when
splice guard blocks ATB inject (P > max_splice_pos)`. This split is the precise
reason Path B costs a full re-prefill of the preceding context (see §3.4): the
fast C++ KV memcpy is bypassed and the model re-reads raw text schema tokens.

### 1.3 `QuantizedBitmapAllocator` memory layout

`src/nexus_bitmap_allocator.hpp:78` — `class QuantizedBitmapAllocator`. It is a
**buddy allocator over 2 MB blocks**, not a per-byte heap.

Arena fields:
- `void* base_ptr_` — `:87`
- `size_t total_size_` — `:88`
- `size_t num_blocks_` — `:89`
- buddy free-lists — `std::vector<BuddyNode> nodes_` / `std::vector<size_t> free_heads_` (`:91-92`)

Hardcoded geometry:
- `#define NEXUS_PAGE_ALIGNMENT (2 * 1024 * 1024)` — 2 MB HugeTLB block (`:12-13`)
- `#define NEXUS_CACHE_LINE 128` (`:16-17`)
- block count: `num_blocks_ = total_size / NEXUS_PAGE_ALIGNMENT` (`:134`)
- free-list init: `nodes_.resize(num_blocks_)`, `free_heads_.assign(max_order_ + 1, -1)` (`:144-145`)
- order ceiling: `while ((1ULL << max_order_) < num_blocks_)` (`:140`)

Block-index ↔ host-address mapping is pure pointer arithmetic over the 2 MB
stride:

```cpp
// allocate — src/nexus_bitmap_allocator.hpp:197
return static_cast<char*>(base_ptr_) + i * NEXUS_PAGE_ALIGNMENT;
// free (reverse) — :203-204
size_t offset = static_cast<char*>(ptr) - static_cast<char*>(base_ptr_);
size_t i = offset / NEXUS_PAGE_ALIGNMENT;
```

Buddy merge XORs the **block index** (not the raw pointer):
`size_t buddy = i ^ (1ULL << order)` (`:215`); split path uses
`size_t buddy = i + (1ULL << found_order)` (`:191`).

**Hardware mapping.** Arena bases are tracked per NUMA node in the block cache:
`std::vector<void*> arena_bases_` (`src/nexus_block_cache.hpp:179`), each arena
≥ 256 MB (`min_arena = 256ULL * 1024 * 1024`, `src/nexus_block_cache.cpp:166`),
registered to the host runtime and freed as huge arenas
(`llama_unregister_host_memory(...)` / `free_huge_arena(...)`,
`nexus_block_cache.cpp:188-189`). NUMA-node lookup is bounds-checked against
`arena_bases_.size()` (`nexus_block_cache.cpp:272-275`).

### 1.4 Pointer provenance across `void*` boundaries

The brief anticipates `std::launder` in the cache-injection path. **It is absent
from the allocator and the splicer** — provenance there is maintained by three
explicit mechanisms instead. (`std::launder` *does* appear in the codebase, but
only in `src/nexus_thread_pool.hpp:34,36,41,45`, where it legitimizes accessing a
type-erased callable `F*` punned out of the task's inline `void*` storage via a
per-type `VTable` — invoke `:34/:36`, move-construct `:41`, destroy `:45`. That is
the thread-pool's small-function optimization, not KV memory virtualization.)
The allocator/splicer provenance mechanisms are:

1. **Alignment-validating ctor** — `nexus_bitmap_allocator.hpp:123-130`:
   `if (reinterpret_cast<uintptr_t>(base_ptr) % NEXUS_PAGE_ALIGNMENT != 0) throw`,
   plus `total_size % NEXUS_PAGE_ALIGNMENT != 0` (`:131`). Provenance is *proven*
   by the arena-base invariant, not laundered.
2. **Offset round-tripping** — a freed pointer is mapped back to its owning
   block index by subtraction from `base_ptr_` (`:203-204`); the type-erased
   `void*` never loses its arena identity because it is only ever reconstituted
   relative to a validated base.
3. **Placement-new across the FFI** —
   `new (self) nexus::QuantizedBitmapAllocator(reinterpret_cast<void*>(base_addr), size)`
   (`bindings.cpp:84`); the matching `self.free(reinterpret_cast<void*>(ptr_addr), size)`
   (`bindings.cpp:90`).

> **Audit note.** Because provenance rests on the *2 MB-alignment invariant*
> rather than `std::launder`, any path that hands the allocator a base that is
> aligned but *aliased* (e.g. a transposed-V re-view, §4) is not caught here — it
> is caught downstream by the splicer's stride validator (§4.3).

---

## 2. Physical Tensor Layout & Splicing Mechanics

### 2.1 `.atb` (Aeon Tool Block) on-disk / in-memory layout

`src/aeon_tool_block.hpp:13-58` — `struct alignas(128) AeonToolBlockHeader`,
`#pragma pack(push,1)`, asserted `sizeof == 128` (`:40`) with a full set of
`offsetof` static-asserts (`:41-58`). Little-endian only (`:6`). Field map (byte
offsets are compiler-verified):

| Field | Offset | Meaning |
|---|---|---|
| `char magic[4]` (`"ATB1"`) | 0 | `:14` |
| `uint32_t version` | 4 | `:15` |
| `uint64_t model_hash` | 8 | FNV-1a of topology, `:17` |
| `uint32_t n_layer` | 16 | `:18` |
| `uint32_t n_head_kv` | 20 | **GQA-aware** KV head count, `:19` |
| `uint32_t d_head` | 24 | `:20` |
| `uint32_t seq_len` (= `L`) | 28 | `:22` |
| `uint32_t base_pos` (= `p_b`) | 32 | **frozen RoPE start**, `:23` |
| `float rope_freq_base` | 36 | `:24` |
| `float rope_freq_scale` | 40 | `:25` |
| `uint32_t rope_scaling_type` | 44 | 0=none,1=linear,2=YaRN, `:26` |
| `uint32_t ggml_type_k/v` | 48/52 | **must be F16** (value 1), `:28-29` |
| `uint64_t k_tensor_offset / v_tensor_offset` | 56/64 | `:31-32` |
| `uint64_t k_total_bytes / v_total_bytes` | 72/80 | `= L·n_layer·n_head_kv·d_head·2`, `:33-34` |

**Sequential physical layout:** `[ 128 B header ][ contiguous K ][ contiguous V ]`.
Confirmed by the reader, which slices on the stored offsets:

```python
# scripts/atb_io.py:50-51
k_raw = raw[hdr["k_tensor_offset"] : hdr["k_tensor_offset"] + hdr["k_total_bytes"]]
v_raw = raw[hdr["v_tensor_offset"] : hdr["v_tensor_offset"] + hdr["v_total_bytes"]]
```

`base_pos` is unpacked from byte 32 (`atb_io.py:20`). The compiler that emits the
block is `src/nexus_kv_compiler.cpp`. K/V are F16 (`sizeof(uint16_t)`) row tensors
throughout the splicer (`nexus_kv_splicer.cpp:93`).

### 2.2 Anchored Turn-1 splice — index offset & slice arithmetic

`NexusKVSplicer::inject_tool_page_raw(...)` (`hpp:15`). The destination physical
cell `p` is derived from the runtime KV cursor with ring-buffer wrap:

```cpp
// src/nexus_kv_splicer.cpp:250-258
int32_t p_prev = llama_kv_cache_get_physical_pos(ctx, n_past - 1);
//   if found:  p = (p_prev + 1) % kv_size;          // :252  (wrap)
//   else:      p = llama_kv_cache_get_physical_pos(ctx, n_past);  // :256
//   fallback:  p = static_cast<int32_t>(n_past % kv_size);        // :258
```

The K source is mapped read-only and memcpy'd row-major into the cache at cell
`p`. Row stride is fixed:

```cpp
// src/nexus_kv_splicer.cpp:93
size_t k_size_row = task->n_head_kv * task->d_head * sizeof(uint16_t);
size_t v_size_row = k_size_row;                                   // :94
// contiguous (no wrap) copy — :107
std::memcpy(dst_k + task->p * k_size_row, src_k_layer, task->seq_len * k_size_row);
// ring wrap split — :111-112
std::memcpy(dst_k + task->p * k_size_row, src_k_layer, first_part * k_size_row);
std::memcpy(dst_k, src_k_layer + first_part * task->n_head_kv * task->d_head, second_part * k_size_row);
```

The async/GPU path mirrors this with `llama_tensor_set_async(...)` at the same
`p * k_size_row` byte offsets (`:462-467`).

### 2.3 Tail-suffix recomputation & denominator calibration

After a stitch, the seam tokens carry stale attention statistics. Two coupled
mechanisms repair them.

**(a) Selective suffix recompute window** — `src/nexus_orchestrator.cpp:490-501`:

```cpp
const bool fused_recompute = schema_tok_it != tool_schema_tokens_.end() && recompute_pct_ > 0.0f; // :490
uint32_t suffix_start = schema_len;                                                                 // :492
const uint32_t n_sel = std::max(1u, (uint32_t)std::ceil((double)schema_len * (double)recompute_pct_ / 100.0)); // :498-499
suffix_start = (schema_len > n_sel) ? (schema_len - n_sel) : 0u;                                     // :500
```

i.e. the trailing `n_sel = ⌈schema_len · recompute_pct/100⌉` tokens of the schema
are invalidated and re-decoded; `invalidate_sequence(ctx_, seq_id, …)` is the
eraser (`nexus_kv_splicer.hpp:11`, call sites `nexus_orchestrator.cpp:205,211,407`).

**(b) Softmax-denominator / attention-score calibration.** Because the spliced
rows were scored at compile-time positions, the runtime softmax denominator is
miscalibrated. The fix is a depth-aware V-scale + KQ bias:

```cpp
// src/nexus_kv_splicer.cpp:512-525  (post-splice)
//   depth_ratio = log(n_past / base_pos)            :513-514
//   v_scales[is] = exp(-0.02 * depth_ratio)         :517
//   llama_kv_cache_apply_attention_bias(ctx, -1, n_past, n_past + seq_len, v_scales.data(), …)  :519-524
```

The richer model lives in `src/nexus_rope_math.cpp:139-205`
(`apply_denominator_calibration_v2`): `depth_ratio = log(target_p / compiled_p)`
(`:177-179`), `global_bias = -deficit·0.15 - 0.15·depth_ratio·n_past_scale`
(`:182`), per-token `out_bias[i] = global_bias + rel·(-0.05)` (`:188`), and the
per-token V multiplier `v_adj = -0.02·depth_ratio·(1 - local/splice_mean)` →
`out_v_scales[i] = exp(v_adj)` (`:191-192`). These `-0.15` (KQ) and `-0.02` (V)
coefficients are the calibration constants of the cache-stitch pass.

---

## 3. Positional Inversion & RoPE Drift Mechanics

### 3.1 The frozen-position contradiction (formal)

A `.atb` block bakes RoPE rotations assuming absolute positions `[p_b, p_b + L)`,
`p_b = header->base_pos` (typically 0). At runtime the same K rows must occupy
absolute positions `[n_past, n_past + L)`. For RoPE, the key at position `p` for
dimension pair `i` carries the rotation operator `R(θ_i · p)`. Splicing a block
compiled at `p_b` into slot `n_past` therefore demands a corrective rotation by

```
Δ = n_past − p_b           (delta_pos)
R_needed = R(θ_i · Δ)      applied per dimension-pair i
```

The offset is computed exactly:

```cpp
// src/nexus_kv_splicer.cpp:278
int32_t delta_pos = static_cast<int32_t>(n_past) - static_cast<int32_t>(header->base_pos);
```

Because `θ_i = freq_base^(−2⌊i/2⌋ / d_head)` spans many decades across `i`, the
phase error `θ_i · Δ` is **non-uniform across dimensions** and grows with `Δ`.
For `n_past > M = 256` the accumulated Δθ across the high-frequency pairs exceeds
what a single relative shift restores cleanly — which is exactly why the guard at
`nexus_orchestrator.cpp:69` refuses the splice past `M` (the comment there names
it the "RoPE Δθ" decline).

### 3.2 Re-anchor math (the compensation that *is* implemented for `Δ ≠ 0`, `n_past ≤ M`)

`src/nexus_rope_math.cpp`:
- `get_rope_frequency(...)` — θ formula `theta_extrap = pow(freq_base, -2.0f·(i0/2)/d_head)` (`:15`); YaRN interp/extrap ramp mix (`:17-31`).
- `apply_relative_rope_shift(...)` (`:40-82`) — precomputes `cos/sin(Δ·θ_i)` and applies the 2×2 rotation `r0 = x0·cos − x1·sin; r1 = x0·sin + x1·cos`.
- `apply_absolute_rope_reanchor(...)` (`:84-137`) — computes `uniform_delta = target_base_pos − compiled_base_pos` (`:101`) and dispatches to the relative shift (`:103-104`), with a per-token YaRN branch (`:118+`).

Dispatch from the splicer, gated on UMA + non-zero delta + CPU:

```cpp
// src/nexus_kv_splicer.cpp:342
if (is_uma && delta_pos != 0 && !use_gpu_rope) { … apply_absolute_rope_reanchor(…) …}  // worker :167-186
```

The discrete-GPU / Metal path instead calls the bridge
`llama_kv_cache_rope_reanchor_gpu(...)` (decl `:33`, invoked under the
`!is_uma || use_gpu_rope` branch). Critically, the reanchor rewrites **K only**
(`dst_k + p*k_size_row`, `:168`) — V is not rotated, consistent with RoPE
applying to Q·K.

### 3.3 Experimental deep-turn recovery

`test/nexus_recompute.py` is the shared harness for clawing back fidelity when
the bare splice drifts:
- `SUFFIX_SELECTORS` (`:10`): `tail`, `hkvd_suffix_start`, `oracle_suffix_start`,
  `boundary_window`, `fused_hkvd`.
- `RecomputePlan` dataclass (`:20-29`): `suffix_start_idx`,
  `actual_recompute_tokens`, `actual_recompute_pct`, `selector`.
- `plan_suffix_recompute(...)` (`:54-108`) selects tokens by **layer-1 K-deviation**
  (`hkvd_suffix_start`, top-n most-deviating) or cross-layer oracle deviation, or
  a CacheBlend-style head/tail `boundary_window`.
- `recompute_selected(...)` (`:138-152`): maps schema-relative index to absolute
  KV position `abs_start = p_start + min_idx` (`:148`), `invalidate_sequence(...)`
  (`:150`), then `decode_tokens(ctx, schema_tokens[min_idx:], abs_start, 0)` (`:151`).

Position-sensitivity is measured deliberately *across* the guard boundary:
- `test/bench_g3_position.py`: `p_starts = [256, 1024, 2048]` (`:74`), reporting KL
  divergence and top-1 vs a no-splice reference (`bare_splice_logits` vs
  `tail_r5_logits`).
- `test/bench_phaseB_selective_recompute.py:288-301`: sweep `p_starts = [256, 1024]`,
  `suffix_pct = [0, 5, 10, 100]`, `selector ∈ {tail, hkvd_suffix_start,
  oracle_suffix_start}`; deviations computed by `compute_deviations_with_preceding`
  (`:132`) as per-token L2 of K difference across layers.

### 3.4 Compute amplification under high multi-turn depth

The penalty has two distinct expressions; the brief's "amplification constant"
does **not** exist as a literal — record this honestly:

1. **In-band damping (when a splice *is* allowed).** The V-scale dampening
   `exp(-0.02·depth_ratio)` (`nexus_kv_splicer.cpp:517`,
   `nexus_rope_math.cpp:191`) and KQ bias `-0.15·depth_ratio`
   (`nexus_rope_math.cpp:182`). With `n_past = 1024`, `p_b ≈ 1`,
   `depth_ratio = ln(1024) ≈ 6.9`, so per-token V scale ≈ `exp(-0.138) ≈ 0.87`.
2. **Out-of-band re-prefill (Path B, `n_past > M`).** The splice is abandoned and
   the gateway re-prefills the entire preceding context
   (`nexus_agent.py:691`). The amplification is therefore structural —
   `O(preceding_len)` token recomputes against an `O(L)` schema — i.e. roughly
   `preceding_len / L`. It is **measured, not hardcoded**, in
   `test/bench_dynamic_context.py`: the prefix-cache byte estimator
   `2·seq_len·n_layers·n_head_kv·d_head` (`:63-65`) and the TTFT-vs-`p_start`
   sweep (`p_start = len(preceding)`), whose stated hypothesis is that splice
   cost stays `O(1)` while prefix-cache hit-rate collapses as concurrent session
   count grows.

---

## 4. Attention Logit Soft-Capping & Transposition Barriers

### 4.1 Where model-layout metadata is parsed at init

`external/llama.cpp/src/llama.cpp`:
- defaults: `f_attn_logit_softcapping = 50.0f` (`:2191`),
  `f_final_logit_softcapping = 30.0f` (`:2192`), `bool attn_soft_cap = false` (`:2212`).
- Gemma2 GGUF keys (`%s.attn_logit_softcapping`, `%s.final_logit_softcapping`,
  `:417-418`) parsed and the flag latched on: `:5050-5052`
  (`hparams.attn_soft_cap = true`).
- soft-cap applied **pre-softmax** as `scale → tanh → scale`:
  `kq = ggml_scale(ctx, kq, 1/f_attn_logit_softcapping)` (`:8284`),
  `ggml_tanh` (`:8285`), `ggml_scale(ctx, kq, f_attn_logit_softcapping)` (`:8286`);
  final-logit variant at `:11889-11891`.
- KV-cache V transposition flag: `cache.v_trans = !cparams.flash_attn` (`:3026`).

### 4.2 The transposition barrier — the *actual* failure chain

The brief's intuition ("soft-capping forces `v_trans = true`") is **correct, but
transitive**, and the precise chain matters for the fix:

```
attn_soft_cap == true            (Gemma2; llama.cpp:5052)
   └─► FlashAttention force-OFF  (llama.cpp:19032-19033:
          "flash_attn is not compatible with attn_soft_cap - forcing off")
          └─► cache.v_trans = !flash_attn = TRUE   (llama.cpp:3026)
                 └─► splicer hard-throws            (nexus_kv_splicer.cpp:243-244)
```

The splicer's guard:

```cpp
// src/nexus_kv_splicer.cpp:241-244
bool v_trans = llama_kv_cache_get_v_trans(ctx);
if (v_trans) {
    throw std::runtime_error("NexusKVSplicer Error: Flash Attention must be enabled (v_trans == false).");
}
```

So the engine is structurally incompatible with the Gemma2 family **not because
of soft-capping arithmetic itself**, but because soft-capping disqualifies
FlashAttention, which flips the V layout to transposed `[d_head, n_ctx]`, which
the contiguous row-major memcpy of §2.2 cannot address. The barrier is a hard
abort, not a degraded path.

### 4.3 Why the fixed-stride memcpy/router breaks on transposed / GQA layouts

The splicer asserts an exact contiguous, **non-transposed**, GQA-grouped stride
shape and refuses anything else:

```cpp
// expected — src/nexus_kv_splicer.cpp:282
// nb[0]=type_size, nb[1]=d_head*type_size, nb[2]=n_head_kv*d_head*type_size
// K validator — :293-295
k_contiguous = (k0->nb[0]==type_size) && (k0->nb[1]==d_head*type_size)
            && (k0->nb[2]==(size_t)n_head_kv * d_head * type_size);
// V validator — :310-312  (identical), throws on mismatch :298,:315
```

- Under **`v_trans = true`**, `v0->nb[1]/nb[2]` no longer equal
  `d_head·type_size` / `n_head_kv·d_head·type_size` (the V axes are swapped) →
  validator throws.
- Under a **GQA/MQA mismatch** (runtime `n_head_kv` ≠ `header->n_head_kv`), the
  row stride `k_size_row = n_head_kv·d_head·sizeof(F16)` (`:93`) computed from the
  block header no longer matches the live cache's `nb[2]` → the layer-0 validator
  throws *before* the memcpy. **This is detection, not silent corruption** (the
  brief implied corruption); the cost is a hard decline, but data integrity is
  preserved.

### 4.4 SLB — the fixed-stride INT8 SIMD dense router

`src/nexus_slb.cpp` (class `NexusSemanticSLB`, `nexus_slb.hpp:16`). This is the
tool-routing front end that selects which `.atb` to splice.

- **Fixed stride / SIMD alignment.** Embedding dim is padded up to a 32-byte
  multiple: `dim_ = ((dim + 31) / 32) * 32` (`:51`). Quantized tool vectors are
  stored back-to-back at exactly `dim_` bytes each
  (`tool_vec = quantized_vectors_.data() + idx*dim_`, used in `search` `:251-253`).
- **INT8 quant.** `scale = max_val / 127.0f` (`:66`), clamp to `[-127,127]`
  (`:230`), epsilon floor `max_val = max(max_val, 1e-4f)` to dodge FP16
  subnormals.
- **Kernel `dot_product_int8(a, b, dim, tool_sum)`** (`:119`), three dispatches:
  - **NEON**: stride `i += 16`, `vld1q_s8`, `accum = vdotq_s32(accum, va, vb)` (`:125-128`).
  - **AVX-512-VNNI**: stride `i += 32`, sign-shift `q_u = q_s + 128`, signed×unsigned
    `accum = _mm256_dpbusd_epi32(accum, q_u, t_s)` (`:161`), then bias removal
    `final_dot_product -= (128 * tool_sum)` (`:170`).
  - **AVX2 fallback**: widen INT8→INT16 and `_mm256_madd_epi16` over two halves
    (`:180, :184`).
- **Thread-local scratch** is 32-byte aligned:
  `static thread_local std::vector<int8_t, AlignedAllocator<int8_t,32>> tls_scratch_buffer`
  (`:222`), with hysteresis release above `TLS_HIGH_WATERMARK = 65536` (`:297-299`).
- **Dequant / fusion constants:** `score = int_dot · meta.scale · query_scale`
  (`:256`); top-K stack cap `MAX_K = 16` (`:244`); hybrid Reciprocal-Rank-Fusion
  `RRF_K = 60.0f` (`:317`, used `:330,:350`).

The router's correctness rests on the same flat-stride premise as the splicer:
each tool occupies precisely `dim_` contiguous INT8 bytes, and each cache row
occupies precisely `n_head_kv·d_head·2` contiguous F16 bytes. Any layout that
violates contiguity — transposed V (§4.2) or re-grouped GQA heads (§4.3) — is
outside the kernel's addressing model and is rejected at the splicer boundary.

### 4.5 Hardcoded cache constants (quoted verbatim)

`src/nexus_block_cache.hpp`:
- `static constexpr size_t MAX_THREADS = 128;` (`:248`)
- `static HazardSlot hazard_pointers[MAX_THREADS];` (`:249`; defn
  `alignas(128)` in `nexus_block_cache.cpp:12`) — lock-free reclamation array.
- cache-line pinning: `struct alignas(NEXUS_CACHE_LINE) {NexusTelemetry|CacheEntry|CacheShard}`
  (`:53, :100, :163`).
- eviction policy is **2Q / CLOCK**, not LFU: `std::atomic<bool> referenced{false} // CLOCK`
  (`:102`), `std::atomic<uint64_t> last_access_tick{0}` (`:106`),
  `bool is_protected{false} // Probation/Protected` (`:110`).

`src/nexus_seq_warm_cache.hpp`:
- `static constexpr llama_seq_id POOL_BASE = 1;` (`:17`)
- `static constexpr size_t POOL_SIZE = 32;` (`:18`) — 32 warm prefix slots
  (`pool_meta_` is `alignas(128) std::array<PoolSlotMeta, POOL_SIZE>`, `:91`).
- `static constexpr llama_seq_id REQUEST_SEQ_BASE = 64;` (`:19`)
- `static constexpr uint32_t NODE_ARENA_RESERVE = 4096;` (`:20`)
- `static constexpr uint32_t TOKEN_ARENA_RESERVE = 65536;` (`:21`)

`src/nexus_os_compat.hpp`: `inline constexpr size_t NEXUS_CACHE_LINE = 128;` (`:9`).

---

## Failure-mode summary

| Trigger | Mechanism | Code | Outcome |
|---|---|---|---|
| `n_past > 256` | RoPE Δθ across dims exceeds single-shift recovery; guard declines | `nexus_orchestrator.cpp:69` | Path B: full text re-prefill in Python (`nexus_agent.py:691`) |
| Gemma2 soft-cap | soft-cap → FlashAttn forced off → `v_trans=true` | `llama.cpp:19032-19033 → 3026` | splicer hard-throw (`nexus_kv_splicer.cpp:243-244`) |
| GQA/MQA stride mismatch | header `n_head_kv` ≠ live `nb[2]` grouping | `nexus_kv_splicer.cpp:293-295,310-312` | layer-0 validator throws (detected, not corrupting) |
| Deep off-anchor splice | per-dim phase error grows with `Δ = n_past − base_pos` | `nexus_kv_splicer.cpp:278`; `nexus_rope_math.cpp:84-137` | drift; mitigated by suffix recompute (`nexus_recompute.py`) + V/KQ calibration (`-0.02`/`-0.15`) |
