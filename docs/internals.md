# Nexus Internals

This is the candid, implementation-level document: the invariants that must hold, the
model-bound assumptions baked into the binary format, what each Phase-H result actually proved
(and what one of them *retired*), which artifacts are authoritative, and the traps a new
contributor will otherwise hit. It pairs with [architecture.md](architecture.md) (the mental
model) and the [README](../README.md) (the headline and scope).

**Validation anchor.** Canonical tuple: Qwen2.5-14B-Instruct Q4_K_M (`model_hash a09ea5e7`),
`llama_cpp`/`llama.cpp` build `cb2463bb`, one Apple-Silicon host, git `1ce4aa4`. Authoritative
artifacts under `results/v2.0_canonical/`; the manifest is the source of truth.

> **Two "Radix" types, do not conflate them.** `NexusRadixPrefixCache`
> (`nexus_seq_warm_cache.{cpp,hpp}`) is the **L0 warm KV prefix cache**. `NexusRadixFSM`
> (`nexus_fsm.{cpp,hpp}`) is a **decode-time logit-masking trie** over tool-name tokens. They
> share a name fragment and nothing else.

---

## 1. Runtime invariants

Hold these true or the splice is incorrect:

1. **Position bound.** Path A runs only when `n_past ≤ MAX_SPLICE_POS` (256). The orchestrator
   enforces it at the top of `route_and_splice` (`nexus_orchestrator.cpp:69`); over the bound it
   returns `0` and the Python layer text-prefills.
2. **Anchored injection.** Path A's `.atb` blocks are compiled at `base_pos = 0` and injected on
   the short path, so `delta_pos = n_past − base_pos = 0`. Anchored splices are output-exact;
   off-anchor splices degrade (§7, §8).
3. **Layout writability.** The splicer only writes a KV layout it can legally address — i.e.
   `v_trans = false` (flash attention on). `v_trans = true` is rejected (§4).
4. **RoPE parameter match.** Block `rope_freq_scale` / `rope_scaling_type` must match the live
   model within `1e-5`; otherwise the orchestrator throws (`nexus_orchestrator.cpp:167-173`).
5. **Lifecycle cleanup.** Every splice is RAII-scoped (`NexusSpliceContext`); the spliced KV
   range is removed on turn exit, including on exception/cancellation.
6. **Context capacity.** `n_past + schema_len + Q` must fit `llama_n_ctx`; checked before
   injection, throws `length_error` otherwise.

## 2. ABI / model-bound assumptions

A `.atb` block is **not portable across models.** It is a frozen dump of FP16 K/V tensors whose
shape and numerical content depend on the exact model topology and RoPE configuration. The
header binds it; the splicer validates the binding; a mismatch is a hard error, never a silent
reinterpretation.

The binary is little-endian only (`static_assert` in `aeon_tool_block.hpp:6`) and physically
2 MiB aligned (`NEXUS_PAGE_ALIGNMENT = 2097152`) for huge-page / direct-I/O friendliness.

## 3. `.atb` header — the model dependence, field by field

128 bytes, `#pragma pack(push,1)`, `alignas(128)`, with `static_assert`s pinning every offset
(`aeon_tool_block.hpp:13-58`):

| Offset | Field | Why it binds the block to a model |
| --- | --- | --- |
| 0 | `magic[4]` = `"ATB1"` | Format guard |
| 4 | `version` = 1 | ABI version |
| 8 | `model_hash` | FNV-1a of model topology — the master binding key |
| 16 | `n_layer` | Layer-loop bound; KV stride |
| 20 | `n_head_kv` | GQA-aware KV head count; tensor layout |
| 24 | `d_head` | Per-head dimension; K/V row size, RoPE dim |
| 28 | `seq_len` | Active tokens in this block (schema length) |
| 32 | `base_pos` | **RoPE compile position** — drives anchored vs off-anchor |
| 36 | `rope_freq_base` | RoPE theta base |
| 40 | `rope_freq_scale` | RoPE scaling (default 1.0); matched at runtime |
| 44 | `rope_scaling_type` | 0 standard / 1 linear / 2 YaRN; matched at runtime |
| 48 | `ggml_type_k` | Must be `GGML_TYPE_F16` (1) |
| 52 | `ggml_type_v` | Must be `GGML_TYPE_F16` (1) |
| 56 / 64 | `k_tensor_offset` / `v_tensor_offset` | Byte offsets to contiguous K/V data |
| 72 / 80 | `k_total_bytes` / `v_total_bytes` | `seq_len * n_layer * n_head_kv * d_head * 2` |
| 88 | `padding[40]` | Pad to exactly 128 bytes |

`model_hash` (FNV-1a over model name, `n_layer`, `d_head`, `rope_freq_base`, and per-layer
`n_head_kv`) is the master key: change the model and the hash, layer count, head count, head
dim, or RoPE params change, and the block is no longer valid. There is no cross-model fallback —
a mismatched block is rejected, not reinterpreted.

## 4. KV layout assumptions — `v_trans` and flash-attention dependence

The single most important portability fact lives in the splicer. The splice writes K and V
tensors directly into llama.cpp's KV cache memory, which means it depends on the **physical
layout** of that cache:

- With **flash attention on**, the V cache is stored non-transposed: `v_trans = false`. The
  splicer can address and write it.
- With **flash attention off**, llama.cpp keeps the V cache **transposed**: `v_trans = true`.
  The splicer **rejects** this — it cannot legally write that layout
  (`nexus_kv_splicer.cpp`, the `v_trans` guard).

This is not a tuning knob. It is why the mechanism's scope is *flash-attention-compatible*
architectures, and it is the mechanism by which Gemma2 is blocked (§8).

The splicer's transfer path also branches on memory topology: on UMA it does a parallel
per-layer blit (and a parallel CPU RoPE reanchor for off-anchor cases); on discrete GPUs it
issues async H2D transfers and a GPU RoPE reanchor. The anchored path needs no RoPE step at all.

## 5. Model metadata dependencies (summary)

The runtime quantities that must agree between block and live model:

- `n_layer`, `n_head_kv`, `d_head` — encoded in `model_hash`; wrong values mean wrong tensor
  shapes.
- `rope_freq_base`, `rope_freq_scale`, `rope_scaling_type` — matched within `1e-5`; mismatch
  throws.
- `ggml_type_k` / `ggml_type_v` — must be FP16.
- `v_trans` — runtime property of the live model's attention; must be `false`.

## 6. What "anchored exactness" means operationally

Anchored = the block is injected at the same RoPE base position it was compiled for, so
`delta_pos == 0` and **no RoPE rotation is applied** to the stored tensors. The K/V written into
the cache are bit-for-bit the tensors a real prefill at that position would have produced. H5
measured the downstream consequence directly: feeding both a real-prefill context and an
anchored-spliced context through the model yields **logit-KL = 0.0** and **top-1 agreement =
1.0** over n=50 (`phaseH/splice_fidelity_anchored_H5.json`). Operationally: an anchored splice
is indistinguishable from prefill at the output-distribution level.

## 7. What "isolated degradation" means operationally

Isolated / off-anchor = `delta_pos != 0`, so the stored tensors must be RoPE-reanchored to the
injection position before they are coherent. H5's off-anchor measurements show the reanchor does
**not** recover prefill-equivalent state: logit-KL rises to ~0.9 and top-1 agreement falls to
~0.33–0.67. Operationally: off-anchor splices change the model's output distribution enough to
matter. The design's response is to **avoid** the off-anchor regime (the `n_past ≤ 256` gate)
rather than to trust the reanchor to fix it.

## 8. What H1 proved — structurally

H1 attempted a second model (Gemma2) and produced a **structural** narrowing, not a numeric one:

1. Gemma2 uses attention logit **soft-capping**.
2. The pinned `llama.cpp` forces flash attention **off** when soft-capping is present.
3. Flash-attention-off leaves the V cache **transposed** → `v_trans = true`.
4. The splicer **rejects** `v_trans = true` at `inject_tool_page`.

The splice therefore cannot execute at all on Gemma2. H1 did **not** measure whether the
anchored/isolated fidelity numbers generalize — it established that the mechanism is
*structurally inapplicable* to a soft-capped family on this stack. Conclusion of record: scope
is **flash-attention-compatible, non-soft-capped** architectures, and **Qwen2.5 is the only
validated working configuration.** Artifact: `phaseH/PHASE_H_H1_SECONDMODEL_REVIEW.md`.

## 9. What H3 proved — statistically

H3 measured **true TTFT** (first-token decode included), serially, n=100 per arm
(`phaseH/h3_e2e_n100.json`, `PHASE_H_H3_E2E_INTERVAL_REVIEW.md`):

- N1x median true-TTFT **457.5 ms** (CI [445, 469] ms); B3 **1131.4 ms** (CI [1121, 1248] ms).
- Speedup **2.47×**, 95% CI **[2.41, 2.72]**.
- Accuracy: N1x 0.91, B3 0.92; delta **−0.010**, 95% CI **[−0.080, +0.050]**; McNemar
  (B3 vs N1) **p = 1.0**, not significant.

What changed from earlier reporting: the prior **n=30 "6-point gap" (0.87 vs 0.93) was small-n
noise and does not survive** at n=100. The correct statement is "no detectable gap at n=100,"
**not** "equal accuracy." The earlier n=30 e2e figures (≈518 / 1214 ms, ≈2.34×) are superseded
by these CI-backed n=100 numbers.

## 10. What H5 invalidated / retired, and what it regenerated

H5 did two things. **Regenerated (live):**

- **SLB scalability** (`phaseH/slb_scalability_H5.csv`): P50 scan **1.375 µs @ N=10 → 21.1 µs @
  N=1000**, **3.71 µs @ N=100**; the <5 µs-at-N≤100 budget holds.
- **Anchored splice fidelity** (`phaseH/splice_fidelity_anchored_H5.json`): logit-KL 0.0,
  top-1 1.0 (§6).

**Invalidated / retired:** the legacy tensor-KL boundary curve.

> The old numbers `0.0076 @P256` and `5.72 @P1024`, presented as a graceful "tensor-KL"
> boundary, are **retired and not citable as regenerated evidence.** Reasons, from
> `phaseH/PHASE_H_TENSORKL_BOUNDARY_REVIEW.md`:
>
> - **No tensor-KL harness exists in the repo.** The only reproducible fidelity metric families
>   are **tensor-L2** and **logit/output-distribution KL**. The "tensor-KL" label described a
>   metric that was never implemented as a harness; the values survive only as a prose note in a
>   legacy verdict file.
> - The regenerated sweep at P=256/512/1024 shows **tensor-L2 ≈ 12, position-independent** — no
>   boundary effect at all — and **logit-KL ≈ 0.9–1.1 uniformly** in the isolated regime, not a
>   `0.0076`-floor-rising-to-`5.72`-spike curve.
> - `5.72` was a **LegoLink scattered-partial-recompute** configuration that no current harness
>   implements; it does not reproduce.
>
> What survives is the **binary regime story**, in the correct metric: anchored → logit-KL 0.0
> (exact); isolated → logit-KL ~0.9 (degraded). Cite that, not the curve.

## 11. Artifact taxonomy — canonical vs raw vs legacy

`results/v2.0_canonical/MANIFEST.json` is the authority. Categories:

| Class | Where | Meaning |
| --- | --- | --- |
| **Canonical** | `v2.0_canonical/*` | Frozen 2026-07-01; reproducible on the pinned stack; the live source of truth |
| **Raw** | `v2.0_canonical/raw/` | Per-seed / per-run inputs behind the pooled canonical metrics |
| **Legacy / retired** | LegoLink scattered partial-recompute `5.72` KL; reference-free gating proxy | Do not cite as live evidence for production path |

Primary canonical artifacts you will actually read:

- `raw/deep_splice_ttft.json` — the TTFT latency under deep splice curves.
- `raw/dkl_sweep.json` — the next-token D_KL vs splice offset sweep.
- `raw/routing_accuracy_n250.json` — routing accuracy vs registry scale.
- `raw/sidecar/accuracy.json` — hybrid sidecar routing & argument accuracies.

## 12. Developer warnings — what NOT to assume

- **Do not assume portability.** One model, one host, one stack. A `.atb` is invalid for any
  other model; the mechanism is structurally blocked on soft-capped architectures.
- **Do not cite `0.0076` / `5.72`.** Retired; no harness; use anchored logit-KL 0.0 / tensor-L2
  ~12 instead (§10).
- **"Copy" is real, not "zero-copy."** The L0 warm tier performs an actual
  `llama_kv_cache_seq_cp`. It is cheap (~3.06 µs P50) but it is a copy; do not describe it as
  zero-copy.
- **L0 cache ≠ FSM trie.** `NexusRadixPrefixCache` (KV reuse) and `NexusRadixFSM` (decode
  masking) are different objects (top of doc).
- **Margin-threshold footgun.** If `src/` is not on the import path,
  `nexus_retrieval.DEFAULT_RERANK_MARGIN` falls back to `0.028`; the real calibrated value is
  `0.013646852970123292` (`src/nexus_calibration.py`). Always put `src/` on the path.
- **Route-only ≠ TTFT.** The gateway 160.2 ms metric sets `decode_us = 0`. Never label it TTFT.
- **Deep-path L0 is read-only.** On Path B, `probe_prefix` measures a potential hit but never
  `seq_cp`s it and never writes back; the warm pool is not populated on the deep path. Whether a
  deep-path hit would save real work is **unproven** — do not wire it as if it does.
- **`bench_phase28_radix_prefix` default model footgun.** Its default model resolves to an
  unrelated Ollama blob; always pass `--model` explicitly.

## 13. Reproducibility and provenance expectations

- **Interpreter:** `.venv/bin/python` only — the system `python3` cannot import `llama_cpp`. (A
  past "BLOCKED" audit verdict was a venv/sandbox artifact, not a real failure.)
- **Path:** `PYTHONPATH=build:src:test`.
- **Serial e2e:** run end-to-end benchmarks one at a time. Concurrent runs load the 14B model
  twice and contend, inflating B3 absolutes (an early contended run reported B3 ≈ 2011 ms and
  was discarded).
- **Provenance to record with any result:** git commit, `llama.cpp` build (`cb2463bb`), model
  hash (`a09ea5e7`), host, sample size, and whether the metric is route-only or true TTFT.

## 14. If you want to extend Nexus — inspect these first

| Goal | Start here |
| --- | --- |
| Change routing / gate logic | `nexus_orchestrator.cpp` → `route_and_splice` (guard `:69`, gate `:155-157`, RoPE check `:167`) |
| Touch the splice / KV write path | `nexus_kv_splicer.cpp` → `inject_tool_page_raw` (header validate, `v_trans` guard, transfer, RoPE) |
| Off-anchor / RoPE reanchor work | `nexus_rope_math.cpp` → `apply_absolute_rope_reanchor`, `apply_relative_rope_shift` |
| Router / retrieval | `nexus_slb.{cpp,hpp}` → `register_tool`, `search`, `search_hybrid` |
| L0 warm cache | `nexus_seq_warm_cache.cpp` → `try_copy_prefix`, `probe_prefix`, `update_from_seq` |
| Block format / I/O / eviction | `aeon_tool_block.hpp` (header), `nexus_block_cache.cpp` (load, hazard pointers, eviction) |
| Compile a new `.atb` | `nexus_kv_compiler.cpp` (`--model`, `--schema`, `--output`, `--base-pos`, `--pos-bucket`) |
| Python lifecycle / FFI | `nexus_agent.py` → `NexusSpliceContext`, `NexusRoutingProcessor`, `_extract_context_pointer` |
| Add bindings | `bindings.cpp` (nanobind module `nexus_fsm_ext`) |

**First thing to verify on any new model:** whether it runs with flash attention on
(`v_trans = false`). If it is soft-capped or otherwise forces flash attention off, the splice
path is structurally inapplicable and no amount of routing or calibration work will change that
— that is the lesson of H1.
