#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/shared_ptr.h>
#include <span>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "nexus_fsm.hpp"
#include "nexus_page_mounter.hpp"
#include "nexus_kv_splicer.hpp"
#include "nexus_slb.hpp"
#include "nexus_orchestrator.hpp"
#include "nexus_rope_math.hpp"
#include "nexus_block_cache.hpp"
#include <string_view>
#include <cstring>
#include "llama.h"

extern "C" {
    void llama_backend_sync(struct llama_context * ctx);
    uint32_t llama_model_n_head_kv(const struct llama_model * model, int il);
    uint32_t llama_model_d_head(const struct llama_model * model);
}

// Deterministic Memory Epoch Handle
struct NexusBlockHandle {
    uint64_t generation_id;
    void* mapped_data;
    size_t seq_len;
    size_t file_size;
};

namespace nb = nanobind;

// FFI helper to resolve Python capsule or raw pointer address to llama_context*
llama_context* resolve_context(nb::object obj) {
    if (obj.is_none()) {
        throw nb::type_error("llama_context cannot be None.");
    }
    llama_context* ctx = nullptr;
    if (nb::isinstance<nb::capsule>(obj)) {
        ctx = static_cast<llama_context*>(nb::cast<nb::capsule>(obj).data());
    } else {
        try {
            uintptr_t addr = nb::cast<uintptr_t>(obj);
            ctx = reinterpret_cast<llama_context*>(addr);
        } catch (...) {
            throw nb::type_error("Expected a capsule or integer pointer address for llama_context.");
        }
    }
    if (!ctx) {
        throw nb::value_error("llama_context pointer address cannot be null (0).");
    }
    return ctx;
}

// FFI wrapper function to convert nanobind::ndarray to std::span<float>
void apply_mask_ffi(NexusRadixFSM& fsm, nb::ndarray<float, nb::c_contig, nb::device::cpu> logits, llama_seq_id seq_id) {
    float* data = logits.data();
    size_t size = logits.size();
    
    std::span<float> logits_span(data, size);
    fsm.apply_logit_mask(logits_span, seq_id);
}

NB_MODULE(nexus_fsm_ext, m) {
    // Register custom exceptions
    nb::exception<ResourceExhaustedException>(m, "ResourceExhaustedError");

    // Enum bindings
    nb::enum_<RoutingState>(m, "RoutingState")
        .value("IDLE", RoutingState::IDLE)
        .value("NAVIGATING", RoutingState::NAVIGATING)
        .value("LEAF_REACHED", RoutingState::LEAF_REACHED)
        .value("INVALID", RoutingState::INVALID);

    // QuantizedBitmapAllocator bindings
    nb::class_<nexus::QuantizedBitmapAllocator>(m, "QuantizedBitmapAllocator")
        .def("__init__", [](nexus::QuantizedBitmapAllocator* self, uintptr_t base_addr, size_t size) {
            new (self) nexus::QuantizedBitmapAllocator(reinterpret_cast<void*>(base_addr), size);
        }, nb::arg("base_addr"), nb::arg("size"))
        .def("allocate", [](nexus::QuantizedBitmapAllocator& self, size_t size) -> uintptr_t {
            return reinterpret_cast<uintptr_t>(self.allocate(size));
        }, nb::arg("size"))
        .def("free", [](nexus::QuantizedBitmapAllocator& self, uintptr_t ptr_addr, size_t size) {
            self.free(reinterpret_cast<void*>(ptr_addr), size);
        }, nb::arg("ptr_addr"), nb::arg("size"));

    // Radix FSM bindings
    nb::class_<NexusRadixFSM>(m, "NexusRadixFSM")
        .def(nb::init<>())
        .def("add_route", [](NexusRadixFSM& fsm, uint32_t tool_id, const std::vector<llama_token>& tokens) {
            fsm.add_route(tool_id, std::span<const llama_token>(tokens.data(), tokens.size()));
        }, nb::arg("tool_id"), nb::arg("tokens"))
        .def("reset", &NexusRadixFSM::reset, nb::arg("seq_id") = 0)
        .def("begin_routing", &NexusRadixFSM::begin_routing, nb::arg("seq_id") = 0)
        .def("advance", &NexusRadixFSM::advance, nb::arg("sampled_token"), nb::arg("seq_id") = 0)
        .def("force_resolved", &NexusRadixFSM::force_resolved, nb::arg("tool_id"), nb::arg("seq_id") = 0)
        .def("apply_logit_mask", &apply_mask_ffi, nb::arg("logits"), nb::arg("seq_id") = 0)
        .def("get_state", &NexusRadixFSM::get_state, nb::arg("seq_id") = 0)
        .def("get_resolved_tool_id", &NexusRadixFSM::get_resolved_tool_id, nb::arg("seq_id") = 0)
        .def_prop_ro("state", [](const NexusRadixFSM& fsm) { return fsm.get_state(0); })
        .def_prop_ro("resolved_tool_id", [](const NexusRadixFSM& fsm) { return fsm.get_resolved_tool_id(0); });

    // Page Mounter bindings
    nb::class_<NexusPageMounter>(m, "NexusPageMounter")
        .def(nb::init<const std::string&>(), nb::arg("atb_filepath"));

    // AeonToolBlock bindings
    nb::class_<AeonToolBlock>(m, "AeonToolBlock")
        .def("get_file_size", &AeonToolBlock::get_file_size)
        .def_prop_ro("seq_len", [](const AeonToolBlock& b) { 
            b.wait_until_resident();
            if (!b.get_header()) throw std::runtime_error("Header is null");
            return b.get_header()->seq_len; 
        });

    // NexusBlockHandle bindings
    nb::class_<NexusBlockHandle>(m, "NexusBlockHandle")
        .def_ro("generation_id", &NexusBlockHandle::generation_id)
        .def_ro("seq_len", &NexusBlockHandle::seq_len)
        .def("get_file_size", [](const NexusBlockHandle& h) { return h.file_size; });

    // HazardGuard bindings (Python Context Manager)
    nb::class_<HazardGuard>(m, "HazardGuard")
        .def("__enter__", [](HazardGuard& self) -> uintptr_t {
            nb::gil_scoped_release release;
            if (self.entry && self.entry->future.valid()) {
                auto res = self.entry->future.get();
                if (res.has_value()) [[likely]] {
                    if (res.value()) [[likely]] {
                        return reinterpret_cast<uintptr_t>(res.value()->get_mapped_data());
                    }
                }
            }
            return 0;
        })
        .def("__exit__", [](HazardGuard& self, nb::object exc_type, nb::object exc_value, nb::object traceback) {
            self.release();
            return false; // Do not suppress exceptions
        });

    // NexusBlockCache bindings
    nb::class_<NexusBlockCache>(m, "NexusBlockCache")
        .def(nb::init<size_t>(), nb::arg("capacity"))
        .def("get_or_load", [](NexusBlockCache& cache, std::string_view atb_filepath) {
            std::shared_ptr<AeonToolBlock> block;
            {
                nb::gil_scoped_release release;
                auto result = cache.get_or_load(atb_filepath);
                if (result.has_value()) [[likely]] {
                    block = result.value();
                } else [[unlikely]] {
                    if (result.error() == NexusErrorCode::RESOURCE_EXHAUSTED) {
                        throw ResourceExhaustedException("NUMA Node Capacity Fully Saturated");
                    }
                    throw std::runtime_error("Internal Cache Error");
                }
            }
            block->wait_until_resident();
            
            NexusBlockHandle handle;
            handle.generation_id = block->get_generation_id();
            handle.mapped_data = block->get_mapped_data();
            handle.seq_len = block->get_header()->seq_len;
            handle.file_size = block->get_file_size();
            return handle;
        }, nb::arg("atb_filepath"))
        .def("prefetch", [](NexusBlockCache& cache, std::string_view atb_filepath) {
            nb::gil_scoped_release release;
            cache.prefetch(atb_filepath);
        }, nb::arg("atb_filepath"))
        .def("hazard_guard", [](NexusBlockCache& cache, uint32_t tool_id) {
            return cache.hazard_guard(tool_id);
        }, nb::rv_policy::move, nb::arg("tool_id"))
        .def("pin_tool", &NexusBlockCache::pin_tool, nb::arg("tool_id"))
        .def("unpin_tool", &NexusBlockCache::unpin_tool, nb::arg("tool_id"))
        .def("is_tool_pinned", &NexusBlockCache::is_tool_pinned, nb::arg("tool_id"));

    // KV Splicer bindings
    m.def("invalidate_sequence", [](nb::object ctx_obj, llama_seq_id seq, int32_t start_pos, int32_t end_pos) {
        llama_context* ctx = resolve_context(ctx_obj);
        nb::gil_scoped_release release;
        NexusKVSplicer::invalidate_sequence(ctx, seq, start_pos, end_pos);
    }, nb::arg("ctx"), nb::arg("seq"), nb::arg("start_pos"), nb::arg("end_pos"));

    m.def("inject_tool_page", [](nb::object ctx_obj, NexusBlockHandle handle, uint32_t n_past, llama_seq_id target_seq) {
        llama_context* ctx = resolve_context(ctx_obj);
        nb::gil_scoped_release release;
        NexusKVSplicer::inject_tool_page_raw(ctx, handle.mapped_data, n_past, target_seq);
    }, nb::arg("ctx"), nb::arg("handle"), nb::arg("n_past"), nb::arg("target_seq") = 0);

    m.def("inject_tool_page", [](nb::object ctx_obj, NexusPageMounter& mounter, uint32_t n_past, llama_seq_id target_seq) {
        llama_context* ctx = resolve_context(ctx_obj);
        nb::gil_scoped_release release;
        NexusKVSplicer::inject_tool_page_raw(ctx, mounter.get_mapped_data(), n_past, target_seq);
    }, nb::arg("ctx"), nb::arg("mounter"), nb::arg("n_past"), nb::arg("target_seq") = 0);

    m.def("apply_relative_rope_shift", [](nb::ndarray<uint16_t, nb::c_contig, nb::device::cpu> k_data,
                                           uint32_t seq_len, uint32_t n_head_kv, uint32_t d_head, int32_t delta_pos,
                                           float freq_base, float freq_scale, uint32_t scaling_type,
                                           float ext_factor, float beta_fast, float beta_slow, int n_ctx_orig) {
        if (k_data.size() != seq_len * n_head_kv * d_head) {
            throw std::runtime_error("ndarray size mismatch with seq_len * n_head_kv * d_head");
        }
        nb::gil_scoped_release release;
        apply_relative_rope_shift(
            k_data.data(),
            seq_len, n_head_kv, d_head, delta_pos,
            freq_base, freq_scale, scaling_type,
            ext_factor, beta_fast, beta_slow, n_ctx_orig
        );
    });

    m.def("clear_kv_cache", [](nb::object ctx_obj) {
        llama_context* ctx = resolve_context(ctx_obj);
        nb::gil_scoped_release release;
        llama_kv_cache_clear(ctx);
    }, nb::arg("ctx"));

    m.def("kv_cache_seq_cp", [](nb::object ctx_obj, llama_seq_id seq_src, llama_seq_id seq_dst,
                                 int32_t p0, int32_t p1) {
        llama_context* ctx = resolve_context(ctx_obj);
        nb::gil_scoped_release release;
        llama_kv_cache_seq_cp(ctx, seq_src, seq_dst,
                              static_cast<llama_pos>(p0),
                              p1 < 0 ? static_cast<llama_pos>(-1) : static_cast<llama_pos>(p1));
    }, nb::arg("ctx"), nb::arg("seq_src"), nb::arg("seq_dst"), nb::arg("p0"), nb::arg("p1") = -1);

    m.def("read_kv_slice", [](nb::object ctx_obj, int il, int32_t p0, int32_t p1, bool read_v) {
        llama_context* ctx = resolve_context(ctx_obj);
        std::vector<float> data;
        {
            nb::gil_scoped_release release;
            data = NexusKVSplicer::read_kv_slice(ctx, il, p0, p1, read_v);
        }
        const struct llama_model* model = llama_get_model(ctx);
        uint32_t n_head_kv = llama_model_n_head_kv(model, il);
        uint32_t d_head = llama_model_d_head(model);
        uint32_t seq_len = static_cast<uint32_t>(p1 - p0);
        return nb::make_tuple(data, seq_len, n_head_kv, d_head);
    }, nb::arg("ctx"), nb::arg("layer"), nb::arg("p0"), nb::arg("p1"), nb::arg("read_v") = false);

    m.def("recompute_fused_with_query", [](nb::object ctx_obj,
                                          const std::vector<int32_t>& schema_tokens,
                                          uint32_t p_start,
                                          int32_t suffix_start_idx,
                                          const std::vector<int32_t>& query_tokens,
                                          llama_seq_id seq_id) {
        llama_context* ctx = resolve_context(ctx_obj);
        if (query_tokens.empty()) {
            throw std::runtime_error("recompute_fused_with_query requires non-empty query_tokens.");
        }
        const int32_t schema_len = static_cast<int32_t>(schema_tokens.size());
        if (suffix_start_idx < 0) {
            suffix_start_idx = 0;
        }
        if (suffix_start_idx > schema_len) {
            suffix_start_idx = schema_len;
        }

        const int32_t abs_start = static_cast<int32_t>(p_start) + suffix_start_idx;
        const int32_t abs_end = static_cast<int32_t>(p_start) + schema_len;
        if (suffix_start_idx < schema_len) {
            NexusKVSplicer::invalidate_sequence(ctx, seq_id, abs_start, abs_end);
        }

        const int32_t n_suffix = schema_len - suffix_start_idx;
        const int32_t n_query = static_cast<int32_t>(query_tokens.size());
        const int32_t n_total = n_suffix + n_query;

        llama_batch batch = llama_batch_init(n_total, 0, 1);
        batch.n_tokens = n_total;
        int32_t bi = 0;
        for (int32_t i = suffix_start_idx; i < schema_len; ++i) {
            batch.token[bi] = schema_tokens[static_cast<size_t>(i)];
            batch.pos[bi] = static_cast<llama_pos>(p_start + static_cast<uint32_t>(i));
            batch.n_seq_id[bi] = 1;
            batch.seq_id[bi][0] = seq_id;
            batch.logits[bi] = 0;
            ++bi;
        }
        for (int32_t i = 0; i < n_query; ++i) {
            batch.token[bi] = query_tokens[static_cast<size_t>(i)];
            batch.pos[bi] = static_cast<llama_pos>(
                p_start + static_cast<uint32_t>(schema_len) + static_cast<uint32_t>(i));
            batch.n_seq_id[bi] = 1;
            batch.seq_id[bi][0] = seq_id;
            batch.logits[bi] = (i == n_query - 1) ? 1 : 0;
            ++bi;
        }

        int decode_res = 0;
        {
            nb::gil_scoped_release release;
            decode_res = llama_decode(ctx, batch);
            llama_backend_sync(ctx);
        }
        if (decode_res != 0) {
            llama_batch_free(batch);
            throw std::runtime_error("recompute_fused_with_query: llama_decode failed with code " +
                                     std::to_string(decode_res));
        }

        const struct llama_model* model = llama_get_model(ctx);
        const int32_t n_vocab = llama_n_vocab(model);
        float* logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
        if (!logits) {
            llama_batch_free(batch);
            throw std::runtime_error("recompute_fused_with_query: llama_get_logits_ith returned null.");
        }
        std::vector<float> result(logits, logits + n_vocab);
        llama_batch_free(batch);
        return result;
    }, nb::arg("ctx"), nb::arg("schema_tokens"), nb::arg("p_start"), nb::arg("suffix_start_idx"),
       nb::arg("query_tokens"), nb::arg("seq_id") = 0);

    m.def("decode_tokens", [](nb::object ctx_obj, const std::vector<int32_t>& tokens, uint32_t start_pos, llama_seq_id seq_id) {
        llama_context* ctx = resolve_context(ctx_obj);
        if (tokens.empty()) {
            throw std::runtime_error("decode_tokens requires at least one token.");
        }

        llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
        batch.n_tokens = static_cast<int32_t>(tokens.size());
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            batch.token[i] = tokens[static_cast<size_t>(i)];
            batch.pos[i] = static_cast<llama_pos>(start_pos + static_cast<uint32_t>(i));
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = seq_id;
            batch.logits[i] = (i == batch.n_tokens - 1) ? 1 : 0;
        }

        int decode_res = 0;
        {
            nb::gil_scoped_release release;
            decode_res = llama_decode(ctx, batch);
            llama_backend_sync(ctx);
        }
        if (decode_res != 0) {
            llama_batch_free(batch);
            throw std::runtime_error("llama_decode failed with code " + std::to_string(decode_res));
        }

        const struct llama_model* model = llama_get_model(ctx);
        int32_t n_vocab = llama_n_vocab(model);
        float* logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
        if (!logits) {
            llama_batch_free(batch);
            throw std::runtime_error("llama_get_logits_ith returned null.");
        }
        std::vector<float> result(logits, logits + n_vocab);
        llama_batch_free(batch);
        return result;
    }, nb::arg("ctx"), nb::arg("tokens"), nb::arg("start_pos"), nb::arg("seq_id") = 0);

    m.def("unsplice_tool", [](nb::object ctx_obj, NexusOrchestrator& orchestrator, llama_seq_id seq_id, uint32_t splice_pos, uint32_t schema_len, uint32_t generated_len) {
        orchestrator.release_hazard(seq_id);
        llama_context* ctx = resolve_context(ctx_obj);
        nb::gil_scoped_release release;
        return NexusKVSplicer::unsplice_tool(ctx, seq_id, splice_pos, schema_len, generated_len);
    }, nb::arg("ctx"), nb::arg("orchestrator"), nb::arg("seq_id"), nb::arg("splice_pos"), nb::arg("schema_len"), nb::arg("generated_len"));

    m.def("unsplice_tool", [](nb::object ctx_obj, llama_seq_id seq_id, uint32_t splice_pos, uint32_t schema_len, uint32_t generated_len) {
        llama_context* ctx = resolve_context(ctx_obj);
        nb::gil_scoped_release release;
        return NexusKVSplicer::unsplice_tool(ctx, seq_id, splice_pos, schema_len, generated_len);
    }, nb::arg("ctx"), nb::arg("seq_id"), nb::arg("splice_pos"), nb::arg("schema_len"), nb::arg("generated_len"));

    // SearchResult bindings
    nb::class_<NexusSemanticSLB::SearchResult>(m, "SearchResult")
        .def_ro("tool_id", &NexusSemanticSLB::SearchResult::tool_id)
        .def_ro("score", &NexusSemanticSLB::SearchResult::score)
        .def_ro("digest_offset", &NexusSemanticSLB::SearchResult::digest_offset)
        .def_ro("digest_len", &NexusSemanticSLB::SearchResult::digest_len)
        .def_prop_ro("scent_tokens", [](const NexusSemanticSLB::SearchResult& res) {
            return res.scent_tokens;
        });

    // Semantic SLB bindings
    nb::class_<NexusSemanticSLB>(m, "NexusSemanticSLB")
        .def(nb::init<size_t, size_t>(), nb::arg("dim"), nb::arg("max_tool_tokens") = 32)
        .def("register_tool", [](NexusSemanticSLB& slb, uint32_t tool_id, nb::ndarray<float, nb::c_contig, nb::device::cpu> vec, const std::vector<int32_t>& digest_tokens,
                                  const std::vector<uint32_t>& lexical_hashes) {
            std::span<const float> vec_span(vec.data(), vec.size());
            std::span<const int32_t> digest_span(digest_tokens.data(), digest_tokens.size());
            std::span<const uint32_t> lex_span(lexical_hashes.data(), lexical_hashes.size());
            slb.register_tool(tool_id, vec_span, digest_span, lex_span);
        }, nb::arg("tool_id"), nb::arg("vector"), nb::arg("scent"), nb::arg("lexical_hashes") = std::vector<uint32_t>{})
        .def("search", [](const NexusSemanticSLB& slb, nb::ndarray<float, nb::c_contig, nb::device::cpu> query, size_t top_k) {
            nb::gil_scoped_release release;
            std::span<const float> query_span(query.data(), query.size());
            return slb.search(query_span, top_k);
        }, nb::arg("query"), nb::arg("top_k") = 3)
        .def("search_hybrid", [](const NexusSemanticSLB& slb, nb::ndarray<float, nb::c_contig, nb::device::cpu> query,
                                 const std::vector<uint32_t>& lexical_hashes, size_t top_k) {
            nb::gil_scoped_release release;
            std::span<const float> query_span(query.data(), query.size());
            std::span<const uint32_t> lex_span(lexical_hashes.data(), lexical_hashes.size());
            return slb.search_hybrid(query_span, lex_span, top_k);
        }, nb::arg("query"), nb::arg("lexical_hashes"), nb::arg("top_k") = 3)
        .def("evaluate_slb", [](const NexusSemanticSLB& slb, nb::ndarray<float, nb::c_contig, nb::device::cpu> query, size_t top_k) {
            nb::gil_scoped_release release;
            std::span<const float> query_span(query.data(), query.size());
            return slb.search(query_span, top_k);
        }, nb::arg("query"), nb::arg("top_k") = 3)
        .def_prop_ro("dim", &NexusSemanticSLB::get_dim)
        .def_prop_ro("padded_dim", &NexusSemanticSLB::get_padded_dim)
        .def_prop_ro("max_tool_tokens", &NexusSemanticSLB::get_max_tool_tokens)
        .def_prop_ro("count", &NexusSemanticSLB::get_count);

    nb::class_<PyTelemetryBlock>(m, "PyTelemetryBlock")
        .def_ro("speculative_hit_count", &PyTelemetryBlock::speculative_hit_count)
        .def_ro("speculative_miss_count", &PyTelemetryBlock::speculative_miss_count)
        .def_ro("fsm_hidden_latency_us", &PyTelemetryBlock::fsm_hidden_latency_us)
        .def_ro("exposed_splice_latency_us", &PyTelemetryBlock::exposed_splice_latency_us);

    // Orchestrator bindings (MMU only — no grammar state)
    nb::class_<NexusOrchestrator>(m, "NexusOrchestrator")
        .def("__init__", [](NexusOrchestrator* self, nb::object ctx_obj, NexusSemanticSLB& slb, NexusRadixFSM& fsm, uint32_t base_pos,
                            float speculative_threshold, float speculative_margin, float auto_route_margin, size_t max_pinned_bytes,
                            uint32_t max_splice_pos) {
            llama_context* ctx = resolve_context(ctx_obj);
            new (self) NexusOrchestrator(ctx, &slb, &fsm, base_pos, speculative_threshold, speculative_margin, auto_route_margin, max_pinned_bytes, max_splice_pos);
        }, nb::arg("ctx"), nb::arg("slb"), nb::arg("fsm"), nb::arg("base_pos"),
           nb::arg("speculative_threshold") = 0.88f, nb::arg("speculative_margin") = 0.05f,
           nb::arg("auto_route_margin") = 0.10f,
           nb::arg("max_pinned_bytes") = 16ULL * 1024 * 1024 * 1024,
           nb::arg("max_splice_pos") = 256)
        .def("register_tool_path", &NexusOrchestrator::register_tool_path, nb::arg("tool_id"), nb::arg("path"))
        .def("register_tool_schema_tokens", &NexusOrchestrator::register_tool_schema_tokens,
             nb::arg("tool_id"), nb::arg("tokens"))
        .def("set_recompute_pct", &NexusOrchestrator::set_recompute_pct, nb::arg("pct"))
        .def("get_recompute_pct", &NexusOrchestrator::get_recompute_pct)
        .def("set_max_splice_pos", &NexusOrchestrator::set_max_splice_pos, nb::arg("pos"))
        .def("get_max_splice_pos", &NexusOrchestrator::get_max_splice_pos)
        .def("preload_tool", [](NexusOrchestrator& self, uint32_t tool_id, const std::string& path) {
            nb::gil_scoped_release release;
            self.preload_tool(tool_id, path);
        }, nb::arg("tool_id"), nb::arg("path"))
        .def("route_and_splice", [](NexusOrchestrator& self, const std::vector<int32_t>& user_query_tokens, const std::vector<float>& query_embedding, uint32_t n_past, llama_seq_id seq_id,
                                    const std::vector<uint32_t>& query_lexical_hashes,
                                    const std::vector<int32_t>& prefix_tokens,
                                    uint32_t force_tool_id) {
            nb::gil_scoped_release release;
            return self.route_and_splice(user_query_tokens, query_embedding, n_past, seq_id, query_lexical_hashes,
                                         prefix_tokens, force_tool_id);
        }, nb::arg("user_query_tokens"), nb::arg("query_embedding"), nb::arg("n_past"), nb::arg("seq_id") = 0,
           nb::arg("query_lexical_hashes") = std::vector<uint32_t>{},
           nb::arg("prefix_tokens") = std::vector<int32_t>{},
           nb::arg("force_tool_id") = 0)
        .def("register_prefix_tokens", &NexusOrchestrator::register_prefix_tokens,
             nb::arg("tool_id"), nb::arg("prefix_tokens"))
        .def("release_hazard", &NexusOrchestrator::release_hazard, nb::arg("seq_id") = 0)
        .def("get_cache", &NexusOrchestrator::get_cache)
        .def("get_telemetry", &NexusOrchestrator::get_telemetry)
        .def("pin_warm_tool", &NexusOrchestrator::pin_warm_tool, nb::arg("tool_id"))
        .def("pin_block_tool", &NexusOrchestrator::pin_block_tool, nb::arg("tool_id"))
        .def("get_splice_guard_fallback_count", &NexusOrchestrator::get_splice_guard_fallback_count);

    m.def("allocate_request_seq", &NexusRadixPrefixCache::allocate_request_seq, nb::arg("request_index"));
}
