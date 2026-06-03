#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/shared_ptr.h>
#include <span>
#include <cstdint>
#include <stdexcept>

#include "nexus_fsm.hpp"
#include "nexus_page_mounter.hpp"
#include "nexus_kv_splicer.hpp"
#include "nexus_slb.hpp"
#include "nexus_orchestrator.hpp"
#include "nexus_rope_math.hpp"
#include "nexus_block_cache.hpp"
#include <string_view>
#include "llama.h"

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
    // Enum bindings
    nb::enum_<RoutingState>(m, "RoutingState")
        .value("IDLE", RoutingState::IDLE)
        .value("NAVIGATING", RoutingState::NAVIGATING)
        .value("LEAF_REACHED", RoutingState::LEAF_REACHED)
        .value("INVALID", RoutingState::INVALID);

    // Radix FSM bindings
    nb::class_<NexusRadixFSM>(m, "NexusRadixFSM")
        .def(nb::init<>())
        .def("add_route", [](NexusRadixFSM& fsm, uint32_t tool_id, const std::vector<llama_token>& tokens) {
            fsm.add_route(tool_id, std::span<const llama_token>(tokens.data(), tokens.size()));
        }, nb::arg("tool_id"), nb::arg("tokens"))
        .def("reset", &NexusRadixFSM::reset, nb::arg("seq_id") = 0)
        .def("begin_routing", &NexusRadixFSM::begin_routing, nb::arg("seq_id") = 0)
        .def("advance", &NexusRadixFSM::advance, nb::arg("sampled_token"), nb::arg("seq_id") = 0)
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
            if (self.entry && self.entry->future.valid()) {
                try {
                    auto block = self.entry->future.get();
                    if (block) {
                        return reinterpret_cast<uintptr_t>(block->get_mapped_data());
                    }
                } catch (...) {}
            }
            return 0;
        })
        .def("__exit__", [](HazardGuard& self, nb::object exc_type, nb::object exc_value, nb::object traceback) {
            if (self.entry) {
                self.entry->active_readers.fetch_sub(1, std::memory_order_release);
                self.entry = nullptr;
            }
            return false; // Do not suppress exceptions
        });

    // NexusBlockCache bindings
    nb::class_<NexusBlockCache>(m, "NexusBlockCache")
        .def(nb::init<size_t>(), nb::arg("capacity"))
        .def("get_or_load", [](NexusBlockCache& cache, std::string_view atb_filepath) {
            auto block = cache.get_or_load(atb_filepath);
            block->wait_until_resident();
            
            NexusBlockHandle handle;
            handle.generation_id = block->get_generation_id();
            handle.mapped_data = block->get_mapped_data();
            handle.seq_len = block->get_header()->seq_len;
            handle.file_size = block->get_file_size();
            return handle;
        }, nb::arg("atb_filepath"))
        .def("prefetch", &NexusBlockCache::prefetch, nb::arg("atb_filepath"))
        .def("hazard_guard", [](NexusBlockCache& cache, uint32_t tool_id) {
            return cache.hazard_guard(tool_id);
        }, nb::rv_policy::move, nb::arg("tool_id"));

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
        apply_relative_rope_shift(
            k_data.data(),
            seq_len, n_head_kv, d_head, delta_pos,
            freq_base, freq_scale, scaling_type,
            ext_factor, beta_fast, beta_slow, n_ctx_orig
        );
    });

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
        .def_prop_ro("scent_tokens", [](const NexusSemanticSLB::SearchResult& res) {
            std::vector<int32_t> tokens(SCENT_TOKENS);
            std::memcpy(tokens.data(), res.scent_tokens, SCENT_TOKENS * sizeof(int32_t));
            return tokens;
        });

    // Semantic SLB bindings
    nb::class_<NexusSemanticSLB>(m, "NexusSemanticSLB")
        .def(nb::init<size_t>(), nb::arg("dim"))
        .def("register_tool", [](NexusSemanticSLB& slb, uint32_t tool_id, nb::ndarray<float, nb::c_contig, nb::device::cpu> vec, const std::vector<int32_t>& scent) {
            if (scent.size() != SCENT_TOKENS) {
                throw std::runtime_error("Scent size must be exactly SCENT_TOKENS");
            }
            std::span<const float> vec_span(vec.data(), vec.size());
            std::span<const int32_t, SCENT_TOKENS> scent_span(scent.data(), SCENT_TOKENS);
            slb.register_tool(tool_id, vec_span, scent_span);
        }, nb::arg("tool_id"), nb::arg("vector"), nb::arg("scent"))
        .def("search", [](const NexusSemanticSLB& slb, nb::ndarray<float, nb::c_contig, nb::device::cpu> query, size_t top_k) {
            nb::gil_scoped_release release;
            std::span<const float> query_span(query.data(), query.size());
            return slb.search(query_span, top_k);
        }, nb::arg("query"), nb::arg("top_k") = 3)
        .def("evaluate_slb", [](const NexusSemanticSLB& slb, nb::ndarray<float, nb::c_contig, nb::device::cpu> query, size_t top_k) {
            nb::gil_scoped_release release;
            std::span<const float> query_span(query.data(), query.size());
            return slb.search(query_span, top_k);
        }, nb::arg("query"), nb::arg("top_k") = 3)
        .def_prop_ro("dim", &NexusSemanticSLB::get_dim)
        .def_prop_ro("count", &NexusSemanticSLB::get_count);

    // Orchestrator bindings (MMU only — no grammar state)
    nb::class_<NexusOrchestrator>(m, "NexusOrchestrator")
        .def("__init__", [](NexusOrchestrator* self, nb::object ctx_obj, NexusSemanticSLB& slb, NexusRadixFSM& fsm, uint32_t base_pos,
                            float speculative_threshold, float speculative_margin, size_t max_pinned_bytes) {
            llama_context* ctx = resolve_context(ctx_obj);
            new (self) NexusOrchestrator(ctx, &slb, &fsm, base_pos, speculative_threshold, speculative_margin, max_pinned_bytes);
        }, nb::arg("ctx"), nb::arg("slb"), nb::arg("fsm"), nb::arg("base_pos"),
           nb::arg("speculative_threshold") = 0.88f, nb::arg("speculative_margin") = 0.05f,
           nb::arg("max_pinned_bytes") = 16ULL * 1024 * 1024 * 1024)
        .def("register_tool_path", &NexusOrchestrator::register_tool_path, nb::arg("tool_id"), nb::arg("path"))
        .def("preload_tool", &NexusOrchestrator::preload_tool, nb::arg("tool_id"), nb::arg("path"))
        .def("route_and_splice", [](NexusOrchestrator& self, const std::vector<int32_t>& user_query_tokens, const std::vector<float>& query_embedding, uint32_t n_past, llama_seq_id seq_id) {
            nb::gil_scoped_release release;
            return self.route_and_splice(user_query_tokens, query_embedding, n_past, seq_id);
        }, nb::arg("user_query_tokens"), nb::arg("query_embedding"), nb::arg("n_past"), nb::arg("seq_id") = 0)
        .def("release_hazard", &NexusOrchestrator::release_hazard, nb::arg("seq_id") = 0)
        .def("get_cache", &NexusOrchestrator::get_cache);
}
