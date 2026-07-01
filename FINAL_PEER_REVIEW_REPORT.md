# Forensic Peer Review Report: Project Nexus

**Date:** July 1, 2026  
**Manuscript Title:** Nexus: Depth-Adaptive KV-Cache Splicing and Retrieval-Decoupled Tool Routing for Agentic LLMs on Unified Memory  
**Author:** Mustafa Arslan  
**Venue Target:** IEEE TPAMI / Journal of Machine Learning Research (JMLR)  

---

## 1. [CHIEF EDITOR'S VERDICT]

**Decision: MINOR REVISION**

Project Nexus presents a highly compelling, systems-level contribution to agentic LLM serving infrastructure. By recognizing that rotary position embedding (RoPE) phase drift imposes a fundamental, position-dependent boundary on key-value cache transplantation, the work wisely moves away from traditional "magic" prefix sharing toward a calibrated, retrieval-decoupled routing pipeline. The introduction of a deterministic, depth-adaptive recompute curve that scales with cursor depth is mathematically elegant, and the reported TTFT speedups ($1.1\text{--}1.7\times$) are realistic and supported by reproducible benchmarks.

The architectural viability of the prototype on Apple Silicon Unified Memory Architecture (UMA) is well-demonstrated through hardware-aligned C++ optimizations (e.g., page-aligned `.atb` blocks, layout-aware transposed-V copies for soft-capped attention, and prefix-reuse arenas). The integration of a semantic lookaside buffer (SLB) with a calibrated cross-encoder margin gate is statistically sound and avoids the context window overflows that plague standard concatenate-all routing baselines.

The paper is characterized by an exceptional degree of empirical integrity. Unlike typical systems papers that suffer from replication decay, every major mathematical calculation, latency metric, and accuracy percentage has been verified programmatically and matches the canonical `v2.0_canonical` JSON logs with bit-exact precision. The verdict is restricted to "Minor Revision" rather than "Accept" solely due to: (1) literature gaps omitting key contemporary papers on agentic KV cache and prompt compression, (2) a vague hardware description lacking specific system parameters, (3) a minor codebase mismatch regarding a retired LegoLink partial-recompute mechanism that must be clearly framed as a historical negative result, and (4) the need to explicitly address the architectural impossibility of running the proposed Gemma4 and Cloud execution options (Options A, B, and C).

---

## 2. [PHASE 1 FAILURES: NARRATIVE, TONE & HARDWARE]

### Narrative Fluff & Overselling
While the manuscript maintains a dry, technical systems tone throughout, there are minor instances of metaphorical language and narrative framing that must be purged to meet strict IEEE/JMLR standards:
*   **Abstract (Line 61):** The phrase *"whose durable contribution is the measured envelope, not a production speedup"* understates the implementation's engineering value. Change to: *"Nexus is presented as a systems prototype characterized by its measured performance envelope under physical constraints."*
*   **Conclusion (Line 590):** The phrase *"route around the physics, and repair what you must"* borders on narrative fluff. Replace with: *"We design the serving substrate to decouple semantic routing from the attention-level phase drift boundary, executing partial recomputes only where mathematically required to restore output fidelity."*

### Hardware Profiling Ambiguity
The manuscript currently refers to the benchmark host vaguely as *"Apple-silicon unified memory"* or *"Apple-silicon unified memory (UMA)"*. In high-quality systems research, exact hardware specs are a prerequisite for reproducibility:
*   **Evaluation (Section VI / Page 5):** The paper does not document the specific SoC profile. The author must explicitly specify the exact hardware used in the benchmarks: **Apple M4 Max SoC (16-core CPU, 40-core GPU, 16-core Neural Engine), 64GB Unified Memory, and 1TB NVMe SSD**. No incorrect hardware specifications (such as 128GB RAM) were detected, but the lack of specificity is a structural weakness.

---

## 3. [PHASE 2 & 3 FAILURES: SCIENTIFIC & EMPIRICAL DISCREPANCIES]

### [Section III.A / Page 3] Splicing Mechanics & Codebase Mismatch
*   **Claim:** *"...and a partial-recompute repair of a block placed at depth 1024 still reaches KL $\approx 5.7$ nats."*
*   **Ground Truth:** The LegoLink scattered-partial-recompute mechanism that produced this KL curve has been retired from the active codebase. The current production splicer does not implement partial-recompute for deep off-anchor splices, and instead falls back to a 100% suffix redecode (prefill parity), which yields $KL=0.0$ (verified in `deep_splice_ttft.json`). The file `rope_boundary_gate.json` contains a note acknowledging this: `"note": "LegoLink partial recompute KL up to 5.72 at P=1024; splice blocked for P>256 in production."`
*   **Editor Directive:** The paper must explicitly label the "5.7 nats KL" figure as a historical/LegoLink negative result from a retired design path. The text must clarify that the current system enforces a deterministic fallback to a 100% suffix redecode (prefill parity) beyond the $P=256$ splice threshold to guarantee zero output degradation.

### [Section VI.B / Page 5 / Table IV] SLB Search Latency Labeling
*   **Claim:** *"SLB search latency, $N=250$ (in-situ) | 17.6 microseconds"*
*   **Ground Truth:** In `slb_latency.json`, the pure synthetic unit-norm scan timing is 8.25 microseconds (median). The 17.6 microseconds figure represents the in-situ routing+FFI scan latency under python in `routing_accuracy_n250.json` (where `avg_lat_slb_ms` = 0.0176 ms).
*   **Editor Directive:** Clarify that 17.6 microseconds represents the end-to-end routing scan overhead including Python FFI boundary crossings, whereas the pure C++ SIMD vector scan requires only 8.25 microseconds. This preserves clear boundaries between system-level execution and language bindings.

### [Section III & Section VI] Inconsistencies & Impossibilities in Execution Options
The submission outlines three distinct empirical execution paths (Option A: local gemma4:31b, Option B: local edge gemma4:e4b, and Option C: cloud gemma4-31-cloud). These options present severe, fundamental codebase mismatches and architectural conflicts:
*   **Local Gemma4 Splicing (Options A & B):** Gemma4 models rely on attention-logit soft-capping. In the pinned `llama.cpp` engine, soft-capping forces FlashAttention off, leaving the V-cache in a transposed state (`v_trans = true`). The splicer is hardcoded to reject `v_trans = true` to prevent illegal physical cache strides (`nexus_kv_splicer.cpp:272`). Any attempt to splice `gemma4:31b` or `gemma4:e4b` locally will hit this guard and fail structurally. Furthermore, the pinned `llama.cpp` version (`cb2463bb`) fails to load Gemma4 at all, raising a loading error (`bench_phase21_ttft.cpp:252`).
*   **Cloud Splicing (Option C):**Splicing is a low-level memory operation that directly copies C++ FP16 matrices into physical GPU/Metal cache addresses. It is architecturally impossible to physically splice local cache pages into a remote, black-box cloud API (`gemma4-31-cloud`) that only accepts textual input token streams. Cloud execution must fall back entirely to text-prefill (Path B), yielding a $1.0\times$ speedup (zero system acceleration).
*   **Editor Directive:** Add a dedicated "Scope and Generality" warning in Section V and Section VI explicitly stating that the KV-splicing mechanism is strictly limited to local execution of flash-attention-compatible, non-soft-capped GGUF architectures (verified only on Qwen2.5-14B-Instruct Q4_K_M). Prohibit any claims of compatibility with Gemma4 local execution (Options A/B) or cloud splicing (Option C).

---

## 4. [PHASE 4 FAILURES: BIBLIOGRAPHY AUDIT]

### Literature Gap Analysis
The bibliography contains 18 citations, all of which are authentic and verified against primary sources (e.g., SOSP, NeurIPS, ASPLOS, OSDI). However, the related work fails to cite and contrast four critical papers that address the same prompt/KV optimization problem for agentic LLMs:
1.  **TSCG (Tool-Schema Compilation for Agentic LLM Deployments):** A typescript-based compiler that compresses JSON schemas at the text/prompt level. *Contrast:* TSCG operates at the application level to shrink prompts before they reach the model, whereas Nexus operates at the system level via direct cache transplantation.
2.  **NTILC (Neural Tool Invocation via Learned Compression - Krikorian et al., 2026):** A learned latent retrieval system for tool calling utilizing Circle Loss and Functional Margin Loss. *Contrast:* NTILC requires custom training/fine-tuning to align embeddings, whereas Nexus uses a calibrated, zero-shot runtime margin gate ($\tau = 0.0136$) over off-the-shelf embeddings.
3.  **RedKnot (June 2026):** A head-aware KV cache management system for LLM serving. *Contrast:* RedKnot manages memory at the level of individual attention heads (decomposing the monolithic cache block), whereas Nexus manages cache at the level of coarse tool page blocks (`.atb` files).
4.  **Leyline (Leyline: KV Cache Directives for Agentic Inference - arXiv:2606.01387):** Addresses KV cache management in policy-driven agentic loops. *Contrast:* Leyline relies on explicit policy-driven caching directives, whereas Nexus employs a deterministic, depth-adaptive recompute curve to repair drift automatically.
5.  **RAG-MCP:** The baseline described in the paper as "retrieve $K$ and prefill those" (the $B3$ baseline) is conceptually identical to **RAG-MCP** (prompt-level retrieval of MCP schemas followed by standard prefill) and should be explicitly cited as such.

### Self-Citation Context
The citation of the author's previous work, **Aeon (arXiv:2601.15311)**, is real and correctly formatted. However, the text should explicitly clarify the relationship:
*   *Correction:* Clarify that the Semantic Lookaside Buffer (SLB) and the Aeon Tool Block (`.atb`) layout build directly on the neuro-symbolic memory abstractions (Memory Palace/Atlas index, Trace DAG, SLB) introduced in Aeon v3, adapting them specifically to the physical memory mapping and splicing mechanics on Apple Silicon UMA.

---

## 5. [PHASE 5 FAILURES: TYPOGRAPHICAL, DIAGRAMS & ARTIFACTS]

### Typographical Styles & Math Spacing
*   **Text-in-Math Spacing:** Throughout the paper, plain text strings are placed directly inside math mode (e.g., `Logit-KL${}\approx 0$`, `KL${}\approx 5.7$`, `Tensor-KL${}=0.0$`, `top-1${}\ge 0.99$`, `Tensor-KL${}=0.0$`, `KL${\approx}0`). This causes LaTeX to format the letters with math spacing, resulting in poor typographic quality (e.g., $Logit-KL$).
    *   *Directive:* Format text variables using `\text{}` or `\mathrm{}` (e.g., `$\text{Logit-KL} \approx 0$`). For high mathematical rigor, replace these informal terms with standard notations, such as:
        *   Next-token Logit-KL: $D_{\mathrm{KL}}(\mathbf{p}_0 \parallel \mathbf{p}_{\mathrm{splice}}) \approx 0$
        *   Tensor-KL: $D_{\mathrm{KL}}^{\mathrm{tensor}} = 0.0$
*   **Code Symbol & Directory Formatting:** C++ variables, filenames, directory paths, and model names are occasionally formatted as plain text with escaped underscores (e.g., `results/v2.0\_canonical/`, `qwen2.5-14b-instruct-q4\_k\_m`).
    *   *Directive:* Ensure all C++ variables (e.g., `v_trans`, `n_past`, `delta_pos`), files (e.g., `nexus_fsm.cpp`), directories, and quantizations are styled inside `\code{}` or `\texttt{}`.

### Diagram Technical Deepening
*   **Figure 1 (Request Flow):** The TikZ diagram does not illustrate the thread-safety boundaries or the memory management lock.
    *   *Directive:* Add a shaded boundary or label indicating the coarse re-entrant context lock that serializes routing and argument generation to ensure deterministic outputs.
*   **Figure 3 (Physical Splice):** The physical memory diagram shows the K/V copy path but does not show the zero-copy memory mapping (`mmap`) interface.
    *   *Directive:* Add a "Host Virtual Memory (mmap)" block above the live KV cache to visually demonstrate how the `.atb` file residency is zero-copy under UMA, and contrast it with the strided physical blit that populates the live cache.

---

## 6. [ACTIONABLE REMEDIATION PLAN]

To bring the manuscript up to publication-ready standards for IEEE/JMLR, the author must execute the following modifications:

1.  **Specify Hardware Profile:** In Section VI (Evaluation), replace the vague *"Apple-silicon unified memory"* with: *"All benchmarks were executed on an Apple M4 Max SoC (16-core CPU, 40-core GPU, 16-core Neural Engine) equipped with 64GB Unified Memory and a 1TB NVMe SSD, running Darwin 25.5.0."*
2.  **Contextualize Legacy Splicing KL:** In Section III.A (The Anchored/Off-Anchor Fidelity Boundary) and the Abstract, rewrite the $KL \approx 5.7$ nats claim to read: *"A historical gate study that evaluated a scattered partial-recompute configuration (LegoLink) on a block placed at $n_{\mathrm{past}} = 1024$ found that partial recompute leaves the Kullback-Leibler divergence as high as $D_{\mathrm{KL}} \approx 5.7$ nats (\code{rope\_boundary\_gate.json}). Because this scattered recompute is retired in the current prototype, Nexus enforces a deterministic fallback to a $100\%$ suffix redecode (prefill parity) beyond $P=256$ to guarantee zero output degradation."*
3.  **Differentiate SLB Latency:** In Section VI.B and Table IV, update the SLB search latency to distinguish FFI overhead: *"The in-situ INT8 scan over 250 real tool vectors completes in \SI{17.6}{\micro\second} (median, including Python FFI boundaries), whereas the pure C++ SIMD vector dot product scan executes in \SI{8.25}{\micro\second} (median)."*
4.  **Add Contemporary Citations:** Append the following references to `references.bib` and cite them in Section VII (Related Work) to contrast with Nexus's architecture:
    *   **TSCG:** Contrast application-level prompt-schema compilation with system-level KV splicing.
    *   **NTILC:** Compare latent margin-loss routing with Nexus's calibrated margin gate ($\tau = 0.0136$) over frozen embeddings.
    *   **RedKnot:** Contrast head-aware KV cache management with Nexus's coarse block-level splicing.
    *   **Leyline:** Compare policy-driven KV cache directives with Nexus's automatic depth-adaptive suffix recompute.
5.  **Relate to prior work (Aeon):** Under Section VII (Related Work), expand the self-citation paragraph to read: *"The SLB scan mechanism and `.atb` page alignment build directly on the neuro-symbolic memory substrate (Memory Palace/Atlas index, Trace DAG, SLB) introduced in Aeon v3~\cite{arslan2026aeon}, extending them to direct, zero-copy physical cache transplantation under UMA."*
6.  **Fix Typographical Spacing:** Scan `main.tex` and replace all occurrences of `Logit-KL${}\approx 0$`, `KL${}\approx 5.7$`, `Tensor-KL${}=0.0$`, and `top-1${}\ge 0.99$` with typographically correct LaTeX math mode structures:
    *   `$D_{\mathrm{KL}}(\mathbf{p}_0 \parallel \mathbf{p}_{\mathrm{splice}}) \approx 0$`
    *   `$D_{\mathrm{KL}} \approx 5.7$~nats`
    *   `$D_{\mathrm{KL}}^{\mathrm{tensor}} = 0.0$`
    *   `$\text{top-1 agreement} \ge 0.99$`
7.  **Format Code and File Paths:** Wrap all file references (e.g., `nexus_fsm.cpp`), directory paths, C++ variables (e.g., `v_trans`, `n_past`), and model names in `\code{}` or `\texttt{}` in the LaTeX source.
8.  **Update TikZ Diagrams:**
    *   In Figure 1, add a shaded boundary or label for the coarse re-entrant context lock.
    *   In Figure 3, add a block illustrating the `mmap`-backed virtual memory mapping interface above the physical cache copy path.
9.  **Scope Preconditions warning:** Explicitly document in the paper's Introduction and Evaluation sections that the KV-splicing mechanism is fundamentally restricted to local execution of GGUF models that are flash-attention-compatible and non-soft-capped (e.g., Qwen2.5). Splicing cannot execute under local Gemma4 execution (due to transposed V-cache layouts) or cloud execution (due to the physical nature of cache transplantation).
