# Nexus v1.1 Canonical — Environment Readiness Report

**Auditor pass:** 2026-06-20 · **Repo HEAD:** `3ea7441` · **Host:** Mustafas-MacBook-Pro (live, Apple Silicon)

Unlike the prior `docs/AUDIT_2026-06-20.md` pass (run in a Linux sandbox with no weights → regeneration BLOCKED), **this pass ran on the original Apple-Silicon host and the full benchmark suite executed.**

## Probe results

| Probe | Command | Result |
|---|---|---|
| OS / arch | `uname -a` | Darwin 25.5.0 arm64 (T6041 / Apple Silicon) — **PASS** |
| Git SHA | `git rev-parse HEAD` | `3ea7441ef7f7834475e063df56ceec457f6070ba` — **PASS** |
| Python (venv) | `.venv/bin/python --version` | 3.14.4 — **PASS** |
| `llama_cpp` (venv) | `import llama_cpp` | **0.2.83 — PASS** (system python3 fails; **must use `.venv/bin/python`**) |
| Compiled ext | `import nexus_fsm_ext` | **PASS** — `nexus_fsm_ext.cpython-314-darwin.so`, exports NexusOrchestrator, NexusRadixFSM, NexusSemanticSLB, … |
| `nexus_agent` | `import nexus_agent` | **PASS** (with `PYTHONPATH=build:src:test`) |
| Main model | Qwen2.5-14B-Instruct Q4_K_M | **PASS** — 3 GGUF shards in `/Volumes/AI_SSD/models/Qwen2.5-14B-Instruct-GGUF/` (model_hash `a09ea5e7…`) |
| Embedding model | nomic-embed-text-v1.5 f16 | **PASS** — HF cache `~/.cache/huggingface/hub/models--nomic-ai--nomic-embed-text-v1.5-GGUF/.../nomic-embed-text-v1.5.f16.gguf` |
| CE checkpoint | `results/tool_cross_encoder_finetuned_v3/` | **PASS** — `model.safetensors` (90 MB) present |
| `.atb` fixtures | `results/phaseA_tool_match_work/tool_*.isolated.atb` | **PASS** — 10 fixtures |
| llama.cpp build SHA | from artifacts | `cb2463bb9f093d7bcaf3194988a8251234fabb69` — **identical to committed paper run** |
| Gateway harness | `test/bench_gateway_e2e.py --query-limit 3` | **PASS** — ran, Path-A splicer telemetry fired |
| L0 microbench | `build/bench_phase28_radix_prefix` | **PASS** — exit 2 (expected/accepted) |
| Calibration | `scripts/calibrate_margin_threshold.py` | **PASS** — bit-exact reproduction |
| Recall analysis | `scripts/analyze_recall_misses.py` | **PASS** |
| E2E baseline | `test/bench_e2e.py` | **PASS** (slow: ~2.4 min/query across 14 arms) |

## Library versions
- numpy 2.4.6, llama_cpp 0.2.83, sentence-transformers 5.5.1, Python 3.14.4

## Footguns discovered (non-blocking)
1. **System `python3` cannot import `llama_cpp`** — all runs MUST use `.venv/bin/python`. The prior audit's "llama_cpp ImportError → BLOCKED" was a venv-selection artifact, not a true blocker.
2. **`nexus_retrieval.DEFAULT_RERANK_MARGIN` silently falls back to 0.028** when `src/` is not on `sys.path` (the calibrated 0.01365 lives in `src/nexus_calibration.py`). Real harnesses set the path; ad-hoc imports do not. This is a latent correctness hazard.
3. **`build/bench_phase28_radix_prefix` default model resolves to a local `gemma4` ollama blob** via `scripts/resolve_ollama_gguf.py` (`gemma4:31b-cloud`). All canonical L0 runs passed `--model` Qwen explicitly to avoid this.
4. **`bench_e2e.py` / `bench_gateway_e2e.py` stdout is block-buffered through pipes** — set `PYTHONUNBUFFERED=1` to see progress; absence of stdout ≠ hang.
