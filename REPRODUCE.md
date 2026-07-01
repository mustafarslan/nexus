# Nexus v2.0 Replication Guide

This guide details the system requirements, compile steps, and script execution parameters required to reproduce the quantitative and qualitative findings in the Nexus paper.

---

## 1. System Requirements

All benchmarks were run on:
- **Hardware**: Apple M4 Max SoC (16-core CPU, 40-core GPU, 16-core Neural Engine), 64 GB Unified Memory, 1 TB NVMe SSD.
- **OS**: macOS/Darwin 25.5.0 (arm64).
- **Stack**: `llama.cpp` at git commit `cb2463bb`.

The C++ splicing mechanics require unified memory access and direct cache transplantation, which are currently restricted to local GGUF execution on macOS/Metal. 

---

## 2. Environment Setup

### 2.1 Dependencies
Ensure you have Xcode Command Line Tools installed (for `clang++` and `make`). Then set up a virtual environment and install dependencies:

```bash
# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install exact pinned requirements
pip install -r requirements.txt
```

### 2.2 Model Files
Download the required model weights in GGUF format:
1. **Primary LLM**: `Qwen2.5-14B-Instruct-Q4_K_M.gguf` (hash `a09ea5e7`).
2. **Retrieval Embedder**: `nomic-embed-text-v1.5.f16.gguf`.

Place these GGUF files in your local models path.

---

## 3. C++ Compilation

Build the C++ components (FSM radix trie, INT8 SLB scan, buddy allocator, and KV-cache compiler bindings):

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
cd ..
```

This will output the compiled binaries (`nexus_kv_compiler`, `verify_atb`) and the shared library extension `nexus_fsm_ext` used by the Python agent.

---

## 4. Reproducing Results

Always set `PYTHONPATH` when running benchmarks:
```bash
export PYTHONPATH=build:src:test
```

### 4.1 Table I (Reference-Free Drift Gate Spearman Correlation)
Verify the K-variance proxy fails to rank-correlate with per-head drift (Mean Spearman $\rho = 0.193$).
```bash
python test/profile_head_drift.py --model /path/to/Qwen2.5-14B-Instruct-Q4_K_M.gguf
```
*Outputs raw data and computes rank correlations across depths and contexts.*

### 4.2 Table II & Figure 3 (Deep-Splice TTFT and Speedups)
Measure the deep-splice latency curves and verify that the recompute schedule scales up to prefill parity at deep context ($KL \approx 0$).
```bash
# Run for curve K=4 (default)
python test/bench_v2_capstone.py --model /path/to/Qwen2.5-14B-Instruct-Q4_K_M.gguf --trials 15 --output results/deep_splice_ttft_k4.json

# Run for curve K=16 (tuned)
# Edit the default curve config or specify the parameter in code/arguments if supported.
```

### 4.3 Table III & Figure 4 (Routing Accuracy & Scale)
Benchmark the SLB scan and gate accuracies as the tool registry scales up to $N=250$ tools.
```bash
python test/bench_routing_accuracy.py \
    --model /path/to/Qwen2.5-14B-Instruct-Q4_K_M.gguf \
    --embed-model /path/to/nomic-embed-text-v1.5.f16.gguf \
    --tool-sizes "10 50 100 250" \
    --runs 1
```

### 4.4 Table III (Radix Cache & Micro-latency)
Measure the L0 warm-tier radix cache copy latencies (expected median $\approx 3.04 \mu s$):
```bash
# Run radix benchmarks
./build/bin/bench_phase28_radix_prefix
```

### 4.5 Figure 1 (Next-token Divergence vs Placement Offset)
Verify next-token KL divergence for contiguous-suffix splices remains in the $10^{-2}$ nat band:
```bash
python test/bench_dkl_sweep.py --model /path/to/Qwen2.5-14B-Instruct-Q4_K_M.gguf
```
*Requires a compiled target `.atb` block to sweep.*
