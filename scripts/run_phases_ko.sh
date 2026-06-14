#!/usr/bin/env bash
# Phases K–O validity restoration runner.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${NEXUS_MODEL:-/Volumes/AI_SSD/models/Qwen2.5-14B-Instruct-GGUF/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf}"
ITER="${NEXUS_ITERATIONS:-100}"
QUERY_LIMIT="${NEXUS_QUERY_LIMIT:-100}"
SMOKE="${NEXUS_SMOKE:-0}"
FORCE="${NEXUS_FORCE:-}"

mkdir -p results results/smoke

SMOKE_ARGS=()
FORCE_ARGS=()
if [[ "$SMOKE" == "1" ]]; then
  SMOKE_ARGS=(--smoke)
  QUERY_LIMIT=5
fi
if [[ "$FORCE" == "1" ]]; then
  FORCE_ARGS=(--force)
fi

echo "=== K: Rebuild (llama-bench + nexus_kv_compiler --base-pos) ==="
cmake --build build -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)" --target nexus_kv_compiler llama-bench 2>/dev/null || \
  cmake --build build -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"

echo "=== K3: F4 throughput gate ==="
python3 scripts/run_throughput_gate.py --model "$MODEL" "${FORCE_ARGS[@]}"

echo "=== K4: Phase C real KV (${ITER} iter) ==="
python3 test/bench_phase21_real_kv.py --model "$MODEL" --iterations "$ITER" "${SMOKE_ARGS[@]}" "${FORCE_ARGS[@]}"

echo "=== L1: Suffix ablation ==="
if [[ "$SMOKE" == "1" ]]; then
  python3 test/bench_phaseB_suffix_recompute.py --quick --smoke "${FORCE_ARGS[@]}"
else
  python3 test/bench_phaseB_suffix_recompute.py --case-limit 50 "${FORCE_ARGS[@]}"
fi

echo "=== L2: G4 gate on valid suffix artifact ==="
python3 scripts/apply_g4_gate.py --input results/bench_phaseB_suffix_recompute_full.json || true

echo "=== L3: G3 root-cause experiments ==="
python3 test/bench_g3_experiments.py --model "$MODEL" "${FORCE_ARGS[@]}" || true

echo "=== M1: Recall miss analysis + rerank ==="
python3 scripts/analyze_recall_misses.py --rerank --model "$MODEL" "${FORCE_ARGS[@]}" || true

echo "=== M2: Phase D e2e (${QUERY_LIMIT} queries) ==="
python3 test/bench_e2e.py --model "$MODEL" --query-limit "$QUERY_LIMIT" "${SMOKE_ARGS[@]}" "${FORCE_ARGS[@]}"

echo "=== M3: N1m multi-splice fidelity ==="
python3 test/bench_n1m_fidelity.py --model "$MODEL" "${SMOKE_ARGS[@]}" "${FORCE_ARGS[@]}" || true

echo "=== N1: Distractor re-embed ==="
python3 scripts/reembed_phaseE_distractors.py "${FORCE_ARGS[@]}" || true

echo "=== N2: TTFT vs N + Zipfian (52 schemas) ==="
python3 test/bench_phaseE_e2e_scale.py --model "$MODEL" --query-limit "${QUERY_LIMIT}" "${SMOKE_ARGS[@]}" "${FORCE_ARGS[@]}"

echo "=== N3: Quant fidelity through splice ==="
python3 test/bench_quant_fidelity.py --model "$MODEL" "${SMOKE_ARGS[@]}" "${FORCE_ARGS[@]}"

echo "=== O: Tombstone superseded + summary.csv ==="
python3 scripts/tombstone_superseded.py "${FORCE_ARGS[@]}" || true
python3 scripts/plot_from_results.py

echo "Phases K-O complete."
