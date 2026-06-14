#!/usr/bin/env bash
# Phases F–J remediation benchmark runner (subset; full ablation may take hours).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${NEXUS_MODEL:-/Volumes/AI_SSD/models/Qwen2.5-14B-Instruct-GGUF/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf}"
EMBED="${NEXUS_EMBED_MODEL:-}"
ITER="${NEXUS_ITERATIONS:-100}"

mkdir -p results

echo "=== Rebuild bindings (kv_cache_seq_cp) ==="
cmake --build build -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"

echo "=== F4 throughput gate ==="
python3 scripts/run_throughput_gate.py --model "$MODEL" || true

echo "=== F1 rebuild Phase E corpora + scale eval ==="
python3 scripts/build_phaseE_corpus.py
python3 test/bench_phaseE_scale.py

echo "=== G3 position study ==="
python3 test/bench_g3_position.py --model "$MODEL" || true

echo "=== Phase B quick ablation (F5 canary) ==="
python3 test/bench_phaseB_selective_recompute.py --quick --output results/bench_phaseB_selective_recompute_quick.json

echo "=== Phase C real KV (100 iter) ==="
python3 test/bench_phase21_real_kv.py --model "$MODEL" --iterations "$ITER"

echo "=== H1 query sets ==="
python3 scripts/build_scale_query_sets.py

echo "=== Phase D e2e (use NEXUS_QUERY_LIMIT=10 for smoke) ==="
python3 test/bench_e2e.py --model "$MODEL" --query-limit "${NEXUS_QUERY_LIMIT:-100}"

echo "=== I2/I3 scale E2E + Zipfian ==="
python3 test/bench_phaseE_e2e_scale.py --model "$MODEL" --query-limit "${NEXUS_QUERY_LIMIT:-20}"

echo "=== G4 gate ==="
python3 scripts/apply_g4_gate.py --input results/bench_phaseB_selective_recompute_full.json || \
  python3 scripts/apply_g4_gate.py --input results/bench_phaseB_selective_recompute_quick.json || true

echo "=== I1 storage economics (sample ATB) ==="
python3 scripts/quantize_atb.py --input results/phaseA_tool_match_work --output results/atb_q8_sidecars --glob "*.isolated.atb" || true

python3 scripts/plot_from_results.py
echo "Done."
