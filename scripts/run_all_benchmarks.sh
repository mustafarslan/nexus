#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${ROOT}/.venv/bin/python3"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

MODEL="${NEXUS_MODEL:-}"
EMBED_MODEL="${NEXUS_EMBED_MODEL:-}"
ITERATIONS="${NEXUS_ITERATIONS:-100}"
WARMUP="${NEXUS_WARMUP:-5}"
SEED="${NEXUS_SEED:-1337}"

mkdir -p results

if [[ ! -x build/bench_phase21_ttft || ! -x build/bench_phase22_concurrent_hazard || ! -x build/nexus_kv_compiler ]]; then
  cmake --build build
fi

# Tombstone pre-fix artifacts before writing new results
"$PYTHON" scripts/tombstone_superseded.py || true

phase21_args=(--output results/bench_phase21_ttft.json --iterations "$ITERATIONS" --warmup "$WARMUP")
if [[ -n "$MODEL" ]]; then
  phase21_args+=(--model "$MODEL")
fi
./build/bench_phase21_ttft "${phase21_args[@]}"

./build/bench_phase22_concurrent_hazard \
  --output results/bench_phase22_concurrent_hazard.json \
  --seed "$SEED"

routing_args=(--output results/bench_routing_accuracy.json --seed "$SEED")
if [[ -n "$MODEL" ]]; then
  routing_args+=(--model "$MODEL")
fi
if [[ -n "$EMBED_MODEL" ]]; then
  routing_args+=(--embed-model "$EMBED_MODEL")
fi
"$PYTHON" test/bench_routing_accuracy.py "${routing_args[@]}"

if [[ -n "$MODEL" ]]; then
  "$PYTHON" scripts/load_mcp_zero_corpus.py --n-queries 1000 || true
  "$PYTHON" scripts/calibrate_rerank_gate.py --query-limit 100 || true

  e2e_args=(--output results/bench_e2e.json --query-limit 100 --force)
  e2e_args+=(--model "$MODEL")
  "$PYTHON" test/bench_e2e.py "${e2e_args[@]}"

  "$PYTHON" test/bench_dynamic_context.py --model "$MODEL" --query-limit 50 --output results/bench_dynamic_context.json

  gw_args=(--output results/bench_gateway_e2e.json --query-limit 100 --force)
  gw_args+=(--model "$MODEL")
  "$PYTHON" test/bench_gateway_e2e.py "${gw_args[@]}"

  "$PYTHON" test/bench_splice_fidelity.py \
    --model "$MODEL" \
    --output results/bench_splice_fidelity.json \
    --seed "$SEED"

  "$PYTHON" scripts/analyze_recall_misses.py --rerank --output results/recall_miss_analysis.json
  "$PYTHON" scripts/reembed_github_tools.py --output results/github_tool_embeddings.npz
  "$PYTHON" test/bench_phaseE_e2e_scale.py --force --cache-blocks "4,8,16,32,64" --lfu --pin-top-n 5
else
  echo "Skipping model-dependent benches: set NEXUS_MODEL to a GGUF path."
fi

"$PYTHON" scripts/generate_executive_summary.py --b3-baseline --plot-missing
"$PYTHON" scripts/plot_from_results.py --results-dir results --output results/summary.csv
