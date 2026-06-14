#!/usr/bin/env bash
# Run remaining K-O GPU benchmarks sequentially; skip steps with valid artifacts.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODEL="${NEXUS_MODEL:-/Volumes/AI_SSD/models/Qwen2.5-14B-Instruct-GGUF/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf}"
export PYTHONUNBUFFERED=1

wait_for_pid() {
  local pid="$1"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "Waiting for PID $pid to finish..."
    while kill -0 "$pid" 2>/dev/null; do sleep 30; done
  fi
}

wait_for_proc() {
  local pattern="$1"
  if pgrep -f "$pattern" >/dev/null 2>&1; then
    echo "Waiting for process matching: $pattern"
    while pgrep -f "$pattern" >/dev/null 2>&1; do sleep 30; done
  fi
}

artifact_ok() {
  local path="$1"
  local check="$2"
  python3 - "$path" "$check" <<'PY'
import json, sys
path, check = sys.argv[1], sys.argv[2]
try:
    data = json.load(open(path))
except (FileNotFoundError, json.JSONDecodeError):
    sys.exit(1)
cfg = data.get("config") or {}
extra = data.get("extra") or {}
if check == "l1":
    ok = cfg.get("n_cases", 0) >= 50 and len(data.get("records", [])) >= 50 * 72
elif check == "k4":
    ok = cfg.get("iterations", 0) >= 100
elif check == "m2":
    ok = (data.get("config") or {}).get("query_limit", 0) >= 100 or data.get("n_queries", 0) >= 100
elif check == "n2":
    tiers = cfg.get("tiers", "")
    ok = "N10000" in tiers and cfg.get("zipf_queries", 0) >= 500
elif check == "exists":
    ok = True
else:
    ok = False
sys.exit(0 if ok else 1)
PY
}

run() { echo "=== $* ==="; "$@" || echo "WARN: exited $? - $*"; }

skip_or_run() {
  local label="$1" path="$2" check="$3"
  shift 3
  if [[ -f "$path" ]] && artifact_ok "$path" "$check"; then
    echo "=== SKIP $label (valid artifact: $path) ==="
    return 0
  fi
  run "$@"
}

# Wait for any in-flight L1 from a parallel start
wait_for_proc "bench_phaseB_suffix_recompute.py"

echo "=== L1: suffix ablation ==="
skip_or_run "L1" "results/bench_phaseB_suffix_recompute_full.json" "l1" \
  python3 test/bench_phaseB_suffix_recompute.py --case-limit 50 --force

echo "=== L2: G4 gate ==="
if [[ -f results/bench_phaseB_suffix_recompute_full.json ]]; then
  run python3 scripts/apply_g4_gate.py || true
else
  echo "SKIP L2: missing L1 artifact"
fi

skip_or_run "L3" "results/bench_g3_experiments.json" "exists" python3 test/bench_g3_experiments.py --model "$MODEL"

echo "=== M1: recall miss analysis + rerank ==="
run python3 scripts/analyze_recall_misses.py --rerank --model "$MODEL" || true

skip_or_run "M2" "results/bench_e2e.json" "m2" \
  python3 test/bench_e2e.py --model "$MODEL" --query-limit 100 --force

skip_or_run "M3" "results/bench_n1m_fidelity.json" "exists" \
  python3 test/bench_n1m_fidelity.py --model "$MODEL" --cases 20 --force || true

skip_or_run "K4" "results/bench_phase21_ttft_real.json" "k4" \
  python3 test/bench_phase21_real_kv.py --model "$MODEL" --iterations 100 --bloat-every 10 --force || true

skip_or_run "N2" "results/bench_phaseE_e2e_scale.json" "n2" \
  python3 test/bench_phaseE_e2e_scale.py --model "$MODEL" \
    --tiers N100,N1000,N10000 --query-limit 20 --zipf-queries 500 \
    --cache-blocks 4,8,16,32 --schemas-dir results/phaseA_tool_match_work --force

skip_or_run "N3" "results/bench_quant_fidelity.json" "exists" \
  python3 test/bench_quant_fidelity.py --model "$MODEL" --force

run python3 scripts/tombstone_superseded.py || true
run python3 scripts/plot_from_results.py
echo "Remaining K-O benchmarks complete."
