#!/usr/bin/env bash
# Launch L1 ablation + tail pipeline (single-instance lock).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
LOCK="$ROOT/results/.ko_pipeline.lock"

exec 9>"$LOCK"
if ! flock -n 9 2>/dev/null; then
  # macOS: flock may be unavailable; fall back to pidfile
  if [[ -f "$ROOT/results/.l1.pid" ]]; then
    old_pid="$(cat "$ROOT/results/.l1.pid" 2>/dev/null || true)"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "L1 already running (PID $old_pid)"
      exit 0
    fi
  fi
fi

if pgrep -f "bench_phaseB_suffix_recompute.py --case-limit 50" >/dev/null 2>&1; then
  echo "L1 already running"
else
  python3 test/bench_phaseB_suffix_recompute.py --case-limit 50 --force \
    >> results/bench_phaseB_suffix_recompute_full.log 2>&1 &
  echo $! > results/.l1.pid
  echo "Started L1 PID $(cat results/.l1.pid)"
fi

if pgrep -f "run_phases_ko_remaining.sh" >/dev/null 2>&1; then
  echo "Tail pipeline already running"
else
  bash scripts/run_phases_ko_remaining.sh >> results/run_phases_ko_remaining.log 2>&1 &
  echo $! > results/.tail.pid
  echo "Started tail PID $(cat results/.tail.pid)"
fi
