#!/bin/bash
# Sequential v2.0 GPU-bench driver. Runs one 14B bench at a time (concurrent 14B runs
# contend for UMA and inflate absolutes). Waits for the in-flight routing bench first.
set -u
cd /Volumes/AI_SSD/Projects/nexus
M14=/Volumes/AI_SSD/models/Qwen2.5-14B-Instruct-GGUF/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf
RAW=results/v2.0_canonical/raw
PY=.venv/bin/python
log(){ echo "[$(date +%H:%M:%S)] $*" >> $RAW/_driver.log; }

log "driver start; waiting for routing bench to finish"
while ps aux | grep -q '[b]ench_routing_accuracy'; do sleep 15; done
log "routing done; running capstone (deep-splice TTFT, K=4 & K=16, trials=15)"

$PY test/bench_v2_capstone.py --model "$M14" \
    --p-starts "256 512 1024 2048" --full-mults "4 16" --trials 15 \
    --output $RAW/deep_splice_ttft.json >> $RAW/_capstone_run.log 2>&1
log "capstone exit=$?"

log "running sidecar hybrid accuracy bench"
$PY test/bench_sidecar_accuracy.py --limit 30 --cons-n 30 \
    --out-dir $RAW/sidecar >> $RAW/_sidecar_run.log 2>&1
log "sidecar exit=$?"
log "driver done"
