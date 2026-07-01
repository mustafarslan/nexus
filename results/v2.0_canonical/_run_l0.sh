#!/bin/bash
set -u
cd /Volumes/AI_SSD/Projects/nexus
M14=/Volumes/AI_SSD/models/Qwen2.5-14B-Instruct-GGUF/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf
RAW=results/v2.0_canonical/raw
for s in 0 1 2 3 4 5 6 7 8 9; do
  ./build/bench_phase28_radix_prefix --model "$M14" --seed $s --output $RAW/_l0_seed_$s.json >>$RAW/_l0_run.log 2>&1
done
.venv/bin/python - <<'PY'
import json, glob, statistics
xs=[json.load(open(f)) for f in sorted(glob.glob("results/v2.0_canonical/raw/_l0_seed_*.json"))]
def g(d,*ks):
    for k in ks:
        if k in d: return d[k]
    return None
hits=[g(d,"hit_rate","hit_rate_pct") for d in xs]
cps =[g(d,"copy_p50_us","copy_us_p50","copy_p50") for d in xs]
hits=[h for h in hits if h is not None]; cps=[c for c in cps if c is not None]
out={"benchmark":"l0_radix_warm_prefix","model":"Qwen2.5-14B-Instruct Q4_K_M","n_seeds":len(xs),
     "hit_rate_mean":round(statistics.mean(hits),4) if hits else None,
     "hit_rate_min":min(hits) if hits else None,"hit_rate_max":max(hits) if hits else None,
     "copy_p50_us_median":round(statistics.median(cps),3) if cps else None,
     "per_seed":xs}
open("results/v2.0_canonical/raw/l0_radix.json","w").write(json.dumps(out,indent=2)+"\n")
print("L0 aggregate: hit_mean=%.3f copy_p50_median=%.3f us over %d seeds"%(
    out["hit_rate_mean"] or 0, out["copy_p50_us_median"] or 0, len(xs)))
PY
echo "l0 done"
