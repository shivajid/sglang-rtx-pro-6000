#!/bin/bash
# HTTP ablation ladder against one Clef server process (runtime reconfiguration via /admin/config).
# Usage: bash sweep_clef.sh <url> <outfile> [configs...]
URL=${1:-http://10.10.0.71:8000}
OUT=${2:-/work/results/clef_ablation.jsonl}
shift 2
CONFIGS=${@:-A B C D F G G8 G4 G32}
cd /work

admin() {
  python - "$URL" "$1" <<'EOF'
import json, sys, urllib.request
url, cfg = sys.argv[1], sys.argv[2]
req = urllib.request.Request(url + "/admin/config", data=cfg.encode(), headers={"Content-Type": "application/json"})
print(urllib.request.urlopen(req, timeout=900).read().decode())
EOF
}

run() {  # tag
  python loadgen.py --url $URL --workload workloads/medium.jsonl --sweep 1,2,4,8,16,32,64 --per-worker 4 --min-num 24 --tag $1 --out $OUT
  python loadgen.py --url $URL --workload workloads/short.jsonl  --sweep 1,4,16,64 --per-worker 4 --min-num 24 --tag $1 --out $OUT
  python loadgen.py --url $URL --workload workloads/mixed.jsonl  --sweep 1,4,16,64 --per-worker 4 --min-num 24 --tag $1 --out $OUT
}

# A..D are runtime toggles; F (fused kernels) and E/G (FP8) are one-way, so keep this order.
for c in $CONFIGS; do
  case $c in
    A)   admin '{"head_impl":"reference","mask_mode":"mask","max_batch_size":1,"max_batch_tokens":16384}' ;;
    B)   admin '{"head_impl":"reference","mask_mode":"mask","max_batch_size":64,"max_batch_tokens":16384,"min_pad_eff":0.0}' ;;
    C)   admin '{"head_impl":"fast","mask_mode":"mask","max_batch_size":64,"max_batch_tokens":16384,"min_pad_eff":0.0}' ;;
    D)   admin '{"head_impl":"fast","mask_mode":"none","max_batch_size":64,"max_batch_tokens":16384,"min_pad_eff":0.8}' ;;
    D8)  admin '{"head_impl":"fast","mask_mode":"none","max_batch_size":64,"max_batch_tokens":8192,"min_pad_eff":0.8}' ;;
    D32) admin '{"head_impl":"fast","mask_mode":"none","max_batch_size":64,"max_batch_tokens":32768,"min_pad_eff":0.8}' ;;
    E)   admin '{"quant":"fp8_rowwise","head_impl":"fast","mask_mode":"none","max_batch_size":64,"max_batch_tokens":16384,"min_pad_eff":0.8}' ;;
    F)   admin '{"fused":"1","head_impl":"fast","mask_mode":"none","max_batch_size":64,"max_batch_tokens":16384,"min_pad_eff":0.8}' ;;
    G)   admin '{"quant":"fp8_fast","head_impl":"fast","mask_mode":"none","max_batch_size":64,"max_batch_tokens":16384,"min_pad_eff":0.8}' ;;
    G8)  admin '{"max_batch_tokens":8192}' ;;
    G4)  admin '{"max_batch_tokens":4096}' ;;
    G32) admin '{"max_batch_tokens":32768}' ;;
  esac
  run $c
done
echo SWEEP_DONE
