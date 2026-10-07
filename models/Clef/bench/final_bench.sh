#!/bin/bash
# Final-deployment benchmark through the Kubernetes Service (run inside clef-bench-client).
# Usage: bash final_bench.sh <url> <outfile> <tag> [open-loop rates for mixed.jsonl]
URL=${1:-http://clef-server:8000}
OUT=${2:-/work/results/final.jsonl}
TAG=${3:-final}
RATES=${4:-}
cd /work
python loadgen.py --url $URL --workload workloads/short.jsonl     --sweep 1,16,64          --per-worker 4 --min-num 32 --tag $TAG --out $OUT
python loadgen.py --url $URL --workload workloads/medium.jsonl    --sweep 1,4,16,32,64,128 --per-worker 4 --min-num 32 --tag $TAG --out $OUT
python loadgen.py --url $URL --workload workloads/mixed.jsonl     --sweep 1,16,64,128      --per-worker 4 --min-num 32 --tag $TAG --out $OUT
python loadgen.py --url $URL --workload workloads/banking77.jsonl --sweep 1,16,64          --per-worker 4 --min-num 32 --tag $TAG --out $OUT
python loadgen.py --url $URL --workload workloads/long.jsonl      --sweep 1,16,32          --per-worker 4 --min-num 24 --tag $TAG --out $OUT
python loadgen.py --url $URL --workload workloads/xlong.jsonl     --sweep 1,4              --per-worker 2 --min-num 8  --tag $TAG --out $OUT
if [ -n "$RATES" ]; then
  python loadgen.py --url $URL --workload workloads/mixed.jsonl --rates $RATES --duration 60 --tag $TAG-open --out $OUT
fi
echo FINAL_DONE
