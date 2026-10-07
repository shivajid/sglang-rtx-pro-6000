#!/bin/bash
# Baseline A sweep: SGLang backbone-only prefill ceiling (max_new_tokens=1, matched token lengths).
# Usage: bash sweep_sglang.sh <url> <tag>
URL=${1:-http://10.10.0.72:30000}
TAG=${2:-sglang-bf16}
OUT=/work/results/${TAG}.jsonl
cd /work
python loadgen.py --url $URL --target sglang --workload workloads/short_fixed.jsonl  --sweep 1,2,4,8,16,32,64 --per-worker 4 --min-num 24 --tag $TAG --out $OUT
python loadgen.py --url $URL --target sglang --workload workloads/medium_fixed.jsonl --sweep 1,2,4,8,16,32,64 --per-worker 4 --min-num 24 --tag $TAG --out $OUT
python loadgen.py --url $URL --target sglang --workload workloads/long_fixed.jsonl   --sweep 1,2,4,8,16,32 --per-worker 3 --min-num 16 --tag $TAG --out $OUT
python loadgen.py --url $URL --target sglang --workload workloads/xlong.jsonl        --sweep 1,2,4,8 --per-worker 3 --min-num 8 --tag $TAG --out $OUT
python loadgen.py --url $URL --target sglang --workload workloads/mixed.jsonl        --sweep 1,4,16,64 --per-worker 4 --min-num 24 --tag $TAG --out $OUT
python loadgen.py --url $URL --target sglang --workload workloads/banking77.jsonl    --sweep 1,8,32 --per-worker 4 --min-num 24 --tag $TAG --out $OUT
echo SWEEP_DONE
