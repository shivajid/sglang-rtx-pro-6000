#!/bin/bash
# Cold-node JIT experiment: does a fresh pod (empty Triton cache) compile kernels on live traffic after
# warmup, and does PAD_MULTIPLE=16 stop it? Runs locally; uses the clef-server Deployment at 1 replica.
# For each arm: fresh TRITON_CACHE_DIR -> rollout -> count cache entries after warmup -> fixed traffic mix
# -> count again (new entries = runtime JIT compiles) -> logits dump for the exactness comparison.
set -u
cd "$(dirname "$0")/.."  # models/Clef: server/, bench/ and results/ paths below are relative to it
K() { timeout 180 kubectl "$@"; }
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
RUN=${RUN:-1}

pod() { K get pods -l app=clef-server --field-selector=status.phase=Running \
          --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1:].metadata.name}'; }
entries() { K exec "$1" -c server -- sh -c "find $2 -name '*.json' ! -name '__grp__*' ! -name '*.autotune.json' | wc -l"; }

client_job() {  # name, command; waits for completion
  K exec clef-bench-client -- bash -c "cd /work && (setsid nohup bash -c '$2; echo JOB_DONE' < /dev/null > /work/results/$1.log 2>&1 &)"
  for i in $(seq 1 120); do
    sleep 15
    K exec clef-bench-client -- grep -q JOB_DONE /work/results/$1.log 2>/dev/null && return 0
  done
  log "client job $1 TIMEOUT"
}

log "update code configmap"
K create configmap clef-server-code \
  --from-file=server/app.py --from-file=server/clef_engine.py --from-file=server/fast_patches.py \
  --from-file=server/fp8_linear.py --from-file=server/nvfp4_linear.py --dry-run=client -o yaml | K apply -f -
K cp bench/answers_dump.py clef-bench-client:/work/answers_dump.py

for PM in 1 16; do
  DIR=/cache/triton_cold_r${RUN}_pm$PM
  log "ARM PAD_MULTIPLE=$PM cache=$DIR"
  K scale deployment clef-server --replicas=1
  K set env deployment/clef-server QUANT=fp8_fast PAD_MULTIPLE=$PM TRITON_CACHE_DIR=$DIR
  sleep 20
  timeout 900 kubectl rollout status deployment/clef-server --timeout=900s
  sleep 10
  P=$(pod); log "pod $P"
  K logs "$P" -c server | grep -E "engine loaded|warmup done" | sed 's/^/  /'
  N0=$(entries "$P" $DIR); log "cache entries after warmup: $N0"
  T0=$(K exec "$P" -c server -- date -u '+%Y-%m-%d %H:%M:%S')
  client_job cold_pm$PM "python loadgen.py --url http://clef-server:8000 --workload workloads/mixed.jsonl --sweep 1,4,16,64 --per-worker 4 --min-num 24 --tag cold-pm$PM --out /work/results/cold.jsonl; python loadgen.py --url http://clef-server:8000 --workload workloads/short.jsonl --sweep 4,16,64 --per-worker 4 --min-num 24 --tag cold-pm$PM --out /work/results/cold.jsonl; python loadgen.py --url http://clef-server:8000 --workload workloads/medium.jsonl --sweep 2,8 --per-worker 4 --min-num 24 --tag cold-pm$PM --out /work/results/cold.jsonl"
  N1=$(entries "$P" $DIR); log "cache entries after traffic: $N1 (runtime compiles: $((N1 - N0)))"
  K exec "$P" -c server -- sh -c "find $DIR -name '*.json' ! -name '__grp__*' ! -name '*.autotune.json' -newermt '$T0' -printf '%TT %f\n' | sort | tail -40" | sed 's/^/  new: /'
  client_job dump_pm$PM "python answers_dump.py http://clef-server:8000 workloads/banking77.jsonl 150 /work/results/ans_b77_pm$PM.json; python answers_dump.py http://clef-server:8000 workloads/medium.jsonl 50 /work/results/ans_med_pm$PM.json"
done

log "exactness PAD_MULTIPLE=1 vs 16"
K exec clef-bench-client -- sh -c 'cd /work && python answers_dump.py --compare results/ans_b77_pm1.json results/ans_b77_pm16.json && python answers_dump.py --compare results/ans_med_pm1.json results/ans_med_pm16.json'
K exec clef-bench-client -- cat /work/results/cold.jsonl > results/cold_jit.jsonl
python3 bench/report_tables.py detail results/cold_jit.jsonl cold-pm1 cold-pm16
log "COLD_DONE"
