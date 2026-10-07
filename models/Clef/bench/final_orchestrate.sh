#!/bin/bash
# Final deployment + benchmark orchestration. Runs locally (detached); every kubectl call is bounded.
# Phases: deploy FP8 x2 -> bench -> scale to 1 -> bench -> NVFP4 x2 -> bench -> restore FP8 x2 -> fetch.
# Usage: bash final_orchestrate.sh [start_phase]
set -u
cd "$(dirname "$0")/.."  # models/Clef: server/, k8s/ and results/ paths below are relative to it
START=${1:-1}
RATES2=${RATES2:-5,9,12,14,17}
RATES1=${RATES1:-2.5,4.5,6,7,8.5}

K() { timeout 180 kubectl "$@"; }
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

client_launch() {  # name, command string (runs in /work, detached)
  log "client launch $1: $2"
  K exec clef-bench-client -- bash -c "cd /work && (setsid nohup bash -c '$2' < /dev/null > /work/results/$1.log 2>&1 &)"
}

client_wait() {  # name, timeout_s
  local s=0
  while [ $s -lt $2 ]; do
    if K exec clef-bench-client -- grep -q FINAL_DONE /work/results/$1.log 2>/dev/null; then
      log "client $1 done"; return 0
    fi
    sleep 30; s=$((s + 30))
  done
  log "client $1 TIMEOUT"; return 1
}

wait_ready() {  # expected replicas
  log "waiting for rollout ($1 replicas)"
  timeout 1500 kubectl rollout status deployment/clef-server --timeout=1500s
  local n=0
  for i in $(seq 1 90); do
    n=$(K get endpoints clef-server -o jsonpath='{.subsets[*].addresses[*].ip}' | wc -w)
    [ "$n" = "$1" ] && break
    sleep 5
  done
  log "ready endpoints: $n"
  K get pods -l app=clef-server -o wide
}

smoke() {  # one request through the Service from the client
  K exec clef-bench-client -- python -c "
import json, urllib.request, time
req = {'state': 'My card payment was declined twice at the store today.', 'questions': {'q': {'type': 'choice', 'instructions': 'Topic?', 'criteria': {'card': 'Card problems', 'transfer': 'Transfers', 'other': 'Anything else'}}}}
t = time.time()
r = urllib.request.urlopen(urllib.request.Request('http://clef-server:8000/v1/systemone', data=json.dumps(req).encode(), headers={'Content-Type': 'application/json'}), timeout=120)
print('smoke', r.status, round((time.time() - t) * 1000, 1), 'ms', r.read()[:300])
"
}

pod_logs() {  # tag
  for p in $(K get pods -l app=clef-server -o name); do
    K logs $p -c server 2>&1 | grep -E "engine loaded|warmup done|autotune|Uvicorn running|Error|error" | head -20 | sed "s|^|$1 $p: |"
  done
}

if [ $START -le 1 ]; then
  log "PHASE 1: deploy FP8 x2"
  K delete pod clef-dev-a clef-dev-b --grace-period=10 --wait=true
  K create configmap clef-server-code \
    --from-file=server/app.py --from-file=server/clef_engine.py --from-file=server/fast_patches.py \
    --from-file=server/fp8_linear.py --from-file=server/nvfp4_linear.py --dry-run=client -o yaml | K apply -f -
  K apply -f k8s/clef-server.yaml
  sleep 20
  wait_ready 2
  pod_logs fp8
  smoke
fi

if [ $START -le 2 ]; then
  log "PHASE 2: FP8 x2 bench"
  client_launch fp8_2rep "bash /work/final_bench.sh http://clef-server:8000 /work/results/final.jsonl fp8-2rep $RATES2"
  client_wait fp8_2rep 3600
fi

if [ $START -le 3 ]; then
  log "PHASE 3: FP8 x1 bench"
  K scale deployment clef-server --replicas=1
  sleep 40
  wait_ready 1
  client_launch fp8_1rep "bash /work/final_bench.sh http://clef-server:8000 /work/results/final.jsonl fp8-1rep $RATES1"
  client_wait fp8_1rep 4800
fi

if [ $START -le 4 ]; then
  log "PHASE 4: NVFP4 x2 bench"
  K set env deployment/clef-server QUANT=nvfp4_mlp
  K scale deployment clef-server --replicas=2
  sleep 30
  wait_ready 2
  pod_logs nvfp4
  smoke
  client_launch nvfp4_2rep "bash /work/nvfp4_bench.sh http://clef-server:8000 /work/results/final.jsonl nvfp4-2rep $RATES2"
  client_wait nvfp4_2rep 3600
fi

if [ $START -le 5 ]; then
  log "PHASE 5: restore FP8 x2"
  K set env deployment/clef-server QUANT=fp8_fast
  sleep 30
  wait_ready 2
  pod_logs restore
  smoke
fi

log "PHASE 6: fetch"
K exec clef-bench-client -- cat /work/results/final.jsonl > results/final.jsonl
for n in fp8_2rep fp8_1rep nvfp4_2rep; do
  K exec clef-bench-client -- cat /work/results/$n.log > results/final_$n.log 2>/dev/null
done
wc -l results/final.jsonl
log "ORCH_DONE"
