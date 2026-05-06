#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-openai/whisper-large-v3}"
SERVED_NAME="${SERVED_NAME:-whisper-large-v3}"
HOST="${HOST:-host.docker.internal}"
PORT="${PORT:-8000}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
DATASET_PATH="${DATASET_PATH:-edinburghcstr/ami}"
HF_SUBSET="${HF_SUBSET:-ihm}"
HF_SPLIT="${HF_SPLIT:-test}"
RATES="${RATES:-8 16 32 64 128 256 inf}"
OUTDIR="${OUTDIR:-/results}"

mkdir -p "$OUTDIR"
cd "$OUTDIR"

echo "Benchmarking $MODEL at $HOST:$PORT"
echo "Rates: $RATES  ·  Prompts per rate: $NUM_PROMPTS"
echo "Results → $OUTDIR"

for RATE in $RATES; do
  echo ""
  echo "=== rate=$RATE ==="
  vllm bench serve \
    --backend openai-audio \
    --endpoint /v1/audio/transcriptions \
    --model "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --dataset-name hf \
    --dataset-path "$DATASET_PATH" \
    --hf-subset "$HF_SUBSET" \
    --hf-split "$HF_SPLIT" \
    --host "$HOST" --port "$PORT" \
    --num-prompts "$NUM_PROMPTS" \
    --request-rate "$RATE" \
    --save-result \
    --result-filename "whisper_rate_${RATE}.json" \
    --seed 42
done

echo ""
echo "=== summary ==="
python3 -c "
import json, glob
print(f'{\"rate\":>6}  {\"RTFx\":>8}  {\"req/s\":>8}  {\"p50_ms\":>10}  {\"p99_ms\":>10}')
for f in sorted(glob.glob('whisper_rate_*.json')):
    d = json.load(open(f))
    rate = f.replace('whisper_rate_','').replace('.json','')
    print(f'{rate:>6}  '
          f'{d.get(\"total_token_throughput\",0):>8.1f}  '
          f'{d.get(\"request_throughput\",0):>8.2f}  '
          f'{d.get(\"p50_e2el_ms\",0):>10.0f}  '
          f'{d.get(\"p99_e2el_ms\",0):>10.0f}')
"
