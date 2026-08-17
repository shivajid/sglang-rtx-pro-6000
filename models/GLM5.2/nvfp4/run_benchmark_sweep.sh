#!/bin/bash
# GLM-5.2 NVFP4 1K/8K Benchmark Sweep Script
# Concurrencies: 128, 256, 512 | Prompts: 3x Concurrency (384, 768, 1536)
set -euo pipefail

HOST="${1:-localhost}"
PORT="${2:-8000}"
MODEL="${3:-nvidia/GLM-5.2-NVFP4}"
OUTPUT_DIR="${4:-/tmp/bench_results_glm52}"

echo "=========================================================================="
echo "🚀 Running GLM-5.2-NVFP4 1K/8K SGLang Benchmark Sweep"
echo "Host: ${HOST}:${PORT} | Model: ${MODEL}"
echo "Concurrencies: 128 (384 prompts), 256 (768 prompts), 512 (1536 prompts)"
echo "=========================================================================="

mkdir -p "$OUTPUT_DIR"

python3 /script/benchmark_sweep_1k8k.py \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL" \
  --tokenizer "$MODEL" \
  --concurrencies 128 256 512 \
  --prompts-multiplier 3 \
  --input-len 1024 \
  --output-len 8192 \
  --random-range-ratio 1.0 \
  --output-dir "$OUTPUT_DIR"
