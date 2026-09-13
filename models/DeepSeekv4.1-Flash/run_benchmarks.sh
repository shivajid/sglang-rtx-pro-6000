#!/bin/bash
set -euo pipefail

# run_benchmarks.sh: Serving Benchmark Suite Driver for DeepSeek-V4.1-Flash on 8x RTX PRO 6000 (SM120)
# Supports direct execution on target VM or remote execution via SSH from local workstation.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${PROJECT_DIR}/results"
mkdir -p "${RESULTS_DIR}/8k_1k" "${RESULTS_DIR}/1k_8k"

VM_NAME="${VM_NAME:-shivaji-g4-384-spot}"
ZONE="${ZONE:-us-central1-b}"
PROJECT="${PROJECT:-northam-ce-mlai-tpu}"
CONTAINER_NAME="sglang-dsv41"
BASE_URL="${BASE_URL:-http://127.0.0.1:30000}"
MODEL="${MODEL:-/models/DeepSeek-V4.1-Flash}"

# Sweep concurrencies: 1 8 16 32 64 128
CONCURRENCIES=(1 8 16 32 64 128)

# Detect execution mode: local docker container or remote VM via SSH
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER_NAME}$"; then
  EXEC_MODE="local_docker"
  echo "Detected local container '${CONTAINER_NAME}'."
else
  EXEC_MODE="ssh"
  echo "Detected remote environment; using gcloud compute ssh to target ${VM_NAME}."
fi

run_bench() {
  local profile="$1"
  local input_len="$2"
  local output_len="$3"
  local c="$4"
  local num_prompts="$5"
  local out_file="${RESULTS_DIR}/${profile}/result_c${c}.json"
  local tmp_file="/tmp/result_${profile}_c${c}.json"

  echo "================================================================================"
  echo "Running benchmark: profile=${profile} concurrency=${c} prompts=${num_prompts}"
  echo "Input length: ${input_len} | Output length: ${output_len}"
  echo "================================================================================"

  if [[ "${EXEC_MODE}" == "local_docker" ]]; then
    docker exec "${CONTAINER_NAME}" rm -f "${tmp_file}"
    docker exec "${CONTAINER_NAME}" python3 -m sglang.bench_serving \
      --backend sglang \
      --base-url "${BASE_URL}" \
      --model "${MODEL}" \
      --dataset-name random \
      --random-input-len "${input_len}" \
      --random-output-len "${output_len}" \
      --random-range-ratio 0.0 \
      --num-prompts "${num_prompts}" \
      --max-concurrency "${c}" \
      --output-file "${tmp_file}" >&2
    docker exec "${CONTAINER_NAME}" cat "${tmp_file}" > "${out_file}"
  else
    # Execute on remote VM and extract the resulting JSON cleanly
    gcloud compute ssh "${VM_NAME}" --zone="${ZONE}" --project="${PROJECT}" --quiet \
      --command="docker exec ${CONTAINER_NAME} rm -f ${tmp_file} && docker exec ${CONTAINER_NAME} python3 -m sglang.bench_serving --backend sglang --base-url ${BASE_URL} --model ${MODEL} --dataset-name random --random-input-len ${input_len} --random-output-len ${output_len} --random-range-ratio 0.0 --num-prompts ${num_prompts} --max-concurrency ${c} --output-file ${tmp_file} >&2 && docker exec ${CONTAINER_NAME} cat ${tmp_file}" > "${out_file}"
  fi

  python3 -c "
import json
with open('${out_file}', 'r') as f:
    d = json.load(f)
d['max_concurrency'] = ${c}
if 'completed' not in d and 'completed_requests' in d:
    d['completed'] = d['completed_requests']
with open('${out_file}', 'w') as f:
    json.dump(d, f, indent=2)
"

  echo "Result written to: ${out_file} ($(wc -c < "${out_file}") bytes)"
}

# Profile 1: 8k input / 1k output (Prefill-Heavy Benchmark Sweep)
echo "=== Profile 8k_1k: 8192 input tokens / 1024 output tokens ==="
for c in "${CONCURRENCIES[@]}"; do
  NUM_PROMPTS=$(( c < 16 ? 16 : c * 2 ))
  run_bench "8k_1k" 8192 1024 "${c}" "${NUM_PROMPTS}"
done

# Profile 2: 1k input / 8k output (Reasoning & Generation Heavy Benchmark Sweep)
echo "=== Profile 1k_8k: 1024 input tokens / 8192 output tokens ==="
for c in "${CONCURRENCIES[@]}"; do
  NUM_PROMPTS=$(( c < 16 ? 16 : c * 2 ))
  run_bench "1k_8k" 1024 8192 "${c}" "${NUM_PROMPTS}"
done

echo "All 12 benchmark sweeps completed successfully."
