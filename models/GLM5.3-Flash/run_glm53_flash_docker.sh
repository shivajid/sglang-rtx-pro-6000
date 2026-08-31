#!/usr/bin/env bash
# Run GLM-5.3-Flash (SGLang, TP8) on a single 8x RTX PRO 6000 Blackwell (SM120)
# host with plain Docker — no GKE/Kubernetes required.
#
# Docker translation of glm53-flash-g4-tuned.yaml:
#   hostIPC: true            -> --ipc=host
#   64Gi emptyDir /dev/shm   -> --shm-size=64g
#   IPC_LOCK capability      -> --cap-add=IPC_LOCK
#   nvidia.com/gpu: 8        -> --gpus all
# No NCCL_* env vars are needed for single-node TP8.
#
# The image bakes in the SM120 kernel patches (tilelang DSA num_stages=1,
# DSA backend -> tilelang) and the tuned E=289 fused-MoE Triton config —
# stock SGLang will NOT serve this model on SM120.
#
# Usage:
#   HF_TOKEN=<your_hf_token> ./run_glm53_flash_docker.sh
#
# Optional overrides (defaults shown):
#   MODEL_CACHE_DIR=/mnt/models   host dir for the HF weight cache
#   PORT=30000                    host port to expose

set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN, e.g. HF_TOKEN=hf_... $0}"

MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/mnt/models}"
PORT="${PORT:-30000}"
IMAGE="aurius/sglang-glm53-flash-rtxpr6000:sm120-moe-tuned-v1"

mkdir -p "${MODEL_CACHE_DIR}"

docker run -d --name glm53-flash \
  --gpus all \
  --ipc=host \
  --shm-size=64g \
  --cap-add=IPC_LOCK \
  -p "${PORT}:30000" \
  -v "${MODEL_CACHE_DIR}:/models" \
  -e MODEL_PATH=zai-org/GLM-5.3-Flash \
  -e TENSOR_PARALLEL_SIZE=8 \
  -e KV_CACHE_DTYPE=bfloat16 \
  -e MOE_RUNNER_BACKEND=triton \
  -e GRAMMAR_BACKEND=xgrammar \
  -e DISABLE_CUDA_GRAPH=0 \
  -e MEM_FRACTION_STATIC=0.85 \
  -e HOST=0.0.0.0 \
  -e PORT=30000 \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" \
  -e HF_HOME=/models/hf \
  "${IMAGE}"

echo "Container 'glm53-flash' started on port ${PORT}."
echo "Model load takes ~10 min; watch progress with:  docker logs -f glm53-flash"
echo "Ready when this returns 200:                    curl http://localhost:${PORT}/health"
