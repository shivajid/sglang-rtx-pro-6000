#!/usr/bin/env bash
set -euo pipefail
# Reproduce the selected SIMULATED benchmark recipe inside the pinned image.
# Exactly four reserved GPUs must be visible; mount the model directory at /model.
# Fixed tokenID100 is synthetic output, not normal model answers.
# Model: XiaomiMiMo/MiMo-V2.6-Flash-RL @ 5711b268169967567844e1e560e8a3966da959b1
# Image: lmsysorg/sglang@sha256:750718d6416b730c6ebbdd2d2f86c78d31878e4bd5cdddcd64dd1ab9f7d5537d
# Representative tested placement: n2b; HTTP port preserved from that test.
# No additional MoE input quantization requested; numerical equivalence is not established.

export SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH=1
export SGLANG_FP8_IGNORED_LAYERS=model.mtp_block.self_attn.o_proj
export SGLANG_SIMULATE_ACC_LEN=3.4
export SGLANG_SIMULATE_ACC_METHOD=match-expected
export SGLANG_SIMULATE_ACC_TOKEN_MODE=fixed

exec sglang serve \
  --model-path /model/MiMo-V2.6-Flash \
  --served-model-name mimo-v2.6-flash \
  --tp 4 \
  --ep 1 \
  --dp 1 \
  --pp-size 1 \
  --moe-dense-tp-size 1 \
  --enable-dp-lm-head \
  --log-level-http warning \
  --enable-cache-report \
  --page-size 1 \
  --cuda-graph-max-bs-decode 64 \
  --max-running-requests 128 \
  --trust-remote-code \
  --mem-fraction-static 0.8 \
  --swa-full-tokens-ratio 0.24 \
  --chunked-prefill-size 16384 \
  --max-prefill-tokens 32768 \
  --reasoning-parser mimo \
  --tool-call-parser mimo \
  --context-length 1048576 \
  --host 0.0.0.0 \
  --port 32000 \
  --moe-runner-backend humming \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path /model/MiMo-V2.6-Flash \
  --speculative-num-draft-tokens 4 \
  --skip-server-warmup \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --enable-multi-layer-eagle \
  --kv-cache-dtype fp8_e4m3 \
  --random-seed 42
