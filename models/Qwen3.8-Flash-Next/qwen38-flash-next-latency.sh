#!/bin/bash
# Qwen3.8-Flash-Next — LATENCY config: single TP8 instance + MTP speculative decoding.
# Best for single-stream / interactive / agentic coding (low TPOT, low TTFT).
# Validated: 1k/8k C=32 -> 1669 tok/s, TPOT 16.6ms, accept-len 3.01; GSM8K 0.982.
set -x
exec docker run --rm --name qwen38 --gpus all --ipc=host --network=host \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  lmsysorg/sglang:qwen38flashnext \
  python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3.8-Flash-Next \
    --tp 8 \
    --mem-fraction-static 0.85 \
    --chunked-prefill-size 8192 \
    --linear-attn-prefill-backend flashinfer \
    --linear-attn-decode-backend flashinfer \
    --linear-attn-verify-backend triton \
    --mamba-ssm-dtype bfloat16 \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --max-running-requests 96 \
    --reasoning-parser auto \
    --host 0.0.0.0 --port 30000
