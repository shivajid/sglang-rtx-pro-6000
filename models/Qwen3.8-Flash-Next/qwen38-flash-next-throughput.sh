#!/bin/bash
# Qwen3.8-Flash-Next — THROUGHPUT config: 2 independent DP replicas (TP4 each) + EP4 + MTP.
# Best for max aggregate tokens/sec at high concurrency.
# Validated (MTP off): 1k/8k C=96 -> 2825 tok/s vs TP8 2526 tok/s (+12%). MTP on here.
set -x
exec docker run --rm --name qwen38 --gpus all --ipc=host --network=host \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  lmsysorg/sglang:qwen38flashnext \
  python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3.8-Flash-Next \
    --dp-size 2 \
    --tp 4 \
    --ep 4 \
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
