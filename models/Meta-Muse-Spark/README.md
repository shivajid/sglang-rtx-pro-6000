# Meta Muse-Spark-Open-816B-A42B (NVFP4) on GCP G4 (RTX PRO 6000)

Serve [`meta-models/Muse-Spark-Open-816B-A42B-NVFP4`](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B-NVFP4) with SGLang on a single GKE `g4-standard-384` node (8× NVIDIA RTX PRO 6000 Blackwell Server Edition, SM120, 96 GB VRAM/GPU). This recipe uses **DFlash speculative decoding** (`RadixArk/Muse-Spark-Open-DFlash2`, 8 draft tokens with block verification), **FlashInfer CUTLASS NVFP4 MoE**, and **HiCache** (pinned host-memory KV cache offload for 256k-context agentic workloads).

**Highlights** (single `g4-standard-384` node, 8× RTX PRO 6000 SM120, benchmarked October 19, 2026):
- **1,987 output tok/s** (3,881 tok/s peak) at 1k in / 8k out with 64 concurrent requests (**7.11** avg speculative accept length).
- **3,852 total tok/s** (1,926 output tok/s, 3,920 tok/s peak) at 1k in / 1k out with 64 concurrent requests (**6.97** avg accept length).
- **14.1 ms median ITL** / **16.1 ms median TPOT** at 1k in / 8k out (`C=64`).
- **153–188 tok/s single-stream decode** with DFlash (up from 37.6 tok/s with decode CUDA graphs alone and 15 tok/s eager).

| File | Description |
| :--- | :--- |
| [`nvfp4-g4-gke-dflash-muse-spark-agentx-hicache.yaml`](./nvfp4-g4-gke-dflash-muse-spark-agentx-hicache.yaml) | Kubernetes `ConfigMap` (SM120 CUTLASS MoE + Granian keep-alive patch), `Deployment`, and `Service` |

---

## Config Details

### Image and Checkpoints

| Component | Value |
| :--- | :--- |
| **Container Image** | [`aurius/sglang-muse-spark-rtxpro6000:10182026`](https://hub.docker.com/r/aurius/sglang-muse-spark-rtxpro6000) (slimmed, single-layer 27.7 GB build of `lmsysorg/sglang:dev-muse-spark-open`, CUDA 13.0, FlashInfer 0.7.0.post1) |
| **Target Model** | [`meta-models/Muse-Spark-Open-816B-A42B-NVFP4`](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B-NVFP4) @ `4239e4db19376904f2e82c167e78f70abb5455fe` (~558 GB on disk, 69.75 GB/GPU at TP8/EP8) |
| **Draft Model (DFlash)** | [`RadixArk/Muse-Spark-Open-DFlash2`](https://huggingface.co/RadixArk/Muse-Spark-Open-DFlash2) @ `46c90f412549c472941a68382d91a99e5e6372a0` (2.64 GB/GPU) |
| **Served Model Name** | `meta-models/Muse-Spark-Open-816B-A42B-NVFP4` |

- **Architecture**: 816B total parameters, 42B active parameters per token, 62 layers (layers 0 and 61 use BF16 Triton MoE; layers 1–60 use 256-expert top-8 NVFP4 MoE with per-expert post-RMSNorm), hybrid sliding-window + full attention (`--swa-full-tokens-ratio 0.4`), 202,048 vocab size.
- **Why a slimmed image**: Upstream `lmsysorg/sglang:dev-muse-spark-open` is 56.4 GB unpacked across 10 layers and overflows a standard 100 GB GKE node boot disk. Removing the duplicate `/sgl-workspace/sglang` checkout, pip caches, non-SM120 prebuilt FlashInfer JIT caches, and SM100/103/107 `trtllm-gen` cubins flattens the image to **27.7 GB** (`aurius/sglang-muse-spark-rtxpro6000:10182026`).

---


### Server Flags

| Category | Flags |
| :--- | :--- |
| **Model & Quantization** | `--model-path /data/model/hub/models--meta-models--Muse-Spark-Open-816B-A42B-NVFP4/snapshots/4239e4db19376904f2e82c167e78f70abb5455fe --served-model-name meta-models/Muse-Spark-Open-816B-A42B-NVFP4 --quantization modelopt_fp4 --trust-remote-code` |
| **Parallelism** | `--tp-size 8 --ep-size 8` |
| **MoE & Attention** | `--moe-runner-backend flashinfer_cutlass --attention-backend trtllm_mha --kv-cache-dtype bfloat16` |
| **Speculative Decoding** | `--speculative-algorithm DFLASH --speculative-draft-model-path /data/model/hub/models--RadixArk--Muse-Spark-Open-DFlash2/snapshots/46c90f412549c472941a68382d91a99e5e6372a0 --speculative-draft-attention-backend flashinfer --speculative-num-draft-tokens 8 --speculative-use-block-verification` |
| **Context, KV & Batching** | `--context-length 262144 --page-size 128 --swa-full-tokens-ratio 0.4 --mem-fraction-static 0.85 --chunked-prefill-size 4096 --max-prefill-tokens 4096 --max-running-requests 128 --cuda-graph-max-bs-decode 128 --disable-prefill-cuda-graph --prefill-decode-interval 1` |
| **HiCache (Host KV)** | `--enable-hierarchical-cache --hicache-size 2.0 --enable-cache-report` *(use `--hicache-size 100` for 100 GB/rank pinned host cache on 1.4 TiB nodes)* |
| **Parsers & Server** | `--reasoning-parser muse --tool-call-parser muse --tokenizer-worker-num 4 --enable-http2 --watchdog-timeout 3600 --host 0.0.0.0 --enable-metrics --port 30080` |

### Environment Variables

| Variable | Value | Purpose |
| :--- | :--- | :--- |
| `HF_HUB_CACHE` / `HF_HUB_OFFLINE` | `/data/model/hub` / `1` | Load pre-staged weights directly from the mounted persistent volume |
| `SGLANG_SET_CPU_AFFINITY` | `1` | Pin each TP rank to its 24-core NUMA slice (also keeps pinned HiCache memory local to each socket) |
| `OMP_NUM_THREADS` | `24` | CPU threads per TP rank |
| `NCCL_P2P_LEVEL` | `SYS` | Enable direct PCIe P2P across dual CPU sockets (GPUs 0–3 ↔ 4–7) |
| `NCCL_MIN_NCHANNELS` | `8` | Minimum NCCL channels for TP8 all-reduces |
| `NCCL_CUMEM_ENABLE` | `1` | Enable CUDA VMM allocations for NCCL |
| `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN` | `1` | Allow overriding the 1M checkpoint context length to 262,144 |
| `SGLANG_TIMEOUT_KEEP_ALIVE` | `1800` | 30-minute HTTP keep-alive for long multi-turn agent tool execution |
| `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND` | `python` | Required radix-tree backend for Muse SWA + HiCache |
| `SGLANG_FLASHINFER_AUTOTUNE_EXTEND` | `0` | Skip prefill-sized FlashInfer extend autotune pass (decode/verify shapes are autotuned and cached) |

### 96 GB Per-GPU VRAM Breakdown (`TP=8, EP=8`)

| Allocation Stage | Memory per GPU (GB) | Remaining Free (GB) |
| :--- | :---: | :---: |
| Target weights (`Muse-Spark-Open-816B-A42B-NVFP4`) | 69.75 | 24.2 |
| DFlash2 draft model weights + draft KV | 2.64 | 21.5 |
| Device KV cache (`389,632` full tokens + `155,776` SWA tokens, BF16) | 6.39 | 14.0 |
| Target verify CUDA graphs (`bs ≤ 128 × 8` tokens) | 2.57 | 11.4 |
| Draft decode CUDA graphs (`bs ≤ 128`) | 3.19 | 8.2 |
| DFlash sampling buffers (`128 × 8 × 202,048` vocab) + runtime headroom | ~6.0 | **2.21** |

---

## Deploy & Test on GKE

### 1. Deploy

Ensure the model weights PVC (`muse-spark-weights-pvc`) is populated in the cluster and update the `nodeAffinity` hostname in [`nvfp4-g4-gke-dflash-muse-spark-agentx-hicache.yaml`](./nvfp4-g4-gke-dflash-muse-spark-agentx-hicache.yaml) to match your target G4 node:

```bash
kubectl apply -f nvfp4-g4-gke-dflash-muse-spark-agentx-hicache.yaml
kubectl logs -f -l app=muse-spark-nvfp4-g4-dflash-agentx-hicache
```

Startup takes ~9 minutes once the image and JIT caches (`/var/lib/muse-spark/sglang-cache` and `/var/lib/muse-spark/flashinfer-cache`) are warm (~447s serial weight load + ~13s CUDA graph capture). Cold start with an empty FlashInfer autotune cache takes ~14 minutes.

### 2. Port-Forward

```bash
kubectl port-forward deploy/muse-spark-nvfp4-g4-dflash-agentx-hicache 30084:30080
```

### 3. Sample `curl` Commands

#### Readiness & Models Check
```bash
curl -i http://127.0.0.1:30084/ready
curl -s http://127.0.0.1:30084/v1/models | jq .
```

#### Chat Completion (Non-Streaming)
> **Note:** Because `--reasoning-parser muse` is enabled, the model emits chain-of-thought reasoning in `reasoning_content` before final `content`. Use `max_tokens >= 512` so both reasoning and final answer complete.

```bash
curl -s http://127.0.0.1:30084/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-models/Muse-Spark-Open-816B-A42B-NVFP4",
    "messages": [
      {"role": "user", "content": "Explain speculative decoding in 3 bullet points."}
    ],
    "max_tokens": 1024,
    "temperature": 0.6
  }' | jq .
```

#### Chat Completion (Streaming)
```bash
curl -N http://127.0.0.1:30084/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-models/Muse-Spark-Open-816B-A42B-NVFP4",
    "messages": [
      {"role": "user", "content": "Write a Python script to check GPU compute capability with PyTorch."}
    ],
    "max_tokens": 1024,
    "temperature": 0.6,
    "stream": true
  }'
```

---

## Benchmark Results

Measured with `sglang.bench_serving` (`--dataset-name random --random-range-ratio 1.0 --max-concurrency 64 --num-prompts 192 --extra-request-body '{"ignore_eos":true,"temperature":0}'`) on a single GKE `g4-standard-384` node (`8× RTX PRO 6000 Blackwell SM120`):

### Throughput & Speculative Decoding (`C=64`, 192 requests)

| Scenario (ISL / OSL) | Completed | Duration (s) | Req/s | Input tok/s | Output tok/s | Total tok/s | Peak Output tok/s | Avg Accept Len |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1k_1k`** (1024 / 1024) | 192 / 192 | 102.09 | 1.88 | 1,925.8 | **1,925.8** | **3,851.6** | 3,920.0 | **6.97** |
| **`1k_8k`** (1024 / 8192) | 192 / 192 | 791.45 | 0.24 | 248.4 | **1,987.3** | **2,235.7** | 3,881.0 | **7.11** |

### Latency Percentiles (`C=64`, in ms)

| Scenario | Metric | Mean | Median (P50) | P90 | P95 | P99 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`1k_1k`** | **TTFT (ms)** | 4,533.3 | 1,421.2 | 13,745.2 | 15,803.3 | 17,219.9 |
| **`1k_1k`** | **TPOT (ms)** | 25.20 | 23.58 | 33.84 | 42.24 | 58.25 |
| **`1k_1k`** | **ITL (ms)** | 25.20 | 15.92 | 43.78 | 86.78 | 126.46 |
| **`1k_1k`** | **E2E (ms)** | 30,307.9 | 26,115.8 | 41,029.5 | 51,690.1 | 61,922.9 |
| **`1k_8k`** | **TTFT (ms)** | 9,750.8 | 6,168.2 | 26,426.2 | 32,987.7 | 53,042.1 |
| **`1k_8k`** | **TPOT (ms)** | 20.51 | 16.09 | 29.81 | 40.17 | 52.87 |
| **`1k_8k`** | **ITL (ms)** | 20.51 | 14.07 | 16.16 | 32.06 | 86.08 |
| **`1k_8k`** | **E2E (ms)** | 177,763.4 | 146,376.8 | 246,631.7 | 331,673.7 | 445,032.7 |
