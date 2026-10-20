# DeepSeek-V4.1-Flash-1017 on GCP G4 (RTX PRO 6000) — DSPARK + HiCache

Serve [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) with SGLang on a single GKE `g4-standard-384` node (8× NVIDIA RTX PRO 6000 Blackwell Server Edition, SM120, 96 GB VRAM/GPU). This recipe uses **TP8 / EP8**, **FP4 routed experts** (FlashInfer SM120 CUTLASS W4A8 MoE) with **FP8 dense GEMMs and FP8 KV cache**, **DSPARK speculative decoding** (verify window 5), **HiCache** (pinned host-memory KV tier, ratio 2.0) and **Engram host tables**, sized for **262k-token agentic contexts** (AgentX).

**Highlights** (single `g4-standard-384` node, 8× RTX PRO 6000 SM120, benchmarked October 20, 2026):
- **1,817 output tok/s** (2,345 tok/s peak) at 1k in / 8k out with 64–128 concurrent requests (**5.2–5.3** avg DSPARK accept length out of 5).
- **2,661 total tok/s** (1,331 output tok/s, 2,355 tok/s peak) at 1k in / 1k out with 128 concurrent requests.
- **4,816 total tok/s** (4,281 input tok/s) at 8k in / 1k out with 128 concurrent requests (prefill-bound).
- **15.9 ms median ITL / 17.1 ms median TPOT** at 1k in / 8k out (`C=32`); 26.6 ms / 28.6 ms at `C=64`.
- **AgentX** (256k agentic traces, `C=16`, 30 min): 295 requests, 0 errors, **742 ms median TTFT**, **26.8 ms median ITL**, ~96k avg input tokens per request with a **94.7 % prompt-cache hit rate**.

| File | Description |
| :--- | :--- |
| [`nvfp4-g4-gke-dspark-dsv41-agentx-hicache.yaml`](./nvfp4-g4-gke-dspark-dsv41-agentx-hicache.yaml) | Kubernetes `PersistentVolumeClaim` (Hugging Face weight cache), `Deployment`, and `Service` |
| [`dsv41-flash-bench-sweep-job.yaml`](./dsv41-flash-bench-sweep-job.yaml) | `sglang.bench_serving` ISL/OSL sweep `Job` (1k/1k, 8k/1k, 1k/8k at `C=32/64/128`) |
| [`dsv41-flash-agentx-bench-job.yaml`](./dsv41-flash-agentx-bench-job.yaml) | AgentX (`agentx-harness` / AIPerf agentic replay, 256k traces) `Job` |

---

## Config Details

### Image and Checkpoint

| Component | Value |
| :--- | :--- |
| **Container Image** | `lmsysorg/sglang:dev-dsv41` (stock image, no patches or ConfigMaps) |
| **Model** | [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), downloaded from Hugging Face on first start (~476 GB) into the `dsv41-flash-hf-cache` PVC (600 Gi `dynamic-rwo`, `HF_HOME=/ssd`) |
| **Served Model Name** | `deepseek-ai/DeepSeek-V4.1-Flash-1017` |
| **Target Node / Pool** | `gke-shd-gem-cluster-shd-gem-g4-gpu-po-fce8ea74-n0k0` / `shd-gem-g4-gpu-pool-2` (pinned with `nodeAffinity` + `nodeSelector`) |

- **Hardware**: 8× RTX PRO 6000 Blackwell SE (SM120), 96 GB each, PCIe Gen5 fabric (no NVLink). `NCCL_P2P_LEVEL=SYS` enables direct cross-socket P2P, `NCCL_MIN_NCHANNELS=16` for the TP8 all-reduces, `SGLANG_SET_CPU_AFFINITY=1` pins each TP rank to its 24-core NUMA slice.
- **Quantization**: FP4 routed experts (FlashInfer SM120 CUTLASS W4A8 MoE), FP8 dense weights (`--fp8-gemm-backend flashinfer_cutlass`), FP8 KV cache (`fp8_e4m3`), FP4 sparse-attention indexer (`--enable-deepseek-v4-fp4-indexer`). DeepGEMM is disabled (no SM120 support).
- **Parallelism**: TP8 + EP8 — the FP4 expert intermediate dims need a multiple of 128 per rank.
- **Speculative decoding**: DSPARK with a verify window of 5 (`--speculative-dspark-block-size 5`). CUDA graphs are captured for decode/verify up to batch size 64.
- **HiCache**: pinned host-memory KV tier behind the GPU radix cache, ratio 2.0, `write_back`, `page_first` layout, `kernel` IO backend, `--enable-cache-report` so clients (and AgentX) see `cached_tokens`.
- **Engram associative memory**: pinned host tables (`SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`), layout `private` (one shard per rank + all-reduce; the only layout besides `shared` accepted by this image, and it got 100 % huge pages on COS). ~387 GB of pinned host RAM across the 8 ranks.
- For the model architecture itself (CED decoder, sparse attention, Engram) see the earlier [DeepSeek-V4.1-Flash recipe](../DeepSeekv4.1-Flash/README.md); this directory tracks the `1017` checkpoint with the DSPARK + HiCache agent configuration.

### Server Flags

| Category | Flags |
| :--- | :--- |
| **Model & Quantization** | `--model deepseek-ai/DeepSeek-V4.1-Flash-1017 --trust-remote-code --kv-cache-dtype fp8_e4m3 --fp8-gemm-backend flashinfer_cutlass --enable-deepseek-v4-fp4-indexer` |
| **Parallelism** | `--tp 8 --ep-size 8` |
| **Speculative Decoding** | `--speculative-algorithm DSPARK --speculative-dspark-block-size 5` |
| **Context, KV & Batching** | `--context-length 262144 --mem-fraction-static 0.70 --chunked-prefill-size 8192 --max-prefill-tokens 8192 --max-running-requests 128 --cuda-graph-max-bs-decode 64 --enable-decoder-swa-bounded-replay` |
| **HiCache (Host KV)** | `--enable-hierarchical-cache --hicache-ratio 2.0 --hicache-write-policy write_back --hicache-mem-layout page_first --hicache-io-backend kernel --enable-cache-report` |
| **Parsers & Server** | `--reasoning-parser auto --tool-call-parser auto --watchdog-timeout 3600 --enable-metrics --host 0.0.0.0 --port 30080` |

### Environment Variables

| Variable | Value | Purpose |
| :--- | :--- | :--- |
| `HF_HOME` / `HF_TOKEN` | `/ssd` / secret `hf-token-dsv41` | Weights are fetched from Hugging Face once and cached on the PVC |
| `SGLANG_DEFAULT_THINKING` | `1` | Thinking mode on by default |
| `SGLANG_DSV41_REASONING_EFFORT` | `high` | Reasoning effort for DeepSeek-V4.1 |
| `SGLANG_SM120_FLASHMLA_BACKEND` | `triton` | Triton FlashMLA path on SM120 (no FlashMLA cubins for this arch) |
| `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH` | `0` | Use the fused kernel, not the PyTorch fallback, for FP8 paged-MQA indexer logits |
| `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE` / `SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT` | `1` / `private` | Engram tables in pinned host RAM, one shard per rank |
| `SGLANG_TIMEOUT_KEEP_ALIVE` | `1800` | 30-minute HTTP keep-alive for long multi-turn agent tool execution |
| `NCCL_P2P_LEVEL` / `NCCL_P2P_DISABLE` / `NCCL_IB_DISABLE` | `SYS` / `0` / `1` | Direct PCIe P2P across both CPU sockets, no InfiniBand |
| `NCCL_MIN_NCHANNELS` | `16` | Minimum NCCL channels for TP8 all-reduces over PCIe |
| `CUDA_DEVICE_ORDER` / `CUDA_VISIBLE_DEVICES` | `PCI_BUS_ID` / `0,1,2,3,4,5,6,7` | Stable GPU ↔ NUMA mapping |
| `SGLANG_ENABLE_DEEP_GEMM` / `SGLANG_ENABLE_JIT_DEEPGEMM` / `SGLANG_DISABLE_DEEP_GEMM` | `0` / `0` / `1` | DeepGEMM has no SM120 kernels; force the FlashInfer CUTLASS FP8 path |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Reduce allocator fragmentation for 256k-context buffers |
| `SAFETENSORS_FAST_GPU` / `FLASHINFER_DISABLE_VERSION_CHECK` | `1` / `1` | Faster weight load; skip FlashInfer version gate |
| `OMP_NUM_THREADS` / `SGLANG_SET_CPU_AFFINITY` | `24` / `1` | 24 CPU threads per TP rank, pinned to its NUMA slice |

### Memory Sizing for 256k Agent Contexts

The first deployment (`--mem-fraction-static 0.75 --chunked-prefill-size 16384 --max-prefill-tokens 32768`, checkpoint-default 1M context) left only ~13.5 GiB of dynamic GPU memory after CUDA-graph capture and **OOM'd in the DSv4.1 sparse-indexer top-k** (`torch.cat(masks)`, 1.6 GiB) as soon as a 16k-token chunk of a ~256k-token AgentX context was prefilled. The current settings fix that:

- `--context-length 262144` — AgentX max; the checkpoint default is 1M and sizes every context-length buffer 4× larger.
- `--mem-fraction-static 0.70`, `--chunked-prefill-size 8192`, `--max-prefill-tokens 8192`, `--max-running-requests 128`.
- Validated on `shivaji-g4-384-spot` with 1×250k, 4×128k and 8×64k + 1×250k concurrent prompts before redeploying.

### Pod Resources, Probes & Storage

- **Resources**: `cpu: 96`, `memory: 1Ti` request (8 × ~64 GB HiCache host pools + 387 GB Engram + RSS) with **no memory limit** (pinned pages are charged to the pod; a limit OOM-kills at start-up), `nvidia.com/gpu: 8`, 32 Gi `/dev/shm`.
- **Probes**: stock SGLang exposes `/health` (503 while starting, 200 once ready) and **no `/ready`** — a `/ready` startup probe 404s forever and restarts the container every 90 min. `startupProbe` allows 60 min (first-run weight download + load + JIT + CUDA graphs); no `livenessProbe` on purpose (`--watchdog-timeout 3600` is SGLang's own hung-scheduler killer).
- **Storage**: weights on the `dsv41-flash-hf-cache` PVC, reused across pod re-creations (the earlier pod-scoped ephemeral volume re-downloaded 476 GB on every redeploy). Triton/FlashInfer JIT caches live on `hostPath` (`/var/lib/sglang-cache/dsv41-flash`, `/var/lib/flashinfer-cache/dsv41-flash`) so restarts skip recompilation.
- `strategy: Recreate` — a surge pod can never fit next to the running one (8 GPUs, RWO volume).

---

## Deploy & Test on GKE

### 1. Deploy

Create the Hugging Face token secret once, update the `nodeAffinity` hostname / `nodeSelector` pool in [`nvfp4-g4-gke-dspark-dsv41-agentx-hicache.yaml`](./nvfp4-g4-gke-dspark-dsv41-agentx-hicache.yaml) to your G4 node, then apply:

```bash
kubectl create secret generic hf-token-dsv41 --from-literal=HF_TOKEN=<token>   # once
kubectl apply -f nvfp4-g4-gke-dspark-dsv41-agentx-hicache.yaml
kubectl logs -f -l app=deepseek-v41-flash-nvfp4-g4-dspark-agentx-hicache
```

The first start downloads ~476 GB of weights into the PVC and JIT-compiles the SM120 kernels; later restarts reuse the PVC and the `hostPath` JIT caches.

### 2. Port-Forward

```bash
kubectl port-forward svc/deepseek-v41-flash-nvfp4-dspark-agentx-hicache 30080:30080
```

### 3. Sample `curl` Commands

#### Health & Models Check
```bash
curl -i http://127.0.0.1:30080/health          # 200 once the engine is up (there is no /ready)
curl -s http://127.0.0.1:30080/v1/models | jq .
```

#### Chat Completion (Non-Streaming)
> **Note:** Thinking is on by default (`SGLANG_DEFAULT_THINKING=1`, `--reasoning-parser auto`), so the model emits chain-of-thought in `reasoning_content` before the final `content`. Use `max_tokens >= 1024` so both complete.

```bash
curl -s http://127.0.0.1:30080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
    "messages": [
      {"role": "user", "content": "Explain speculative decoding in 3 bullet points."}
    ],
    "max_tokens": 1024,
    "temperature": 0.6
  }' | jq .
```

#### Chat Completion (Streaming)
```bash
curl -N http://127.0.0.1:30080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
    "messages": [
      {"role": "user", "content": "Write a Python script to check GPU compute capability with PyTorch."}
    ],
    "max_tokens": 1024,
    "temperature": 0.6,
    "stream": true
  }'
```

### 4. Run the Benchmarks

Both benchmark Jobs run on the serving node (no GPU requested), talk to the in-cluster Service, and keep their pod alive for 2 h after finishing so results can be copied out.

```bash
# ISL/OSL sweep (sglang.bench_serving, ~70 min)
kubectl apply -f dsv41-flash-bench-sweep-job.yaml
kubectl logs -f job/dsv41-flash-bench-sweep
kubectl cp <pod>:/results ./results/sweep && kubectl delete job dsv41-flash-bench-sweep

# AgentX agentic replay (agentx-harness 1.0.7, C=16 and C=32, 30 min each)
kubectl apply -f dsv41-flash-agentx-bench-job.yaml
kubectl logs -f job/dsv41-agentx-bench
kubectl cp <pod>:/results ./results/agentx && kubectl delete job dsv41-agentx-bench
```

---

## Benchmark Results (Initial, October 20, 2026)

### ISL/OSL Sweep — `sglang.bench_serving`

Measured with `python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-range-ratio 1.0 --seed 20261020` (default `temperature 0.0`, 1 warm-up request, `/flush_cache` before every run) from a pod on the same node, against the deployment above. Prompt counts are 3×C for 1k/1k and 8k/1k, and 2×C (min 64) for 1k/8k.

#### Throughput & Speculative Decoding

| Scenario (ISL / OSL) | C | Completed | Duration (s) | Req/s | Input tok/s | Output tok/s | Total tok/s | Peak Output tok/s | Avg Accept Len |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1k_1k`** (1024 / 1024) | 32 | 96 | 92.60 | 1.04 | 1,061.6 | 1,061.6 | 2,123.2 | 1,691.0 | 4.79 |
| **`1k_1k`** (1024 / 1024) | 64 | 192 | 158.88 | 1.21 | 1,237.4 | 1,237.4 | 2,474.9 | 2,059.0 | 4.61 |
| **`1k_1k`** (1024 / 1024) | 128 | 384 | 295.51 | 1.30 | 1,330.7 | **1,330.7** | **2,661.3** | 2,355.0 | 4.81 |
| **`8k_1k`** (8192 / 1024) | 32 | 96 | 193.14 | 0.50 | 4,071.9 | 509.0 | 4,580.9 | 1,866.0 | 5.26 |
| **`8k_1k`** (8192 / 1024) | 64 | 192 | 367.77 | 0.52 | 4,276.8 | 534.6 | 4,811.3 | 2,212.0 | 5.18 |
| **`8k_1k`** (8192 / 1024) | 128 | 384 | 734.80 | 0.52 | **4,281.1** | 535.1 | **4,816.2** | 2,380.0 | 5.19 |
| **`1k_8k`** (1024 / 8192) | 32 | 64 | 382.81 | 0.17 | 171.2 | 1,369.6 | 1,540.8 | 1,917.0 | **5.33** |
| **`1k_8k`** (1024 / 8192) | 64 | 128 | 577.03 | 0.22 | 227.2 | **1,817.2** | 2,044.4 | 2,260.0 | 5.28 |
| **`1k_8k`** (1024 / 8192) | 128 | 256 | 1,154.11 | 0.22 | 227.1 | 1,817.1 | 2,044.3 | 2,345.0 | 5.24 |

#### Latency Summary (ms)

| Scenario | C | Median TTFT | P99 TTFT | Mean TPOT | Median TPOT | Median ITL | P99 ITL | Median E2E |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1k_1k`** | 32 | 422.1 | 6,233.6 | 24.14 | 21.84 | 16.12 | 104.69 | 23,398.9 |
| **`1k_1k`** | 64 | 876.0 | 12,354.4 | 42.97 | 39.23 | 26.85 | 191.28 | 43,350.1 |
| **`1k_1k`** | 128 | 3,340.1 | 24,559.5 | 81.58 | 75.09 | 55.65 | 336.67 | 84,978.9 |
| **`8k_1k`** | 32 | 2,890.3 | 42,866.7 | 50.77 | 50.34 | 15.91 | 474.92 | 59,187.5 |
| **`8k_1k`** | 64 | 4,264.4 | 85,519.1 | 97.67 | 95.16 | 26.84 | 763.95 | 115,444.6 |
| **`8k_1k`** | 128 | 12,882.9 | 172,394.0 | 193.89 | 193.15 | 56.50 | 1,531.61 | 228,310.6 |
| **`1k_8k`** | 32 | 1,455.7 | 6,082.1 | 18.00 | 17.14 | **15.90** | 54.08 | 143,074.1 |
| **`1k_8k`** | 64 | 2,509.8 | 12,200.0 | 30.06 | 28.57 | 26.58 | 82.14 | 237,497.3 |
| **`1k_8k`** | 128 | 6,415.1 | 34,547.0 | 61.82 | 59.14 | 55.93 | 170.91 | 492,664.7 |

<details>
<summary>Full latency percentiles (Mean / P50 / P90 / P95 / P99, ms)</summary>

| Scenario | C | Metric | Mean | Median (P50) | P90 | P95 | P99 |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| `1k_1k` | 32 | TTFT | 1,994.2 | 422.1 | 5,995.9 | 6,232.9 | 6,233.6 |
| `1k_1k` | 32 | TPOT | 24.14 | 21.84 | 32.38 | 38.19 | 43.73 |
| `1k_1k` | 32 | ITL | 24.14 | 16.12 | 50.82 | 53.24 | 104.69 |
| `1k_1k` | 32 | E2E | 26,690.0 | 23,398.9 | 35,921.8 | 40,558.4 | 50,377.5 |
| `1k_1k` | 64 | TTFT | 3,438.9 | 876.0 | 10,883.9 | 12,064.2 | 12,354.4 |
| `1k_1k` | 64 | TPOT | 42.97 | 39.23 | 61.24 | 66.76 | 86.87 |
| `1k_1k` | 64 | ITL | 42.97 | 26.85 | 75.66 | 95.34 | 191.28 |
| `1k_1k` | 64 | E2E | 47,401.2 | 43,350.1 | 65,656.4 | 75,590.5 | 92,964.1 |
| `1k_1k` | 128 | TTFT | 6,879.5 | 3,340.1 | 19,380.2 | 22,744.7 | 24,559.5 |
| `1k_1k` | 128 | TPOT | 81.58 | 75.09 | 111.12 | 123.62 | 161.54 |
| `1k_1k` | 128 | ITL | 81.58 | 55.65 | 138.34 | 173.08 | 336.67 |
| `1k_1k` | 128 | E2E | 90,331.9 | 84,978.9 | 121,405.8 | 136,251.7 | 170,833.4 |
| `8k_1k` | 32 | TTFT | 9,885.4 | 2,890.3 | 32,410.9 | 38,959.8 | 42,866.7 |
| `8k_1k` | 32 | TPOT | 50.77 | 50.34 | 76.68 | 82.57 | 94.58 |
| `8k_1k` | 32 | ITL | 50.77 | 15.91 | 47.57 | 243.83 | 474.92 |
| `8k_1k` | 32 | E2E | 61,821.8 | 59,187.5 | 92,742.2 | 100,696.1 | 105,913.4 |
| `8k_1k` | 64 | TTFT | 18,358.6 | 4,264.4 | 61,851.8 | 74,990.7 | 85,519.1 |
| `8k_1k` | 64 | TPOT | 97.67 | 95.16 | 145.34 | 164.38 | 176.94 |
| `8k_1k` | 64 | ITL | 97.67 | 26.84 | 254.46 | 305.33 | 763.95 |
| `8k_1k` | 64 | E2E | 118,272.8 | 115,444.6 | 180,144.4 | 195,756.8 | 209,984.9 |
| `8k_1k` | 128 | TTFT | 39,687.3 | 12,882.9 | 123,635.7 | 151,287.9 | 172,394.0 |
| `8k_1k` | 128 | TPOT | 193.89 | 193.15 | 292.55 | 318.13 | 372.37 |
| `8k_1k` | 128 | ITL | 193.89 | 56.50 | 338.40 | 739.87 | 1,531.61 |
| `8k_1k` | 128 | E2E | 238,039.2 | 228,310.6 | 364,425.5 | 395,216.9 | 434,404.3 |
| `1k_8k` | 32 | TTFT | 2,709.7 | 1,455.7 | 6,050.5 | 6,081.8 | 6,082.1 |
| `1k_8k` | 32 | TPOT | 18.00 | 17.14 | 21.06 | 21.89 | 29.01 |
| `1k_8k` | 32 | ITL | 18.00 | 15.90 | 20.99 | 30.10 | 54.08 |
| `1k_8k` | 32 | E2E | 150,154.6 | 143,074.1 | 177,412.4 | 185,280.4 | 241,577.9 |
| `1k_8k` | 64 | TTFT | 4,548.9 | 2,509.8 | 12,040.3 | 12,144.3 | 12,200.0 |
| `1k_8k` | 64 | TPOT | 30.06 | 28.57 | 37.13 | 38.77 | 44.69 |
| `1k_8k` | 64 | ITL | 30.06 | 26.58 | 39.76 | 53.18 | 82.14 |
| `1k_8k` | 64 | E2E | 250,791.0 | 237,497.3 | 310,471.3 | 325,359.5 | 375,047.4 |
| `1k_8k` | 128 | TTFT | 10,491.4 | 6,415.1 | 22,819.3 | 24,138.0 | 34,547.0 |
| `1k_8k` | 128 | TPOT | 61.82 | 59.14 | 73.60 | 79.31 | 89.56 |
| `1k_8k` | 128 | ITL | 61.82 | 55.93 | 83.19 | 110.71 | 170.91 |
| `1k_8k` | 128 | E2E | 516,880.2 | 492,664.7 | 617,661.2 | 670,339.5 | 745,788.0 |

</details>

#### Observations

- **Decode saturates at ~1.8k output tok/s.** At `1k/8k`, `C=64` and `C=128` deliver the same 1,817 output tok/s; doubling concurrency only doubles per-request latency (median E2E 237 s → 493 s). `C=64` is the sweet spot for decode-heavy traffic; `C=32` gives the best interactive feel (15.9 ms median ITL).
- **Prefill is the bottleneck for long inputs.** `8k/1k` tops out at ~4.3k input tok/s at every concurrency, and with `--max-prefill-tokens 8192` each 8k prompt occupies a full prefill step, so TTFT queues up (P99 172 s at `C=128`). Keep `C ≤ 32–64` for prompt-heavy workloads on this configuration.
- **DSPARK accept length is 4.6–5.3 of 5** on this random-token, greedy benchmark; treat it as an upper bound (the AgentX replay below measured 3.3).
- **The 256k-safe memory settings cost nothing.** A sweep against the first (crashed) configuration (`--mem-fraction-static 0.75`, `--chunked-prefill-size 16384`, `--max-prefill-tokens 32768`, 1M context) gave 2,360 / 2,428 total tok/s at `1k/1k` `C=64/128` and 1,698 / 1,802 output tok/s at `1k/8k` `C=64/128` — i.e. the current config is 1–9 % faster while being able to prefill 256k-token contexts.

### AgentX — Agentic Replay (256k traces)

Run with [`dsv41-flash-agentx-bench-job.yaml`](./dsv41-flash-agentx-bench-job.yaml): `agentx-harness==1.0.7`, scenario `inferencex-agentx-mvp`, public dataset `semianalysis_cc_traces_weka_062126_256k`, `--max-context-length 262144`, streaming chat completions, `ignore_eos:true`, `--cache-bust first_turn_prefix`, `--system-idle-gap-cap-seconds 10`, `--random-seed 20260707`, 1,800 s profiling per concurrency after a 30-request warm-up.

#### `C=16` (completed; `C=32` is running and will be added)

| Aggregate | Value |
| :--- | :--- |
| Benchmark duration | 1,797.7 s |
| Requests completed / errors | **295 / 0** (1 in-flight request cancelled at the 30-min cutoff) |
| Request throughput | 0.16 req/s |
| Output token throughput | 126.6 tok/s (232,859 completion tokens total) |
| Input token throughput | 15,334 tok/s (28.2 M prompt tokens total) |
| Prompt-cache read (server `cached_tokens`) | **94.68 %** (theoretical prefix-cache hit 95.22 %) |
| Effective concurrency (AIPerf) | avg 3.5, max 7 — the replay honors the traces' inter-turn gaps (capped at 10 s) and sessions block on earlier turns, so far fewer than 16 requests are in flight at once |
| DSPARK accept rate / length (server metrics) | 45.8 % / **3.29** mean (p50 3.17, p90 3.92) |

| Per-request metric | avg | p50 | p90 | p99 | max |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TTFT (ms)** | 1,555.4 | **742.4** | 3,003.8 | 12,342.5 | 15,364.6 |
| **Inter-token latency (ms)** | 29.71 | **26.78** | 40.24 | 96.41 | 110.44 |
| **Request latency (ms)** | 21,952.4 | 10,283.8 | 44,872.2 | 182,051.5 | 375,876.8 |
| **Output tok/s per user** | 39.83 | 37.35 | 58.01 | 79.85 | 234.22 |
| Input sequence length (tokens) | 95,643 | 95,875 | 143,275 | 218,090 | 254,216 |
| Output sequence length (tokens) | 789 | 282 | 1,491 | 6,947 | 15,568 |

> **Note:** AIPerf reported counter resets on several SGLang server-side metrics while it was scraping, so the server-side speculative-decoding numbers above carry a "statistics may be inaccurate" warning from the tool; the client-side latency/throughput metrics are unaffected. Metric coverage passed (TTFT 98.7 %, ITL 99.9 %, required ≥ 95 %).
