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
