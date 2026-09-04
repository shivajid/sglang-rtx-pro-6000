# zai-org/GLM-5.3

Serving recipes for the **full GLM-5.3** (~755B MoE + DSA reasoning model, FP8) on Google Cloud — **G4** (RTX PRO 6000, SM120) in FP8 across two nodes: 2 nodes × 8 RTX PRO 6000 Blackwell (SM120), **TP=8 × PP=2, DP=8 attention** (16 GPUs total).

This page is for the full checkpoint. For the smaller, single-node `GLM-5.3-Flash` variant see [`../GLM5.3-Flash/`](../GLM5.3-Flash/README.md).

| Recipe | Hardware | Quantization | Config | Status |
|--------|----------|--------------|--------|--------|
| **FP8 · 2 nodes (Stock)** | 16× RTX PRO 6000 | `zai-org/GLM-5.3` | [`sglang-2node-glm53.yaml`](./sglang-2node-glm53.yaml) | ✅ Working (Stock dev image) |
| [**FP8 · 2 nodes (Tuned MoE)**](#results--validation) | 16× RTX PRO 6000 | `zai-org/GLM-5.3` | [`fp8/sglang-glm53-fp8-2node.yaml`](./fp8/sglang-glm53-fp8-2node.yaml) | ✅ Deployed + correctness-verified (GSM8K 0.900 / 0.925) |

> 📖 Rendered docs: **[GLM on the site](https://shivajid.github.io/sglang-rtx-pro-6000/)**

---

## Model

The full GLM-5.3 is the **GLM-5.1/5.2 architecture family**, not the Flash variant:

| Property | Value |
|----------|-------|
| Architecture | `GlmMoeDsaForCausalLM` |
| Decoder layers | **78** (first 3 dense, then MoE) |
| Experts | 256 routed + 1 shared, top-8 |
| Hidden size | 6144 |
| MLA geometry | `kv_lora_rank=512`, `qk_nope_head_dim=192`, `qk_rope_head_dim=64` (standard 576-dim) |
| Quantization | FP8 (e4m3 dynamic, weight block 128×128) |
| Context | 1,048,576 tokens |
| Weights | **~753 GB** (141 safetensors shards) |
| MTP | 1 `num_nextn_predict_layers` |

### Why this is straightforward on SM120 (unlike GLM-5.3-Flash)

- It uses the **standard 576-dim MLA** layout (nope 192 + rope 64), the same as GLM-5.1/5.2 — **not** the Flash variant's unusual `qk_rope_head_dim=0` / 256-dim KV path, which needed custom tilelang patches.
- No KDA linear-attention layers and no mHC norm — it is plain DeepSeek-style DSA MoE.
- SGLang's `_dsa_split_backend_resolution` already has a **dedicated SM120 FP8 path** for `GlmMoeDsaForCausalLM`: on compute-capability 12 with an FP8 KV cache it routes DSA to `flashinfer_sparse_mla`. FP8 KV is the default on Blackwell, so the model lands in the working path out of the box.

### VRAM sizing

~753 GB of FP8 weights does **not** fit on a single 8× 96 GB node (768 GB total, minus CUDA context + KV cache). Two nodes (`TP=8 · PP=2`, 16 GPUs) gives ~47 GB/GPU of weights — comfortable headroom for KV cache and activation buffers.

---

## Results & Validation

**Deployed and correctness-verified** on `shivaji-minimax-g4-384-cluster` (`g4-384-np-0`), 2026-09-01.

| Check | Result |
|-------|--------|
| Server startup (2-node TP8·PP2, `flashinfer` attn + `trtllm` DSA) | ✅ Up — `The server is fired up and ready to roll` on both nodes |
| GSM8K correctness gate (40 Q, `--dsa-*-backend trtllm`) | ✅ **Accuracy 0.900, Invalid 0.000** |
| GSM8K with `--dsa-*-backend flashinfer_sparse_mla` | ❌ **Accuracy 0.000, Invalid 0.525** (degenerate output) |

### Why `flashinfer_sparse_mla` fails here (and `trtllm` works)

The full GLM-5.3 on SM120 + FP8-KV with `--attention-backend flashinfer` + `flashinfer_sparse_mla` DSA comes up **cleanly** (no crash) but emits garbage — completions degenerate and GSM8K scores **0.000**. This is the same silent-corruption signature as the 1-node GLM-5.2-NVFP4 build. Forcing `trtllm` for both DSA prefill and decode fixes it (**0.900**). The server stays up and throughput looks fine either way, so **always run the GSM8K gate before benchmarking** — a "healthy" server is not proof of correctness.

### MoE kernel tuning

The deployment injects a tuned SM120 Triton fused-MoE config (`E=256,N=256,K=6144,fp8_w8a8,block=[128,128]`, 18 batch sizes) via the `glm53-moe-tuned-config` ConfigMap or baked-in image. Startup confirms `Using MoE kernel config from .../E=256,N=256,...RTX_PRO_6000....json`. Correctness unchanged: **GSM8K 0.900 / 0.000**. The config lives at [`fp8/tuned/`](./fp8/tuned/) and in `gs://northam-ce-mlai-tpu-glm53-artifacts/moe-tuned/`.

Still untuned (perf headroom, not correctness):
- **MoE down-projection TMA** (`..._down.json`) — reuses the up-proj config; tune with `tuning_fused_moe_triton_sep.py` + captured `topk_ids` (same as GLM-5.3-Flash's pending step).
- **Dense FP8 GEMM** (`N=2624,K=6144` under `quantization/configs/`) — qkv/o_proj/dense MLP.

### Deployment Image

The serving StatefulSet runs the custom image `us-central1-docker.pkg.dev/northam-ce-mlai-tpu/glm53/glm53-fp8:sm120-moe-tuned-v1` (base `lmsysorg/sglang:dev-cu13` + the tuned MoE config baked into `triton_3_7_1/`). Also pushed to `aurius/glm53-fp8:sm120-moe-tuned-v1`. Re-validated on this image: **GSM8K 0.925 / 0.000**.

### Throughput Status

A low-concurrency sweep (c8/16/32) showed decode is currently **~6 tok/s/request** at low batch (and the server is unstable above ~c64 — crash-loops under sustained load). This is expected: the **MoE down-projection** and **dense FP8 GEMM** are still on default (untuned) configs, and TP8 collectives ride PCIe. Until those are tuned (and/or `flashinfer_cutlass` is enabled), benchmark numbers are not meaningful, so the sweep is deferred. The next optimization pass should target: MoE down-proj TMA tune, dense-GEMM tune, `flashinfer_cutlass` MoE runner, and load-stability at c≥64.

---

## Kubernetes Deployment (GKE + Hyperdisk ML)

```bash
# 1. Provision weights on Hyperdisk ML (once) — ~753 GB
kubectl apply -f fp8/glm53-hdml-writer.yaml      # PV + writer PVC (RWO)
kubectl apply -f fp8/glm53-downloader-job.yaml   # downloads zai-org/GLM-5.3 to the disk
kubectl logs -f job/glm53-hdml-downloader

# 2. Rebind the volume ReadOnlyMany and start the server
kubectl apply -f fp8/glm53-hdml-ro.yaml          # PV + PVC (ROX) shared by both nodes
kubectl apply -f fp8/sglang-glm53-fp8-2node.yaml
kubectl port-forward svc/sglang-serving-on-master 8000:8000
```

The server loads weights from the local path `/models/GLM-5.3`, mounted read-only from the shared Hyperdisk ML disk (`glm53-hdml-pvc`), so neither node re-downloads the 753 GB checkpoint. To pull from HF directly instead, set `MODEL_NAME` back to `zai-org/GLM-5.3` and remove the `model-disk` volume mount.

> **Target cluster:** `shivaji-minimax-g4-384-cluster` (zone `us-central1-f`, project `northam-ce-mlai-tpu`), GPU nodepool **`g4-384-np-0`**. The `volumeHandle` in the PV manifests points at a Hyperdisk ML disk named `glm53-hyperdisk-ml` in `us-central1-f` — the disk must be created in that zone (Hyperdisk ML is zonal and must be co-located with the nodepool).
>
> ```bash
> gcloud container clusters get-credentials shivaji-minimax-g4-384-cluster \
>   --zone us-central1-f --project northam-ce-mlai-tpu
> ```
>
> **Create the weight disk (once).** Hyperdisk ML provisions throughput that counts against the regional `HDML_TOTAL_THROUGHPUT` quota (100 GB/s in `us-central1`). A default 1000 GB disk asks for ~30 GB/s and can fail with `Quota 'HDML_TOTAL_THROUGHPUT' exceeded` if other HDML disks are provisioned in the region. Create with an explicit lower throughput to fit (12 GB/s is ample for weight serving):
>
> ```bash
> gcloud compute disks create glm53-hyperdisk-ml \
>   --zone us-central1-f --project northam-ce-mlai-tpu \
>   --type hyperdisk-ml --size 1000 --provisioned-throughput 12000
> ```

> ⚠️ **Shell-comment footgun:** do **not** put `#` comment lines inside the backslash-continued launch command in the manifest. In a `\`-continuation, `#` comments out the *rest of the logical line*, silently dropping every flag after it (the server then boots with defaults — port 30000, `flashinfer_sparse_mla`, `mem 0.65`, no parsers — and looks "up" while misconfigured). Keep the command free of inline comments; put rationale in YAML comments *outside* the `args` string or in this README instead.

### Correctness gate — run this before benchmarking

```bash
python3 -m sglang.test.few_shot_gsm8k --num-questions 50 --port 8000
# Expect: Accuracy ~0.900, Invalid 0.000
```

If accuracy collapses or completions degenerate to `!`, the DSA backend flags did not take effect.

---

## Docker image & Running Outside GKE

| | |
|---|---|
| **Image** | `lmsysorg/sglang:dev-cu13` (or custom tuned image) |
| **Registry** | [docker.io/lmsysorg/sglang](https://hub.docker.com/r/lmsysorg/sglang) |
| **Pull policy** | `Always` |
| **Base** | CUDA 13 runtime, SGLang dev branch — includes Blackwell (SM120) kernels |
| **Extra deps** | `python3 -m pip install distro` at container start |
| **GPU req** | NVIDIA driver ≥ 580 (CUDA 13 compatible), 8× RTX PRO 6000 per node |

### Running outside GKE (plain Docker, 2 nodes)

The same image runs the 2-node recipe on any two 8× RTX PRO 6000 hosts — no Kubernetes required. Run **identical commands on both hosts**, substituting `$NODE_RANK` (0 or 1) and `$MASTER_HOST` (rank-0 IP/hostname, reachable from both). Weights must be present at `/models/GLM-5.3` on **both** nodes (or set `MODEL_NAME=zai-org/GLM-5.3` to pull from HF — 753 GB per node).

```bash
docker run -d --name glm53-2node \
  --gpus all \
  --ipc=host \
  --shm-size=64g \
  --cap-add=IPC_LOCK \
  --network host \
  -v /models/GLM-5.3:/models/GLM-5.3:ro \
  -e NODE_RANK=0 \
  -e MASTER_HOST=10.0.0.1 \
  -e MODEL_NAME=/models/GLM-5.3 \
  -e HF_TOKEN=<your_hf_token> \
  lmsysorg/sglang:dev-cu13 \
  bash -c '
    python3 -m pip install distro &&

    # NCCL over PCIe/Ethernet (see NCCL table below)
    export NCCL_SOCKET_IFNAME=eth0,eth1 NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=SYS
    export NCCL_SOCKET_NTHREADS=8 NCCL_NSOCKS_PERTHREAD=8 NCCL_MIN_NCHANNELS=8
    export NCCL_ALLOC_P2P_NET_LL_BUFFERS=1 NCCL_NVLS_ENABLE=0 NCCL_CUMEM_ENABLE=0

    # SGLang runtime
    export SGLANG_PP_LAYER_PARTITION=38,40
    export SGLANG_SET_CPU_AFFINITY=1 OMP_NUM_THREADS=24
    export SAFETENSORS_FAST_GPU=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export SGLANG_ENABLE_DEEP_GEMM=0 SGLANG_ENABLE_JIT_DEEPGEMM=0

    python3 -m sglang.launch_server \
      --model $MODEL_NAME --quantization fp8 \
      --tensor-parallel-size 8 --pipeline-parallel-size 2 \
      --nnodes 2 --node-rank $NODE_RANK --dist-init-addr $MASTER_HOST:5000 \
      --dp-size 8 --enable-dp-attention \
      --attention-backend flashinfer --dsa-prefill-backend trtllm --dsa-decode-backend trtllm \
      --kv-cache-dtype fp8_e4m3 --page-size 64 --mem-fraction-static 0.80 \
      --moe-a2a-backend none --ep-size 1 --moe-runner-backend triton \
      --disable-shared-experts-fusion --disable-radix-cache \
      --reasoning-parser glm45 --tool-call-parser glm47 \
      --trust-remote-code --host 0.0.0.0 --port 8000'
```

Docker-flag rationale:

| Flag | Why |
|------|-----|
| `--network host` | The API port (8000), metrics (8080), and the torch rendezvous (5000) must be directly reachable — host networking avoids NAT breaking NCCL socket transport. |
| `--ipc=host` | Shared-memory transport for intra-node NCCL/CUDA IPC across the 8 GPUs. |
| `--shm-size=64g` | Generous /dev/shm for the TP=8 collectives' staging buffers. |
| `--cap-add=IPC_LOCK` | Allows pinning CUDA/NCCL host memory (required by NCCL P2P registration). |
| `-v /models/GLM-5.3:...:ro` | Read-only weight mount — never let the server write into the checkpoint dir. |

---

## SGLang Launch Configuration

Run on **each** node (`NODE_RANK` = 0 or 1; both nodes start concurrently):

```bash
python3 -m sglang.launch_server \
    --model /models/GLM-5.3 \
    --quantization fp8 \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --nnodes 2 \
    --node-rank $NODE_RANK \
    --dist-init-addr <master-host>:5000 \
    --dp-size 8 \
    --enable-dp-attention \
    --attention-backend flashinfer \
    --dsa-prefill-backend trtllm \
    --dsa-decode-backend trtllm \
    --kv-cache-dtype fp8_e4m3 \
    --moe-a2a-backend none \
    --ep-size 1 \
    --moe-runner-backend triton \
    --disable-shared-experts-fusion \
    --page-size 64 \
    --mem-fraction-static 0.80 \
    --disable-radix-cache \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --trust-remote-code \
    --host 0.0.0.0 --port 8000
```

### Key parameters

| Flag | Value | What it does / why it matters |
|------|-------|-------------------------------|
| `--model` | `/models/GLM-5.3` | ~753 GB FP8 checkpoint, 78 decoder layers. Point at a local path if weights are pre-downloaded; use `zai-org/GLM-5.3` to pull from HF (slow — 753 GB per node). |
| `--quantization` | `fp8` | Checkpoint is FP8; weights stay FP8 on device. |
| `--tensor-parallel-size` | `8` | Shards attention heads and MoE experts across the 8 GPUs **within** each node. |
| `--pipeline-parallel-size` | `2` | Splits the 78 layers across the 2 nodes (38 + 40). Cross-node traffic is one point-to-point send/recv per boundary — far cheaper than per-layer AllReduce over PCIe/Ethernet. |
| `--nnodes` / `--node-rank` | `2` / `0…1` | Multi-node setup. Rank 0 hosts the rendezvous endpoint. |
| `--dist-init-addr` | `<master>:5000` | torch-distributed rendezvous; must resolve to the rank-0 node from both nodes. |
| `--dp-size` + `--enable-dp-attention` | `8` | **DP attention**: DSA attention/KV is data-parallel (replicated per DP group) instead of TP-sharded. Removes the attention AllReduce from the TP group — the dominant cost in single-node TP8 runs. |
| `--attention-backend` | `flashinfer` | General attention kernel backend. |
| `--dsa-prefill-backend` / `--dsa-decode-backend` | `trtllm` | Kernels for **DSA (DeepSeek-style Sparse Attention)**. TRT-LLM kernels work here because the 2-node layout avoids the SM120 shared-memory limit that forces tilelang on single-node GLM-5.3-Flash. |
| `--kv-cache-dtype` | `fp8_e4m3` | FP8 KV cache — halves KV footprint for long-context reasoning at negligible quality cost. |
| `--moe-a2a-backend` / `--ep-size` | `none` / `1` | **No expert parallelism**: experts are replicated per DP group and dispatched locally. Avoids all-to-all collectives, which would be painful over PCIe + Ethernet. |
| `--moe-runner-backend` | `triton` | Triton fused-MoE kernels (DeepGEMM is disabled on this hardware — see env vars below). |
| `--disable-shared-experts-fusion` | — | Keeps GLM-5.3's shared expert separate from the routed experts (the single-node Flash recipe fuses it; this build does not). |
| `--page-size` | `64` | Larger KV pages — fewer metadata ops for the 1k/8k reasoning profile. |
| `--mem-fraction-static` | `0.80` | Fraction of VRAM for weights + KV cache. 0.80 leaves headroom for DP-attention KV replication; raise only after verifying no OOM during load. |
| `--disable-radix-cache` | — | Prefix caching off: reasoning traffic is largely unique-prompt, and the radix tree is per-DP-group under DP attention anyway. |
| `--reasoning-parser` / `--tool-call-parser` | `glm45` / `glm47` | Parse R1-style `</think>` CoT and GLM tool-call syntax into the OpenAI-compatible API. |
| `--trust-remote-code` | — | Required — GLM-5.3 ships custom modeling code. |

### The one non-obvious setting: `SGLANG_PP_LAYER_PARTITION=38,40`

Export this **before launching** (it's an env var, not a CLI flag):

```bash
export SGLANG_PP_LAYER_PARTITION=38,40
```

GLM-5.3 has 78 decoder layers. The split **cannot** be the naive even `39,39`: GLM-5.3 uses `index_topk_freq=4` + `index_skip_topk_offset=3`, so a DSA layer computes fresh top-k indices only on "full-topk" layers (those ≡ 2 mod 4, plus dense layers 0–2) and **reuses** a previous layer's indices everywhere else. A PP boundary must land on a full-topk layer; otherwise the upstream stage must forward `topk_indices` across the boundary, which the decode CUDA-graph capture does not do — it asserts:
```
PP stage ending at layer N must forward DSA topk_indices because the next stage starts on a skip-topk layer
```
and the scheduler dies (exit code -3). Layer 39 is a skip-topk layer (crash); layer 38 is full-topk, so `38,40` is valid and near-balanced.

### Other SGLang / runtime env vars

| Env var | Value | Why |
|---------|-------|-----|
| `SGLANG_SET_CPU_AFFINITY` | `1` | Pin server threads — matters on 384-vCPU nodes. |
| `OMP_NUM_THREADS` | `24` | CPU thread budget per process. |
| `SAFETENSORS_FAST_GPU` | `1` | GPU-direct safetensors load — speeds up the 753 GB load. |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Reduces fragmentation during the multi-hundred-GB sharded load. |
| `SGLANG_ENABLE_DEEP_GEMM` / `SGLANG_ENABLE_JIT_DEEPGEMM` | `0` / `0` | DeepGEMM is disabled; MoE runs through the Triton runner on SM120. |

---

## NCCL environment variables

G4 nodes have **no NVLink/NVSwitch and no InfiniBand** — intra-node collectives traverse PCIe Gen 5, inter-node collectives traverse up to 400 Gbps Ethernet. Stock NCCL settings assume datacenter fabric and stall or pick slow paths here. Set on **both** nodes:

```bash
export NCCL_SOCKET_IFNAME=eth0,eth1
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=SYS
export NCCL_SOCKET_NTHREADS=8
export NCCL_NSOCKS_PERTHREAD=8
export NCCL_MIN_NCHANNELS=8
export NCCL_ALLOC_P2P_NET_LL_BUFFERS=1
export NCCL_NVLS_ENABLE=0
export NCCL_CUMEM_ENABLE=0
```

| Env var | Value | Why |
|---------|-------|-----|
| `NCCL_SOCKET_IFNAME` | `eth0,eth1` | Use both NICs for the socket transport — required to fill 400 Gbps inter-node bandwidth. |
| `NCCL_IB_DISABLE` | `1` | G4 has no InfiniBand; explicitly disable so NCCL doesn't probe/hang on IB detection. |
| `NCCL_P2P_LEVEL` | `SYS` | Permit P2P through the PCIe root complex (GPU↔GPU across the CPU socket). |
| `NCCL_SOCKET_NTHREADS` | `8` | More threads per socket channel → higher per-collective socket throughput. |
| `NCCL_NSOCKS_PERTHREAD` | `8` | More sockets per thread → parallelizes the inter-node stream. |
| `NCCL_MIN_NCHANNELS` | `8` | Force a minimum channel count so large collectives (MoE AllReduce, PP send/recv) get enough concurrency. |
| `NCCL_ALLOC_P2P_NET_LL_BUFFERS` | `1` | Pre-allocate low-latency network P2P buffers — avoids first-collective allocation stalls on the inter-node path. |
| `NCCL_NVLS_ENABLE` | `0` | NVLink SHARP (NVLS) is an NVSwitch feature; off prevents fallback stalls on PCIe-only nodes. |
| `NCCL_CUMEM_ENABLE` | `0` | Disable cuMem allocations (NVLink-era path); plain allocator is more reliable on this topology. |
| `NCCL_DEBUG` | `INFO` *(optional)* | Uncomment in the YAML when debugging NCCL route/transport selection. |

> **Note:** single-node TP8 runs (e.g. GLM-5.3-Flash) need none of these — the socket/P2P tuning matters only for the 2-node job.

---

## Quick reference: single-node vs 2-node

| | GLM-5.3-**Flash** (1 node) | **GLM-5.3** (2 nodes, this page) |
|---|---|---|
| GPUs | 8× RTX PRO 6000 | 16× RTX PRO 6000 |
| Parallelism | TP8 | TP8 × PP2, DP8 attention |
| DSA backend | tilelang (SM120 smem cap forces it) | trtllm |
| KV cache | bf16 | fp8_e4m3 |
| Shared expert | fused into routed experts (E=289) | separate (`--disable-shared-experts-fusion`) |
| NCCL tuning | not needed | required (see table above) |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `PP stage ending at layer N must forward DSA topk_indices ... skip-topk layer` during CUDA-graph capture | PP boundary on a skip-topk layer — keep `SGLANG_PP_LAYER_PARTITION=38,40` (boundary ≡ 2 mod 4). |
| NCCL hangs or falls back to slow paths | Verify the full NCCL env block is exported on **both** nodes; keep `NVLS=0`, `CUMEM=0`, `IB_DISABLE=1`. |
| OOM during load | 753 GB weights + DP-attention KV replication — confirm fp8 KV is active before raising `--mem-fraction-static` above 0.80. |
| Rendezvous timeout on node 1 | `--dist-init-addr` must resolve to the rank-0 host from both nodes; start both nodes concurrently. |
| Model completions degenerate / GSM8K = 0.000 | Silent corruption on `flashinfer_sparse_mla`. Set `--dsa-prefill-backend trtllm --dsa-decode-backend trtllm`. |

---

## Files

| Path | Purpose |
|------|---------|
| [`sglang-2node-glm53.yaml`](./sglang-2node-glm53.yaml) | 2-node FP8 serving StatefulSet + Services (stock `lmsysorg/sglang:dev-cu13` image) |
| [`fp8/sglang-glm53-fp8-2node.yaml`](./fp8/sglang-glm53-fp8-2node.yaml) | 2-node FP8 serving StatefulSet + Services with baked-in tuned MoE kernel image |
| [`fp8/glm53-hdml-writer.yaml`](./fp8/glm53-hdml-writer.yaml) · [`fp8/glm53-downloader-job.yaml`](./fp8/glm53-downloader-job.yaml) · [`fp8/glm53-hdml-ro.yaml`](./fp8/glm53-hdml-ro.yaml) | Hyperdisk ML provisioning: write → download → rebind read-only |
| [`fp8/glm53-moe-tune-job.yaml`](./fp8/glm53-moe-tune-job.yaml) | Triton fused-MoE tuner Job (never-exits, uploads result to GCS) |
| [`fp8/tuned/`](./fp8/tuned/) | Tuned SM120 MoE kernel config (`E=256,N=256,K=6144,fp8_w8a8`), also in ConfigMap `glm53-moe-tuned-config` |

---

## Attribution

- Model: [`zai-org/GLM-5.3`](https://huggingface.co/zai-org/GLM-5.3) (FP8)
- Engine: [SGLang](https://github.com/sgl-project/sglang) (`lmsysorg/sglang:dev-cu13`)
- Hardware: 2× GCP `g4-standard-384` (8× NVIDIA RTX PRO 6000 Blackwell, SM120 each)
- Sibling recipes: [GLM-5.3-Flash · 1 node](../GLM5.3-Flash/README.md) · [GLM-5.2 · NVFP4/FP8](../GLM5.2/README.md)
