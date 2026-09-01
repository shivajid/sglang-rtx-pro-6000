# zai-org/GLM-5.3

Serving recipe for **GLM-5.3** (full-size MoE + DSA reasoning model, FP8) on G4 hardware —
2 nodes × 8 RTX PRO 6000 Blackwell (SM120), **TP=8 × PP=2, DP=8 attention** (16 GPUs total).

| Recipe | Hardware | Quantization | Config | Status |
|--------|----------|--------------|--------|--------|
| **FP8 · 2 nodes** | 16× RTX PRO 6000 | `zai-org/GLM-5.3` (FP8) | [`sglang-2node-glm53.yaml`](./sglang-2node-glm53.yaml) | ✅ Working |

> ⏳ **Benchmarks pending.** Recipe is validated for bring-up (server healthy, decode CUDA
> graphs captured on both PP stages). Standard-profile numbers will be added once the sweep
> completes.

This page documents the **launch command and the environment variables** — the parameters
that actually determine whether GLM-5.3 serves correctly and how fast. The Kubernetes
wrapper (StatefulSet, services, Hyperdisk ML mount) is in the YAML above; nothing on this
page is GKE-specific except where noted.

---

## SGLang launch command

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

GLM-5.3's DSA layers reuse a previous layer's top-k indices on "skip-topk" layers
(`index_topk_freq=4`, `index_skip_topk_offset=3`); only **full-topk** layers
(≈ 2 mod 4, plus dense layers 0–2) recompute indices. A pipeline boundary must land on a
full-topk layer, or decode CUDA-graph capture aborts:

```
PP stage ending at layer N must forward DSA topk_indices ... skip-topk layer
```

The even split (39/39) is a skip-topk layer → **crash**. `38,40` is a full-topk boundary and
near-balanced (78 = 38 + 40). If a future checkpoint changes `index_topk_freq`, re-derive it.

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

G4 nodes have **no NVLink/NVSwitch and no InfiniBand** — intra-node collectives traverse
PCIe Gen 5, inter-node collectives traverse up to 400 Gbps Ethernet. Stock NCCL settings
assume datacenter fabric and stall or pick slow paths here. Set on **both** nodes:

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

> **Note:** single-node TP8 runs (e.g. GLM-5.3-Flash) need none of these — the socket/P2P
> tuning matters only for the 2-node job.

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

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `PP stage ending at layer N must forward DSA topk_indices ... skip-topk layer` during CUDA-graph capture | PP boundary on a skip-topk layer — keep `SGLANG_PP_LAYER_PARTITION=38,40` (boundary ≡ 2 mod 4). |
| NCCL hangs or falls back to slow paths | Verify the full NCCL env block is exported on **both** nodes; keep `NVLS=0`, `CUMEM=0`, `IB_DISABLE=1`. |
| OOM during load | 753 GB weights + DP-attention KV replication — confirm fp8 KV is active before raising `--mem-fraction-static` above 0.80. |
| Rendezvous timeout on node 1 | `--dist-init-addr` must resolve to the rank-0 host from both nodes; start both nodes concurrently. |

## Attribution

- Model: [`zai-org/GLM-5.3`](https://huggingface.co/zai-org/GLM-5.3) (FP8)
- Engine: [SGLang](https://github.com/sgl-project/sglang) (`lmsysorg/sglang:dev-cu13`)
- Hardware: 2× GCP `g4-standard-384` (8× NVIDIA RTX PRO 6000 Blackwell, SM120 each)
- Sibling recipes: [GLM-5.3-Flash · 1 node](../GLM5.3-Flash/README.md) · [GLM-5.2 · NVFP4/FP8](../GLM5.2/README.md)
