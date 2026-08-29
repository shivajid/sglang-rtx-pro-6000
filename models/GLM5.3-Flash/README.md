# zai-org/GLM-5.3-Flash

Serving recipes and benchmarks for **GLM-5.3-Flash** (DeepSeek-style MoE + DSA, FP8, R1-style CoT
reasoning) on a single **G4** node (8× RTX PRO 6000 Blackwell, SM120), TP=8.

| Recipe | Hardware | Quantization | Config | Status |
|--------|----------|--------------|--------|--------|
| [**FP8 · 1 node, tuned MoE**](#fp8--1-node-tuned-moe) | 8× RTX PRO 6000 | FP8 + tuned SM120 MoE Triton kernels | [`glm53-flash-g4-tuned.yaml`](./glm53-flash-g4-tuned.yaml) | ✅ Working — recommended |
| [**FP8 · 1 node** (TP8)](#fp8--1-node-tp8) | 8× RTX PRO 6000 | `zai-org/GLM-5.3-Flash` (FP8) | [`glm53-flash-g4.yaml`](./glm53-flash-g4.yaml) | ✅ Working |

> 📖 Full sweep report: **[results/benchmark_sweep_results.md](./results/benchmark_sweep_results.md)**

---

## Results

Standard profile: random dataset, `inf` request rate, isolated load generator. Workloads swept
across `1k/1k` (balanced), `1k/8k` (reasoning), and `8k/1k` (long-context prefill), concurrency
32 → 256.

| Workload | Concurrency | Output tok/s | Total tok/s | Median TTFT | Median ITL |
|----------|:---:|:---:|:---:|:---:|:---:|
| **8k / 1k** (prefill) | 128 | 1,064.2 | **9,481.7** | 828.8 ms | 53.7 ms |
| **1k / 1k** (balanced) | 256 | 1,913.1 | 3,870.0 | 7,694 ms | 69.4 ms |
| **1k / 1k** (balanced) | 128 | 1,616.5 | 3,194.5 | 299.1 ms | 52.8 ms |
| **1k / 8k** (reasoning) | 256 | **2,579.8** | 2,898.7 | 34,619 ms | 70.6 ms |
| **1k / 8k** (reasoning) | 128 | 1,790.5 | 2,018.5 | 395.5 ms | 54.0 ms |

**Headline numbers:**
- **Peak total throughput:** **9,481.7 tok/s** (`8k/1k` @ C=128 — 8,417.5 prefill + 1,064.2 decode).
- **Peak decode generation:** **2,579.8 tok/s** (`1k/8k` @ C=256).
- **Recommended production envelope:** **64 ≤ C ≤ 128** — near-peak saturation with sub-second
  median TTFT (204–695 ms) and stable ITL (~30–54 ms). Past C=128, requests queue behind KV-cache
  memory slots and TTFT climbs into the tens of seconds.

See the [full 12-run matrix and per-workload analysis](./results/benchmark_sweep_results.md).

---

## What's in the two configs

### FP8 · 1 node (TP8) — `glm53-flash-g4.yaml`
Baseline single-node bring-up. SGLang serving GLM-5.3-Flash on 8× RTX PRO 6000 with:
- `TENSOR_PARALLEL_SIZE=8`, `MOE_RUNNER_BACKEND=triton`, `KV_CACHE_DTYPE=bfloat16`.
- **DSA attention** on tilelang (`dsa_prefill_backend` / `dsa_decode_backend = tilelang`) — the
  flashinfer sparse-MLA kernel expects a 576-dim (512+64) KV layout; GLM-5.3 uses a rope=0,
  256-dim nope DSA layout, so on SM120 the backends default to tilelang (BF16 KV baseline).
- **CUDA graphs ON** for decode (`DISABLE_CUDA_GRAPH=0`), made possible by the SM120 tilelang
  `num_stages=1` fix (below).
- `hostIPC: true` + 64 Gi `/dev/shm` for NCCL/torch shared memory across the 8 GPUs.

### FP8 · 1 node, tuned MoE — `glm53-flash-g4-tuned.yaml`
Same deployment, but the image bakes in a **tuned Triton fused-MoE kernel config** for SM120.
Out of the box, SGLang logs:

```
Using default MoE kernel config. Performance might be sub-optimal!
Config file not found at .../triton_3_7_1/E=289,N=256,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition,dtype=fp8_w8a8,block_shape=[128, 128].json
```

GLM-5.3 fuses its single shared expert into the routed-expert tensor → **E = 288 + 1 = 289**,
`N = 2048 / TP8 = 256`, fp8_w8a8, block `[128,128]`. We generated the config by running
`benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py --model zai-org/GLM-5.3-Flash
--tp-size 8 --tune` on the node (~29 min, 1,280 configs × 18 batch sizes over 8 GPUs) and baked
the result into the image at the path SGLang expects. Two repo patches were needed:
1. A **`Glm5NextForConditionalGeneration` branch** in the tuning script (`common_utils.py`) —
   GLM-5.3's HF config uses `n_routed_experts` / `num_experts_per_tok` / `moe_intermediate_size`,
   not the Mixtral-style `num_local_experts`, so the stock tuner crashes.
2. The tuned filename must include **`dtype=fp8_w8a8`** (the tuner's filename helper drops the
   dtype, but the runtime loader requires it).

---

## SM120 kernel fixes baked into the image

GLM-5.3 would not serve on RTX PRO 6000 (SM120) without these patches:

- **tilelang DSA `num_stages=1`.** SM120 caps dynamic shared memory at 101,376 B (99 KB); the
  default `num_stages=2` pipeline in the tilelang sparse-MLA kernel exceeds it, so decode CUDA-graph
  capture fails with `Failed to set the allowed dynamic shared memory size`. The patch drops to a
  single pipeline stage on SM120, which both fixes serving and lets **decode CUDA graphs stay on**.
- **GLM-5.3 DSA backend override.** Forces `dsa_prefill_backend=dsa_decode_backend=tilelang` on
  SM120 for the rope=0 / BF16 geometry (see above).

---

## Profiling findings & roadmap

A `torch.profiler` kernel breakdown of the tuned TP8 deployment (decode-heavy window) shows where
GPU time actually goes — and what to attack next:

| Kernel family | % of GPU time | Note |
|---------------|:---:|------|
| **NCCL AllReduce (bf16)** | ~43% | TP8 collective over **PCIe** — the dominant cost |
| **fused_moe (Triton)** | ~25% | Already tuned (up-proj); down-proj TMA tune pending |
| **dense GEMM (bf16 cutlass/cuBLAS)** | ~17% | qkv/o_proj/dense MLP — candidate for fp8 tensor-core |
| **DSA attn (tilelang)** | ~9% | `num_stages=1` is a forced downgrade (smem cap) |
| linear attn (KDA/mamba), norm, sampling | ~4% | — |

Things we **tried or ruled out** on this hardware:

- **DP-attention (`--enable-dp-attention --dp-size 8 --enable-dp-lm-head`)** — *fails on SM120.*
  GLM-5.3 has `index_n_heads=32`; DP-attention replicates all 32 DSA heads per rank (vs 4/rank under
  TP8), so the tilelang decode kernel needs 141,312 B smem > the 99 KB cap → CUDA-graph capture
  crashes. The eliminated attention AllReduce would also traverse the same PCIe fabric, so expected
  net gain on this node is ≈ 0–7% even if the smem issue were fixed. **Not recommended on
  PCIe-only SM120** — it's designed for NVLink datacenter Blackwell (B200/GB200).
- **flashinfer / torch symm-mem allreduce fusion, NCCL NVLS, quant-comm** — all gated on SM90/SM100
  or NPU; **not applicable to SM120.** The node is also PCIe-only (no NVLink/NVSwitch), so
  `CustomAllReduceV2` self-disables ("not supported on more than two PCIe-only GPUs") and serving
  falls back to plain NCCL — exactly the ~43% slice above.
- **DCP (`--dcp-size`)** — decode context parallelism shards the KV cache, not the TP AllReduce, and
  adds its own all-gather/reduce-scatter. Doesn't address the MoE AllReduce bottleneck.

**Highest-ROI next steps:**
1. **Prefill CUDA graphs** — prefill currently runs eager (`prefill.backend='disabled'`), the main
   driver of TTFT under load. Enable via `cuda_graph_config.prefill.backend=breakable/tc_piecewise`.
2. **MoE down-projection TMA tune** (`tuning_fused_moe_triton_sep.py` + captured `topk_ids`) —
   refines the 25% MoE slice beyond the up-projection tune already applied.
3. **fp8 dense GEMM path** — route the bf16 dense projections (17%) through fp8 tensor-cores.
4. **`--enable-fused-moe-sum-all-reduce`** — fuse the MoE local-sum with the TP AllReduce to shave
   the dominant NCCL slice without needing NVLink.

---

## Reproducing

```bash
# Prereq: HF token secret
kubectl create secret generic hf-secret --from-literal=HF_TOKEN=<token>

# Deploy (tuned image: aurius/sglang-glm53-flash-rtxpr6000:sm120-moe-tuned-v1,
# the Docker Hub mirror of the tuned Artifact Registry build)
kubectl apply -f glm53-flash-g4-tuned.yaml

# Port-forward and benchmark
kubectl port-forward svc/glm53-flash 30000:80
python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 \
  --model zai-org/GLM-5.3-Flash --dataset-name random \
  --num-prompts 512 --random-input 1024 --random-output 8192 --max-concurrency 128
```

Model load is slow (~10 min for the FP8 checkpoint across 8 GPUs); the readiness probe allows 600 s.

---

## Attribution

- Model: [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash)
- Engine: [SGLang](https://github.com/sgl-project/sglang) (main + GLM-5.3 support, SM120 patches)
- Hardware: GCP `g4-standard-384` (8× NVIDIA RTX PRO 6000 Blackwell, SM120), GKE
