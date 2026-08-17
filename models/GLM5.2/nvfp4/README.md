# GLM-5.2 NVFP4 Serving Configurations

This directory contains configurations for GLM-5.2 using NVFP4 quantization on G4 instances (NVIDIA RTX PRO 6000 Ada / SM120).

## Configurations

### 1-Node Setup (DP Attention on `g4-384-pool-pm-crwd`)
- **Config**: [sglang-glm52-nvfp4-1node.yaml](./sglang-glm52-nvfp4-1node.yaml)
- **Node Pool**: `g4-384-pool-pm-crwd` (`g4-standard-384`)
- **GPUs**: 8x RTX PRO 6000 (SM120)
- **Parallelism**: TP8, DP8 (`--enable-dp-attention`, `--ep-size 1`)
- **Weights / Checkpoint**: `nvidia/GLM-5.2-NVFP4`
- **MoE Runner Backend**: `flashinfer_cutlass`
- **Attention Backend**: `flashinfer`
- **KV Cache**: `fp8_e4m3`
- **Memory Fraction**: `0.975`
- **Serving Image**: `lmsysorg/sglang:dev-cu13`

#### Critical Configuration Details & Fixes
1. **Unquoted CUDA Graph Batch Sizes (`--cuda-graph-bs`)**:
   On Kubernetes/GKE, each batch size integer must be passed as an independent element in the `args` array (`"16"`, `"32"`, `"48"`, `"64"`). Passing them as a single space-separated string triggers an argparse integer conversion error and causes `CrashLoopBackOff`.
2. **Piecewise CUDA Graph Disabled (`--disable-piecewise-cuda-graph`)**:
   Prefill-CUDA-graph capture unconditionally accesses DSA indexer metadata, which is `None` with `--attention-backend flashinfer`, crashing rank schedulers during init (`AttributeError: 'NoneType' object has no attribute 'get_seqlens_expanded'`).
3. **TRTLLM DSA Backends Forced (`--dsa-prefill-backend trtllm --dsa-decode-backend trtllm`)**:
   Prevents silent corruption where auto-selected `flashinfer_sparse_mla` on SM120 FP8 KV cache emits NaN logits in decode (collapsing completions to token 0 / `!`). Forcing `trtllm` ensures GSM8K accuracy ~0.900 and enlarges KV cache token capacity.
4. **Environment Variables**:
   - `SGLANG_DISABLE_DSA_INDEXER_FUSION=1`
   - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   - NCCL optimizations for single-node 8-GPU interconnect.

#### Correctness Gate (Run before benchmarking)
```bash
python3 -m sglang.test.few_shot_gsm8k --num-questions 50 --port 8000
# Expect Accuracy ~0.900, Invalid 0.000
```

#### Benchmark Command
```bash
python3 -m sglang.bench_serving --backend sglang --model nvidia/GLM-5.2-NVFP4 \
  --dataset-name random --random-input-len 1024 --random-output-len 8192 --random-range-ratio 1.0 \
  --max-concurrency 512 --num-prompts 2048
```

---

### 2-Node Setup — ❌ not working
- **Config**: [sglang-glm52-nvfp4-2node_notworking.yaml](./sglang-glm52-nvfp4-2node_notworking.yaml)
- **GPUs**: 16x RTX PRO 6000 (2x g4-standard-384)
- **Parallelism**: TP8, PP2, DP8
- **Quantization**: `modelopt_fp4`
- **Serving Image**: `europe-west4-docker.pkg.dev/northam-ce-mlai-tpu/sglang-repo/sglang:glm-opt`
