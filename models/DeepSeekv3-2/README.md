# DeepSeek-V3/V3.2 on GCP G4

Optimized configurations and benchmarks for [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) on GCP G4 instances using SGLang.

## Model Overview
DeepSeek-V3 is a strong Mixture-of-Experts (MoE) model. These configurations demonstrate high-performance serving using both standard FP8 and native NVFP4 quantization.

## Serving Configurations

### 1. NVFP4 (Single Node - Optimized)
- **Model**: `nvidia/DeepSeek-V3.2-NVFP4`
- **Setup**: 1x `g4-standard-384` (8x RTX PRO 6000)
- **Quantization**: `modelopt_fp4`
- **Speculative Decoding**: EAGLE (3 steps)
- **Manifest**: `nvp4/sglang-ds32-job-v2.yaml`

### 2. FP8 (Distributed - 2 Nodes)
- **Model**: `deepseek-ai/DeepSeek-V3.2`
- **Setup**: 2x `g4-standard-384` (16x RTX PRO 6000)
- **Parallelism**: TP8, PP2
- **Manifest**: `fp8/sglang-2node-setup-optimized.yaml`

## Benchmark Results

| Configuration | Output Throughput | Total Throughput | Mean TPOT | Median TTFT |
|---------------|-------------------|------------------|-----------|-------------|
| **NVFP4 (1 Node)** | 2675.33 tok/s | 3012.42 tok/s | 106.03 ms | 18.68 s |
| **FP8 (2 Nodes)** | 2962.79 tok/s | 3324.21 tok/s | 149.29 ms | 0.32 s |

**Attribution**: Optimization strategy and benchmark execution by **Shivaji Dutta**.

## Usage
Refer to the respective subdirectories for YAML manifests and detailed results.
