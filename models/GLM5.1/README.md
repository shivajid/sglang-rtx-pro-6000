# GLM-5.1 on GCP G4

Optimized configurations and benchmarks for [Zhipu AI's GLM-5.1](https://huggingface.co/zai-org/GLM-5.1-FP8) on GCP G4 instances using SGLang.

## Model Overview
GLM-5.1 is an advanced language model optimized for high-throughput inference. These configurations leverage FP8 quantization for efficient distributed serving.

## Serving Configuration

### FP8 (2-Node Setup)
- **Model**: `zai-org/GLM-5.1-FP8`
- **Setup**: 2x `g4-standard-384` (16x RTX PRO 6000)
- **Parallelism**: Tensor Parallel (TP) 8, Pipeline Parallel (PP) 2, Data Parallel (DP) 8
- **Quantization**: FP8
- **Attention Backend**: `flashinfer`
- **KV Cache**: `bfloat16`
- **Memory Fraction**: 0.90
- **Serving Image**: `lmsysorg/sglang:dev-cu13`

### NVFP4 (1-Node Setup)
- **Model**: `lukealonso/GLM-5.1-NVFP4`
- **Setup**: 1x `g4-standard-384` (8x RTX PRO 6000)
- **Parallelism**: Tensor Parallel (TP) 8
- **Quantization**: `modelopt_fp4`
- **Speculative Algorithm**: `NEXTN` (Steps: 3, Draft Tokens: 4)
- **Attention Backend**: `flashinfer`
- **KV Cache**: `bfloat16`
- **Memory Fraction**: 0.88
- **Serving Image**: `voipmonitor/sglang:cu130`

## Benchmark Results

| Metric | FP8 Result (2-node) | NVFP4 Result (1-node) |
|--------|---------------------|-----------------------|
| Output Throughput | 2785.55 tok/s | 1407.71 tok/s |
| Total Throughput | 3125.35 tok/s | 1585.08 tok/s |
| Peak Output Throughput | 4092.00 tok/s | 770.00 tok/s |
| Mean TPOT | 155.26 ms | 81.01 ms |
| Median TTFT | 344.23 ms | 443131.76 ms |

**Attribution**: Optimization strategy and benchmark execution by **Shivaji Dutta**.

## Usage
Apply the distributed manifest:
```bash
kubectl apply -f sglang-glm51-v1-04232026.yaml
```
