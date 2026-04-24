# GLM-5.1 on GCP G4

Optimized configurations and benchmarks for [Zhipu AI's GLM-5.1](https://huggingface.co/zai-org/GLM-5.1-FP8) on GCP G4 instances using SGLang.

## Model Overview
GLM-5.1 is an advanced language model optimized for high-throughput inference. These configurations leverage FP8 quantization for efficient distributed serving.

## Serving Configuration
- **Model**: `zai-org/GLM-5.1-FP8`
- **Setup**: 2x `g4-standard-384` (16x RTX PRO 6000)
- **Parallelism**: TP8, PP2
- **Quantization**: FP8
- **Serving Image**: `lmsysorg/sglang:dev-cu13`

## Benchmark Results

| Metric | Result |
|--------|--------|
| Output Throughput | 2785.55 tok/s |
| Total Throughput | 3125.35 tok/s |
| Peak Output Throughput | 4092.00 tok/s |
| Mean TPOT | 155.26 ms |
| Median TTFT | 344.23 ms |

**Attribution**: Optimization strategy and benchmark execution by **Shivaji Dutta**.

## Usage
Apply the distributed manifest:
```bash
kubectl apply -f sglang-glm51-v1-04232026.yaml
```
