# Kimi-K2.5 on GCP G4

Optimized configurations and benchmarks for [Moonshot AI's Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5) on GCP G4 instances using SGLang.

## Model Overview
Kimi-K2.5 is a large-scale Mixture-of-Experts (MoE) model designed for high-performance reasoning and tool use.

## Serving Configuration
The model is served using a 2-node distributed SGLang setup:
- **Tensor Parallelism**: 8
- **Pipeline Parallelism**: 2
- **Quantization**: FP8 (KV Cache)
- **Serving Image**: `lmsysorg/sglang:v0.5.10.post1`

The configuration uses specialized reasoning and tool-call parsers optimized for Kimi's architecture.

## Benchmark Results
The following benchmarks were conducted on a cluster of 2x `g4-standard-384` instances (16x RTX PRO 6000 Blackwell GPUs).

| Metric | Result |
|--------|--------|
| Output Throughput | 3069.15 tok/s |
| Total Throughput | 3443.55 tok/s |
| Peak Output Throughput | 6889.00 tok/s |
| Mean TPOT | 147.45 ms |
| Median TTFT | 249.73 ms |

**Attribution**: Final benchmark numbers and optimization tuning provided by **Karim Roukoz**.

## Usage
To deploy Kimi-K2.5, apply the `sglang-ss.yaml` manifest:
```bash
kubectl apply -f sglang-ss.yaml
```
