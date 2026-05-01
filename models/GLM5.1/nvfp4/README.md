# GLM-5.1 NVFP4 Benchmarks

This directory contains configurations and results for GLM-5.1 using NVFP4 quantization on G4 instances.

## Configurations

### 1-Node Setup
- **Config**: [sglang-glm51-nvfp4-1node.yaml](./sglang-glm51-nvfp4-1node.yaml)
- **GPUs**: 8x RTX PRO 6000 (1x g4-standard-384)
- **Quantization**: `modelopt_fp4`
- **Speculative Decoding**: NEXTN (Steps: 3, Draft: 4)

### 2-Node Setup
- **Config**: [sglang-glm51-nvfp4-2node.yaml](./sglang-glm51-nvfp4-2node.yaml)
- **GPUs**: 16x RTX PRO 6000 (2x g4-standard-384)
- **Parallelism**: TP8, PP2, DP8
- **Quantization**: `modelopt_fp4`

## Benchmark Results Comparison

| Metric | 1-Node (NVFP4) | 2-Node (NVFP4) |
|--------|----------------|----------------|
| Output Throughput | 1490.31 tok/s | 3075.85 tok/s |
| Total Throughput | 1672.11 tok/s | 3451.06 tok/s |
| Peak Output Throughput | 734.00 tok/s | 4606.00 tok/s |
| Mean TPOT | 73.82 ms | 141.36 ms |
| Median TTFT | 1064328.97 ms | 331.11 ms |

## Detailed Logs
- [1-Node Results](./results/benchmark_results_1node.md)
- [2-Node Results](./results/benchmark_results_2node.md)
