# OSS Model Benchmarks on GCP G4

Optimized GKE configurations and benchmarks for serving LLMs on GCP G4 instances using SGLang.

## Infrastructure
- **GPU**: NVIDIA RTX PRO 6000 Blackwell (SM120)
- **Architecture Details**: [Technical Specifications: GCP G4](./gcp_g4_specs.md)
- **Serving Framework**: [SGLang](https://github.com/sgl-project/sglang) (`dev-cu13`, `0.5.10.post1`)

## Performance Benchmarks (Latest)

| Model | Quantization | Setup | Output Throughput (tok/s) | Total Throughput (tok/s) | Peak Throughput (tok/s) | TPOT (ms) |
|-------|--------------|-------|---------------------------|--------------------------|-------------------------|-----------|
| [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) | FP8 | 2 Nodes (16x RTX 6000) | 2962.79 | 3324.21 | 4951.00 | 149.29 |
| [DeepSeek-V3.2](https://huggingface.co/nvidia/DeepSeek-V3.2-NVFP4) | NVFP4 | 1 Node (8x RTX 6000) | 2675.33 | 3012.42 | 2046.00 | 106.03 |
| [GLM-5.1](https://huggingface.co/zai-org/GLM-5.1-FP8) | FP8 | 2 Nodes (16x RTX 6000) | 2785.55 | 3125.35 | 4092.00 | 155.26 |
| [GLM-5.1](https://huggingface.co/lukealonso/GLM-5.1-NVFP4) | NVFP4 | 1 Node (8x RTX 6000) | 1490.31 | 1672.11  | 734.00  | 73.82 |
| [Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5) | INT4* | 2 Nodes (16x RTX 6000) | 3069.15 | 3443.55 | 6889.00 | 147.45 |

*Benchmarks conducted using `inf` request rate and 512 max concurrency. Tests utilized a random dataset with 1024 input tokens and 8192 output tokens (1536 total prompts). The load generator was isolated on a dedicated CPU-only node pool to ensure zero interference with GPU performance.*

*\*Kimi-K2.5 uses native INT4 quantization for model weights and FP8 for the KV cache to optimize memory efficiency and inference speed.*

## Project Structure

- `models/`: Model-specific SGLang job configurations and benchmarks.
  - `DeepSeekv3-2/`: Configs for DeepSeek-V3 and V2.5.
    - `fp8/`: Optimized 2-node FP8 serving setup.
    - `nvp4/`: Native FP4 serving using `modelopt_fp4` with EAGLE speculative decoding.
  - `GLM5.1/`: Optimized configurations and results for GLM-5.1.
    - `fp8/`: 2-node FP8 serving optimization.
    - `nvfp4/`: 1-node native FP4 serving with NEXTN speculative decoding.
  - `KimiK2.5/`: Configurations for Kimi-K2.5.
- `gkecluster/`: Infrastructure-as-Code for GKE provisioning.
  - `createCluster_template.sh`: Automated script to provision VPC, networking, and GKE clusters optimized for Blackwell G4.
  - `createCluster_README.md`: Detailed setup and usage instructions for the GKE template.
- `benchmarking_scripts/`: Global benchmark definitions and performance scripts.
  - `benchmark-dsv2.yaml`: Load generator config for DeepSeek.
  - `benchmark-glm51.yaml`: Load generator config for GLM-5.1.
- `gcp_g4_specs.md`: Detailed hardware and infrastructure specifications.

## Key Updates (April 2026)
- **Native FP4 Support**: Successfully validated DeepSeek-V3.2 and GLM-5.1 on single-node setups using NVFP4 quantization, achieving high efficiency on Blackwell architecture.
- **Speculative Decoding**: Integrated EAGLE for DeepSeek-V3.2 and NEXTN for GLM-5.1 NVFP4 to optimize token generation speeds.
- **GLM-5.1 Optimization**: Completed both FP8 (2-node) and NVFP4 (1-node) serving optimizations.
- **Distributed SGLang**: Standardized 2-node configurations for ultra-large models using `pipeline-parallel-size 2` and `tensor-parallel-size 8`.

## GKE Infrastructure Setup

The `gkecluster` directory contains a comprehensive template for provisioning a GKE environment optimized for SGLang:
- **Custom VPC**: High MTU (8896) for optimized multi-node traffic.
- **Multi-Networking**: Specialized network interfaces for distributed inference.
- **Blackwell Node Pools**: Automated creation of `g4-standard-384` pools with 8x RTX PRO 6000 Blackwell GPUs.
- **Benchmarking Isolation**: Dedicated node pools for load generators to ensure clean performance metrics.

## Viewing Detailed Benchmark Results

Detailed performance logs, including TTFT/TPOT latency distributions and throughput metrics, are located within each model's `results` directory:

- [DeepSeek-V3.2 (FP8): models/DeepSeekv3-2/fp8/results/benchmark_results.md](./models/DeepSeekv3-2/fp8/results/benchmark_results.md)
- [DeepSeek-V3.2 (NVFP4): models/DeepSeekv3-2/nvp4/results/benchmark_results.md](./models/DeepSeekv3-2/nvp4/results/benchmark_results.md)
- [GLM-5.1 (FP8): models/GLM5.1/results/benchmark-results.md](./models/GLM5.1/results/benchmark-results.md)
- [GLM-5.1 (NVFP4): models/GLM5.1/nvfp4/results/benchmark_results.md](./models/GLM5.1/nvfp4/results/benchmark_results.md)
- [Kimi-K2.5 (FP8): models/KimiK2.5/results/benchmark_results.md](./models/KimiK2.5/results/benchmark_results.md)

## Usage

For detailed instructions on deploying models and running benchmarks, see the [Benchmarking Guide](./benchmarking_guide.md).

Each model directory also contains a dedicated `README.md` with specific optimization details and attribution.

## Contributing

This repository is updated as new optimization techniques (e.g., native FP4 serving) and models are validated on the G4 architecture.
