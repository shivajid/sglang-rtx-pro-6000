# OSS Model Benchmarks on GCP G4

Optimized GKE configurations and benchmarks for serving LLMs on GCP G4 instances using SGLang.

## Infrastructure
- **GPU**: NVIDIA RTX PRO 6000 Blackwell (SM120)
- **Architecture Details**: [Technical Specifications: GCP G4](gcp_g4_specs.md)
- **Serving Framework**: [SGLang](https://github.com/sglang-project/sglang) (`dev-cu13`, `0.5.10.post1`)

## Performance Benchmarks (Latest)

| Model | Quantization | Setup | Total Throughput (tok/s) | Peak Throughput (tok/s) | TPOT (ms) |
|-------|--------------|-------|---------------------------|--------------------------|-----------|
| DeepSeek-V3.2 | FP8 | 2 Nodes (16x RTX 6000) | 3324.21 | 4951.00 | 149.29 |
| DeepSeek-V3.2 | NVFP4 | 1 Node (8x RTX 6000) | 3012.42 | 2046.00 | 106.03 |
| GLM-5.1 | FP8 | 2 Nodes (16x RTX 6000) | 3125.35 | 4092.00 | 155.26 |

*Benchmarks conducted using `inf` request rate and 512 max concurrency.*

## Project Structure

- `models/`: Model-specific SGLang job configurations and benchmarks.
  - `DeepSeekv3-2/`: Configs for DeepSeek-V3 and V2.5.
    - `fp8/`: Optimized 2-node FP8 serving setup.
    - `nvp4/`: Native FP4 serving using `modelopt_fp4` with EAGLE speculative decoding.
  - `GLM5.1/`: Optimized configurations and results for GLM-5.1 FP8.
  - `KimiK2.5/`: Configurations for Kimi-K2.5 (INT4/FP8).
- `gkecluster/`: Infrastructure-as-Code for GKE provisioning.
  - `createCluster_template.sh`: Automated script to provision VPC, networking, and GKE clusters optimized for Blackwell G4.
  - `createCluster_README.md`: Detailed setup and usage instructions for the GKE template.
- `benchmarks/`: Global benchmark definitions and performance scripts.
  - `benchmark-dsv2.yaml`: Load generator config for DeepSeek.
  - `benchmark-glm51.yaml`: Load generator config for GLM-5.1.
- `gcp_g4_specs.md`: Detailed hardware and infrastructure specifications.

## Key Updates (April 2026)
- **Native FP4 Support**: Successfully validated DeepSeek-V3.2 on a single node using NVFP4 quantization, achieving performance comparable to 2-node FP8 setups.
- **Speculative Decoding**: Integrated EAGLE speculative decoding for DeepSeek-V3.2 NVFP4, significantly reducing TPOT.
- **GLM-5.1 Optimization**: Completed FP8 serving optimization for GLM-5.1 on 2-node Blackwell clusters.
- **Distributed SGLang**: Standardized 2-node configurations for ultra-large models using `pipeline-parallel-size 2` and `tensor-parallel-size 8`.

## GKE Infrastructure Setup

The `gkecluster` directory contains a comprehensive template for provisioning a GKE environment optimized for SGLang:
- **Custom VPC**: High MTU (8896) for optimized multi-node traffic.
- **Multi-Networking**: Specialized network interfaces for distributed inference.
- **Blackwell Node Pools**: Automated creation of `g4-standard-384` pools with 8x RTX PRO 6000 Blackwell GPUs.
- **Benchmarking Isolation**: Dedicated node pools for load generators to ensure clean performance metrics.

## Usage

Model-specific deployment instructions and performance results are located within the `models/` directory.

## Contributing

This repository is updated as new optimization techniques (e.g., native FP4 serving) and models are validated on the G4 architecture.
