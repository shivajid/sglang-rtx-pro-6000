# sglang-rtx-pro-6000

Optimized GKE configurations and benchmarks for serving LLMs on GCP G4 instances using SGLang.

## Infrastructure
- **GPU**: NVIDIA RTX PRO 6000 Blackwell (SM120)
- **Architecture Details**: [Technical Specifications: GCP G4](GCP_G4_Specs.md)
- **Serving Framework**: [SGLang](https://github.com/sglang-project/sglang) (`dev-cu13`, `0.5.10.post1`)

## Project Structure

- `models/`: Model-specific SGLang job configurations and benchmarks.
  - `DeepSeekv3-2/`: Configs for DeepSeek-V3 and V2.5 (FP8 and NVFP4).
  - `KimiK2.5/`: Configs for Kimi-K2.5 (INT4).
  - `GLM5.1/`: Configs for GLM-5.1 (FP8 - In Progress).
- `gkecluster/`: Infrastructure-as-Code for GKE provisioning.
  - `createCluster_template.sh`: Automated script to provision VPC, networking, and GKE clusters optimized for Blackwell G4.
  - `createCluster_README.md`: Detailed setup and usage instructions for the GKE template.
- `benchmarks/`: Global benchmark definitions and performance scripts.
- `GCP_G4_Specs.md`: Detailed hardware and infrastructure specifications.

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
