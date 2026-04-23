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
- `benchmarks/`: Global benchmark definitions and performance scripts.
- `GCP_G4_Specs.md`: Detailed hardware and infrastructure specifications.

## Usage

Model-specific deployment instructions and performance results are located within the `models/` directory.

## Contributing

This repository is updated as new optimization techniques (e.g., native FP4 serving) and models are validated on the G4 architecture.
