# sglang-rtx-pro-6000

This repo is for showcasing optimized GKE configs for inferencing state-of-the-art LLM Models on GCP G4 machines.

- **Infrastructure**: Powered by [GCP G4 Machine Architecture](GCP_G4_Specs.md) featuring NVIDIA RTX PRO 6000 Blackwell (SM120) GPUs.
- **Models**:
  - DeepSeekv3-2
     - Base FP8
     - NVFP4
  - KimiK 2.5
     - Base int4
  - GLM 5.1
     - fp8 (coming)

The base models are run on 2-node GCP G4 machines. G4 machines are based on NVIDIA's RTX PRO 6000 Blackwell server edition GPUs, offering massive 96GB VRAM and native FP4 support.

We have used **SGLang** as the serving framework. The SGLang versions used are `lmsysorg/sglang:dev-cu13` and `lmsys/sglang:0.5.10.post1`.




## Project Structure

- `models/`: Model-specific SGLang job configurations and results.
  - `DeepSeekv3-2/`: Configs for DeepSeek-V3/V2.5 models (FP8 and NVP4).
  - `KimiK2.5/`: Configs for Kimi-K2.5.
  - `GLM5.1/`: Configs for GLM-5.1.
- `benchmarks/`: General benchmark definitions and scripts.

## Usage

Documentation for specific model setups can be found within their respective directories under `models/`.

## Contributing

This repository is a work in progress as new models and optimizations are added.
