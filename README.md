# sglang-rtx-pro-6000

This repo is for showcasing optimzed GKE configs for inferencing state of the art LLM Models on GCP G4 machines.

- Models
  - DeepSeekv3-2
     - Base FP8
     - NVFP4
  - KimiK 2.5
     - Base int4
  - GLM 5.1
     - fp8 (coming)

 The base models are run on 2 Node GCP G4 machines. G4 machines are based on NVIDIA's RTX Pro 6000 nodes.
 
 We have used sgLang as the serving framework. The sgLang versions used are lmsysy/sglang:dev-cu13 and lmsys/sglang:0.5.10.post1.




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
