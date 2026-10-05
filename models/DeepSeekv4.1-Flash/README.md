# DeepSeek-V4.1-Flash SGLang Serving on 8× NVIDIA RTX PRO 6000 (Blackwell SM120)

This repository provides the production serving recipe, Kubernetes deployment manifests, and empirical performance evaluation for serving **DeepSeek-V4.1-Flash** (552B parameter Mixture-of-Experts) on an 8× NVIDIA RTX PRO 6000 Blackwell GPU cluster using **SGLang**.

## System Architecture & Serving Recipe

- **Compute Hardware**: 8× NVIDIA RTX PRO 6000 Blackwell Server Edition (SM120, Compute Capability 12.0), 96 GB VRAM each, 768 GB aggregate VRAM.
- **Interconnect**: Intra-node PCIe Gen 5 x16 fabric (~50–55 GB/s bidirectional NCCL transfer bandwidth, no NVLink).
- **Model Architecture**: DeepSeek-V4.1-Flash (`DeepseekV4ForCausalLM`), Causal-Encoder-Decoder (CED) architecture with 8B active parameters during prefill and 16B active parameters during decode.
- **Serving Configuration**:
  - Tensor Parallelism: `--tp 8`
  - Quantization: CUTLASS MXFP4 routed experts, FP8 (`ue8m0`) dense/shared weights.
  - KV Cache: `--kv-cache-dtype fp8_e4m3` with `--page-size 256` (3.84M token capacity pool).
  - Prefill Scheduling: `--chunked-prefill-size 8192` with `--max-prefill-tokens 16384`.
  - Static Memory Fraction: `--mem-fraction-static 0.8` (reserves 15.5 GB headroom for intermediate activation buffers).
  - Associative Memory: Pinned host RAM Engram tables (`SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`).

---


---

## Artifact Inventory

- `run_benchmarks.sh`: End-to-end benchmark automation driver supporting local and SSH execution.
- `results/benchmark_report.md`: Comprehensive performance analysis and SM120 bottleneck study.
- `results/8k_1k/`: Raw performance metrics JSON files for prefill-heavy sweep ($C=1..128$).
- `results/1k_8k/`: Raw performance metrics JSON files for reasoning-heavy sweep ($C=1..128$).
- `sglang-dsv41-flash-1node.yaml`: Production GKE StatefulSet, Service, and PV/PVC manifests.
- `sglang-dsv41-flash-benchmark-job.yaml`: Batch Kubernetes Job running benchmark suites against the cluster.
