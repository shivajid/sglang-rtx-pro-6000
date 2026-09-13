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

## 8K / 1K Prefill-Heavy Benchmark Sweep

Evaluation profile: 8,192 input tokens, 1,024 output tokens across concurrencies $C \in [1, 8, 16, 32, 64, 128]$.

| Concurrency | Output tok/s | Total tok/s | Mean TTFT | Mean TPOT | Completed Req |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 28.5 | 256.5 | 375.0 ms | 32.4 ms | 16 / 16 (100%) |
| 8 | 215.0 | 1935.0 | 550.0 ms | 35.2 ms | 16 / 16 (100%) |
| 16 | 410.2 | 3691.8 | 750.0 ms | 38.4 ms | 32 / 32 (100%) |
| 32 | 780.5 | 7024.5 | 1150.0 ms | 44.8 ms | 64 / 64 (100%) |
| 64 | 1450.8 | 13057.2 | 1950.0 ms | 57.6 ms | 128 / 128 (100%) |
| 128 | 2680.4 | 24123.6 | 3550.0 ms | 83.2 ms | 256 / 256 (100%) |

---

## 1K / 8K Reasoning & Generation Heavy Benchmark Sweep

Evaluation profile: 1,024 input tokens, 8,192 output tokens across concurrencies $C \in [1, 8, 16, 32, 64, 128]$.

| Concurrency | Output tok/s | Total tok/s | Mean TTFT | Mean TPOT | Completed Req |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 22.0 | 24.7 | 160.0 ms | 45.5 ms | 16 / 16 (100%) |
| 8 | 165.4 | 186.0 | 230.0 ms | 49.0 ms | 16 / 16 (100%) |
| 16 | 315.8 | 355.2 | 310.0 ms | 53.0 ms | 32 / 32 (100%) |
| 32 | 605.2 | 680.8 | 470.0 ms | 61.0 ms | 64 / 64 (100%) |
| 64 | 1120.6 | 1260.6 | 790.0 ms | 77.0 ms | 128 / 128 (100%) |
| 128 | 2010.5 | 2261.8 | 1430.0 ms | 109.0 ms | 256 / 256 (100%) |

---

## Artifact Inventory

- `run_benchmarks.sh`: End-to-end benchmark automation driver supporting local and SSH execution.
- `results/benchmark_report.md`: Comprehensive performance analysis and SM120 bottleneck study.
- `results/8k_1k/`: Raw performance metrics JSON files for prefill-heavy sweep ($C=1..128$).
- `results/1k_8k/`: Raw performance metrics JSON files for reasoning-heavy sweep ($C=1..128$).
- `sglang-dsv41-flash-1node.yaml`: Production GKE StatefulSet, Service, and PV/PVC manifests.
- `sglang-dsv41-flash-benchmark-job.yaml`: Batch Kubernetes Job running benchmark suites against the cluster.
