# OSS Model Benchmarks on Google Cloud on Blackwell chips

This page has optimized recipes mostly for GCP G4 (RTX Pro 6000) machines.

But with launch of GLM5.2 and KimiK3, the bechmarks have been run GB300 and GB200. There are corresponding benchmarks and recipes.

Overall they have been updated.

📖 **[Link to Documentation](https://shivajid.github.io/sglang-rtx-pro-6000/)** 


Optimized GKE configurations and benchmarks for serving LLMs on GCP G4 instances.

## Infrastructure
- **GPU**: NVIDIA RTX PRO 6000 Blackwell (SM120)
- **Architecture Details**: [Technical Specifications: GCP G4](./gcp_g4_specs.md)
- **Serving Framework**: [SGLang](https://github.com/sgl-project/sglang) (`dev-cu13`, `0.5.10.post1`)

## Performance Benchmarks (Latest)

| Model | Quantization | Setup | Output Throughput (tok/s) | Total Throughput (tok/s) | Peak Throughput (tok/s) | TPOT (ms) |
|-------|--------------|-------|---------------------------|--------------------------|-------------------------|-----------|
| [nvidia/Kimi-K2.6-NVFP4](./models/KimiK2.6/nvfp4/results/benchmark-results.md) | NVFP4 | 2 Nodes (16x RTX 6000) | 3261.28 | 3662.79 | 4725.00 | 138.54 |
| [moonshotai/Kimi-K3](./models/kimik3/results/benchamrk_sweep_report.md) | BF16 | 4 Nodes (16x GB200) | 1666.54 | 1874.86 | — | 24.88 |
| [zai-org/GLM-5.2-FP8](./models/GLM5.2/fp8/results/benchmark_results.yaml) | FP8 | 2 Nodes (16x RTX 6000) | 1645.21 | 1855.07 | 2608.00 | 240.43 |
| [moonshotai/Kimi-K2.6](./models/KimiK2.6/results/benchmark_results.md) (wip)| INT4* | 1 Node (8x RTX 6000) (not optimized) | 1459.26 | 1637.28 | 850.00 | 82.43 |
| [nvidia/Kimi-K2.5-NVFP4](./models/KimiK2.5/nvfp4/results/benchmarks_2node.yaml) | NVFP4 | 2 Nodes (16x RTX 6000) | 3237.46 | 3632.39 | 5535.00 | 137.89 |
| [moonshotai/Kimi-K2.5](./models/KimiK2.5/results/benchmark_results.md) | INT4* | 2 Nodes (16x RTX 6000) | 3152.79 | 3537.39 | 4793.00 | 136.52 |
| [lukealonso/GLM-5.1-NVFP4](./models/GLM5.1/nvfp4/results/benchmark_results_2node.md) | NVFP4 | 2 Nodes (16x RTX 6000) | 3075.85 | 3451.06 | 4606.00 | 141.36 |
| [lukealonso/GLM-5.1-NVFP4](./models/GLM5.1/nvfp4/results/benchmark_results_1node.md) | NVFP4 | 1 Node (8x RTX 6000) | 1490.31 | 1672.11 | 734.00 | 73.82 |
| [zai-org/GLM-5.1-FP8](./models/GLM5.1/results/benchmark-results.md) | FP8 | 2 Nodes (16x RTX 6000) | 2785.55 | 3125.35 | 4092.00 | 155.26 |
| [nvidia/DeepSeek-V3.2-NVFP4](./models/DeepSeekv3-2/nvp4/results/benchmark_results.md) | NVFP4 | 1 Node (8x RTX 6000) | 2675.33 | 3012.42 | 2046.00 | 106.03 |
| [deepseek-ai/DeepSeek-V3.2](./models/DeepSeekv3-2/fp8/results/benchmark_results.md) | FP8 | 2 Nodes (16x RTX 6000) | 2962.79 | 3324.21 | 4951.00 | 149.29 |
| [Qwen/Qwen3.5-397B-A17B-FP8](./models/Qwen3.5-397B-A17B-FP8/results/hicache/benchmark_results.md) | FP8 | 1 Node (8x RTX 6000) | 390.65 | 8202.16 | 1120.00 | 100.59 |
| [datalab-to/chandra-ocr-2](./models/datalab2-ocr/benchmark_results.md)** | BF16| 1 Node (1x RTX 6000)| 2600.67 | 5267.08 | 4603.00| 32.47 |

**[openai/whisper-large-v3](./models/whisper-v3-large/results/benchmark_results.md)** - Since this is ASR model, we did not apply the standard ISL/OSL of 1K/8K and concurrancy of 512.

*Table last updated: July 31, 2026*
 
*Benchmarks conducted using `inf` request rate and 512 max concurrency. Tests utilized a random dataset with 1024 input tokens and 8192 output tokens (1536 total prompts). The load generator was isolated on a dedicated CPU-only node pool to ensure zero interference with GPU performance.*

*\*Kimi-K2.5 and Kimi-K2.6 use native INT4 quantization and KV cache optimization to improve memory efficiency and inference speed.*

**\** datalab-to/chandra-ocr-2 is an VLM model. We have run an image benchmark different for the rest of the models **

## [moonshotai/Kimi-K2.6](./models/KimiK2.6/agent_benchmark/README.md) Agentic Benchmark
[Detailed Configuration & Results](./models/KimiK2.6/agent_benchmark/)

Evaluates performance under real-world agentic traces with long sequences and high prompt volume. This benchmark uses non-standard traffic profiles and SGLang features like **HiCache** and **EAGLE3 Speculative Decoding** to evaluate performance on complex workloads.

### Benchmark Settings
- **Traffic Profile:** Simulated real-world agentic traces (Replay).
  - **Request Configuration:** Temperature=0.6, Top_P=0.95, Max Tokens=4096.
  - **Dataset:** Kimi K2.6 real-world agentic trace replay (Long-tail steps and sequences).
- **Parallelism:** 64 (Single-Node), 256 (Two-Node).
- **SGLang Features:** HiCache, EAGLE3 Speculative Decoding, SMG Router (Dual-node).
- **Environment:** TP=8 per node, GKE (Single-node), GCE (Dual-node).

| Metric | Single-Node (GKE) | Two-Node (GCE) |
| :--- | :---: | :---: |
| **Requests per Second** | 0.353 | 0.481 |
| **Total Tokens per Second** | 6,550.98 | 8,924.50 |
| **P50 Latency (s)** | 16.13 | 33.96 |
| **P99 Latency (s)** | 699.50 | 952.79 |
| **Prompt Cache Hit Rate** | **81.19%** | 0.00%* |

*\*Note: 0% hit rate on dual-node is a reporting limitation of the SMG router.*

## [Qwen/Qwen3.5-397B-A17B-FP8](./models/Qwen3.5-397B-A17B-FP8/BENCHMARK_REPORT.md) Latency Benchmark
[Detailed Configuration & Results](./models/Qwen3.5-397B-A17B-FP8/)

Focuses on latency characteristics of an ultra-large MoE model, comparing performance with and without **SGLang HiCache**. This benchmark evaluates the model's performance under ultra-large workload scenarios.

### Benchmark Settings
- **Traffic Profile:**
  - **Input Length:** 20,000 tokens
  - **Output Length:** 1,000 tokens
  - **Concurrency:** 40
  - **Number of Prompts:** 2,000
- **Total Tokens:** ~19.7M Input, ~986K Generated.
- **Server Configuration:** TP=8, Chunked Prefill (4096), Max Prefill (32768), Mixed Chunk Enabled.
- **HiCache Config:** `--enable-hierarchical-cache --hicache-ratio=2.0 --hicache-io-backend=kernel`.

| Metric | HiCache (Enabled) | No Radix Cache |
| :--- | :---: | :---: |
| **Median TTFT (ms)** | **1,054.01** | 1,128.88 |
| **Mean TTFT (ms)** | **1,121.17** | 1,371.28 |
| **Median TPOT (ms)** | 101.18 | **90.41** |
| **Mean TPOT (ms)** | 100.59 | **90.45** |

## [moonshotai/Kimi-K3](./models/kimik3/results/benchamrk_sweep_report.md) Performance Sweep
[Detailed Configuration & Results](./models/kimik3/)

Comprehensive evaluation of the **Kimi-K3** reasoning model across various workload patterns on GB200 infrastructure. This model leverages SGLang's multi-node optimizations and Blackwell-specific kernels.

### Benchmark Settings
- **Setup:** 4 Nodes (16x GB200 GPUs).
- **Architecture:** ARM64 with NCCL MNNVL and GIB plugin.
- **Workload Patterns:**
  - **Pattern A (1k/8k):** Reasoning-heavy, long output.
  - **Pattern B (8k/1k):** Context-heavy, prompt prefill.
  - **Pattern C (1k/1k):** Balanced conversational.

| Workload Pattern | Peak Total Throughput | Optimal Concurrency | Stream Speed (t/s) |
| :--- | :---: | :---: | :---: |
| **1k / 8k (Reasoning)** | 1,874.86 tok/s | 128 | 24.88 |
| **8k / 1k (Prompt)** | 2,731.09 tok/s | 128 | 24.26 |
| **1k / 1k (Balanced)** | **2,883.45 tok/s** | 256 | 23.68 |

## [nvidia/GLM-5.2-NVFP4](./models/GLM5.2/GB300_GLM52_single_host_setup.md) GB300 Validation
[GKE Configuration & Setup Guide](./models/GLM5.2/GB300_GLM52_single_host_setup.md)

Validated GLM-5.2 on **GCP A4X Max (GB300)** infrastructure. While formal throughput benchmarks were only conducted on G4, the GB300 setup was used for qualitative validation of the Blackwell-optimized SGLang stack.

- **Hardware:** GCP A4X Max (4x GPUs per node).
- **Quantization:** `modelopt_fp4`.
- **Note:** A production-ready GKE configuration for GB300 is available in the repository at [models/GLM5.2/GB300_GLM52_single_host_setup.md](./models/GLM5.2/GB300_GLM52_single_host_setup.md).

## [nvidia/Kimi-K2.6-NVFP4](./models/KimiK2.6/nvfp4/results/batch_one_bench_results.md) Batch Throughput Benchmark
[Detailed Configuration & Results](./models/KimiK2.6/nvfp4/)

This benchmark measures the raw throughput of the **Kimi-K2.6 NVFP4** model using a single large batch to evaluate peak processing capabilities on a 2-node setup.

### Benchmark Settings
- **Configuration**:
  - **Batch size**: 512
  - **Input sequence length**: 1024 tokens
  - **Output sequence length**: 8192 tokens
  - **Setup**: 2 Nodes (16x RTX 6000)
  - **Quantization**: FP4 (`modelopt_fp4`)

| Metric | Value |
| --- | --- |
| **Input Prefill Throughput** | **14,015.66 tokens/s** |
| **Output Decode Throughput** | **3,807.51 tokens/s** |
| **Overall Token Throughput** | **4,142.77 tokens/s** |
| **Average Generation Speed** | **487.57 tokens/s (per rank)** |

## Project Structure

- `models/`: Model-specific SGLang job configurations and benchmarks.
  - `DeepSeekv3-2/`: Configs for DeepSeek-V3 and V2.5.
  - `GLM5.1/`: Optimized configurations and results for GLM-5.1.
  - `GLM5.2/`: Benchmarks and Blackwell (GB300) setup guides for GLM-5.2.
  - `kimik3/`: Performance sweep and multi-node GB200 configs for Kimi-K3.
  - `KimiK2.5/`: Configurations for Kimi-K2.5.
  - `KimiK2.6/`: Agentic benchmark results and HiCache configurations.
  - `Qwen3.5-397B-A17B-FP8/`: Latency benchmarks for ultra-large MoE model.
- `gkecluster/`: Infrastructure-as-Code for GKE provisioning.
- `benchmarking_scripts/`: Global benchmark definitions and performance scripts.
  - `agentic_benchmark/`: Scripts for simulating agentic workloads.
- `gcp_g4_specs.md`: Detailed hardware and infrastructure specifications.

## Key Updates (July 2026)
- **Kimi-K3 Performance Sweep**: Completed a full concurrency sweep for Kimi-K3 on a 4-node (16x GB200) Blackwell setup, achieving nearly 3000 tok/s on balanced workloads.
- **GLM-5.2 Validation**: Successfully benchmarked GLM-5.2 on a 2-node G4 setup and validated the stack on GB300 (A4X Max) infrastructure.
- **Native FP4 Support for Kimi K2.6**: Successfully optimized and benchmarked Kimi-K2.6 using native NVFP4 quantization on a 2-node (16x GPU) setup, achieving over 3000 tok/s output throughput.
- **Qwen3.5-397B Validation**: Successfully benchmarked the 397B MoE model on a single node using FP8 and HiCache, showing massive TTFT improvements.
- **Agentic Benchmarking**: Introduced agentic trace simulation for Kimi K2.6, achieving over 80% cache hit rate with HiCache.
- **Kimi-K2.5 NVFP4 Validation**: Successfully optimized and benchmarked Kimi-K2.5 using native NVFP4 quantization on a 2-node (16x GPU) setup.
- **Native FP4 Support**: Successfully validated DeepSeek-V3.2 and GLM-5.1 on single-node setups using NVFP4 quantization.

## GKE Infrastructure Setup

The `gkecluster` directory contains a comprehensive template for provisioning a GKE environment optimized for SGLang:
- **Custom VPC**: High MTU (8896) for optimized multi-node traffic.
- **Multi-Networking**: Specialized network interfaces for distributed inference.
- **Blackwell Node Pools**: Automated creation of `g4-standard-384` pools with 8x RTX PRO 6000 Blackwell GPUs.
- **Benchmarking Isolation**: Dedicated node pools for load generators to ensure clean performance metrics.

## Viewing Detailed Benchmark Results

Detailed performance logs, including TTFT/TPOT latency distributions and throughput metrics, are located within each model's `results` directory:

- [moonshotai/Kimi-K3: models/kimik3/results/benchamrk_sweep_report.md](./models/kimik3/results/benchamrk_sweep_report.md)
- [zai-org/GLM-5.2-FP8: models/GLM5.2/fp8/results/benchmark_results.yaml](./models/GLM5.2/fp8/results/benchmark_results.yaml)
- [Qwen/Qwen3.5-397B-A17B-FP8: models/Qwen3.5-397B-A17B-FP8/BENCHMARK_REPORT.md](./models/Qwen3.5-397B-A17B-FP8/BENCHMARK_REPORT.md)
- [moonshotai/Kimi-K2.6 Agentic: models/KimiK2.6/agent_benchmark/README.md](./models/KimiK2.6/agent_benchmark/README.md)
- [deepseek-ai/DeepSeek-V3.2 (FP8): models/DeepSeekv3-2/fp8/results/benchmark_results.md](./models/DeepSeekv3-2/fp8/results/benchmark_results.md)
- [nvidia/DeepSeek-V3.2-NVFP4 (NVFP4): models/DeepSeekv3-2/nvp4/results/benchmark_results.md](./models/DeepSeekv3-2/nvp4/results/benchmark_results.md)
- [zai-org/GLM-5.1-FP8 (FP8): models/GLM5.1/results/benchmark-results.md](./models/GLM5.1/results/benchmark-results.md)
- [lukealonso/GLM-5.1-NVFP4 (NVFP4): models/GLM5.1/nvfp4/README.md](./models/GLM5.1/nvfp4/README.md)
- [moonshotai/Kimi-K2.5 (INT4): models/KimiK2.5/results/benchmark_results.md](./models/KimiK2.5/results/benchmark_results.md)
- [nvidia/Kimi-K2.5-NVFP4 (NVFP4): models/KimiK2.5/nvfp4/results/benchmarks_2node.yaml](./models/KimiK2.5/nvfp4/results/benchmarks_2node.yaml)
- [moonshotai/Kimi-K2.6 (Standard): models/KimiK2.6/results/benchmark_results.md](./models/KimiK2.6/results/benchmark_results.md)
- [nvidia/Kimi-K2.6-NVFP4 (NVFP4): models/KimiK2.6/nvfp4/results/benchmark-results.md](./models/KimiK2.6/nvfp4/results/benchmark-results.md)
- [datalab-to/chandra-ocr-2: models/datalab2-ocr/benchmark_results.md](./models/datalab2-ocr/benchmark_results.md)
- [openai/whisper-large-v3: models/whisper-v3-large/results/benchmark_results.md](./models/whisper-v3-large/results/benchmark_results.md)

## Usage

For detailed instructions on deploying models and running benchmarks, see the [Benchmarking Guide](./benchmarking_guide.md).

Each model directory also contains a dedicated `README.md` with specific optimization details and attribution.

## Contributing

This repository is updated as new optimization techniques (e.g., native FP4 serving) and models are validated on the G4 architecture.
