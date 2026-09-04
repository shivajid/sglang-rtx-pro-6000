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

*Rows are ordered by benchmark date, newest first.*

| Model | Benchmarked | Quantization | Setup | Output Throughput (tok/s) | Total Throughput (tok/s) | Peak Throughput (tok/s) | TPOT (ms) |
|-------|-------------|--------------|-------|---------------------------|--------------------------|-------------------------|-----------|
| [deepseek-ai/DeepSeek-V4-Flash-0731 (1-Node)](./models/DeepSeekV4-Flash-0731/README.md)# | 2026-09-04 | MXFP4 | 1 Node (8x RTX 6000) | 3880.89 | — | 6122.00 | 88.71 |
| [zai-org/GLM-5.3-Flash](./models/GLM5.3-Flash/README.md)§ | 2026-08-29 | FP8 (tuned SM120 MoE) | 1 Node (8x RTX 6000) | 2579.80 | 9481.70 | 9481.70 | 70.60 |
| [nvidia/GLM-5.2-NVFP4](./models/GLM5.2/nvfp4/resuts/benchmark_results.md)‡ | 2026-08-17 | NVFP4 | 1 Node (8x RTX 6000) | 1100.92 | 1238.53 | 1280.00 | 115.00 |
| [moonshotai/Kimi-K3 (G4)](./models/kimik3/g4/BENCHMARK_REPORT.md)† | 2026-08-16 | BF16 (FP8 KV) | 4 Nodes (32x RTX 6000) | 583.58 | 649.11 | 816.00 | 78.85 |
| [google/gemma-4-26B-A4B](./models/Gemma4-26B/results/master_benchmark_summary.md)**** | 2026-08-09 | FP8 | 1 Node (1x RTX 6000, vLLM) | 3623.30 | 4094.73 | 4054.82 | 61.75 |
| [deepseek-ai/DeepSeek-V4-Flash-0731](./models/DeepSeekV4-Flash-0731/results/benchamrk_sweep_report.md)*** | 2026-08-05 | MXFP4 | 2 Nodes (16x RTX 6000) | 1606.27 | — | 4710.94 | 107.64 |
| [moonshotai/Kimi-K3](./models/kimik3/results/benchamrk_sweep_report.md) | 2026-08-04 | BF16 | 4 Nodes (16x GB200) | 1666.54 | 1874.86 | — | 24.88 |
| [zai-org/GLM-5.2-FP8](./models/GLM5.2/fp8/results/benchmark_results.yaml) | 2026-07-08 | FP8 | 2 Nodes (16x RTX 6000) | 1645.21 | 1855.07 | 2608.00 | 240.43 |
| [nvidia/Kimi-K2.6-NVFP4](./models/KimiK2.6/nvfp4/results/benchmark-results.md) | 2026-06-11 | NVFP4 | 2 Nodes (16x RTX 6000) | 3261.28 | 3662.79 | 4725.00 | 138.54 |
| [moonshotai/Kimi-K2.5](./models/KimiK2.5/results/benchmark_results.md) | 2026-06-10 | INT4* | 2 Nodes (16x RTX 6000) | 3152.79 | 3537.39 | 4793.00 | 136.52 |
| [Qwen/Qwen3.5-397B-A17B-FP8](./models/Qwen3.5-397B-A17B-FP8/results/hicache/benchmark_results.md) | 2026-05-22 | FP8 | 1 Node (8x RTX 6000) | 390.65 | 8202.16 | 1120.00 | 100.59 |
| [moonshotai/Kimi-K2.6](./models/KimiK2.6/results/benchmark_results.md) (wip)| 2026-05-15 | INT4* | 1 Node (8x RTX 6000) (not optimized) | 1459.26 | 1637.28 | 850.00 | 82.43 |
| [nvidia/Kimi-K2.5-NVFP4](./models/KimiK2.5/nvfp4/results/benchmarks_2node.yaml) | 2026-05-05 | NVFP4 | 2 Nodes (16x RTX 6000) | 3237.46 | 3632.39 | 5535.00 | 137.89 |
| [datalab-to/chandra-ocr-2](./models/datalab2-ocr/benchmark_results.md)** | 2026-05-04 | BF16| 1 Node (1x RTX 6000)| 2600.67 | 5267.08 | 4603.00| 32.47 |
| [lukealonso/GLM-5.1-NVFP4](./models/GLM5.1/nvfp4/results/benchmark_results_2node.md) | 2026-05-01 | NVFP4 | 2 Nodes (16x RTX 6000) | 3075.85 | 3451.06 | 4606.00 | 141.36 |
| [lukealonso/GLM-5.1-NVFP4](./models/GLM5.1/nvfp4/results/benchmark_results_1node.md) | 2026-05-01 | NVFP4 | 1 Node (8x RTX 6000) | 1490.31 | 1672.11 | 734.00 | 73.82 |
| [zai-org/GLM-5.1-FP8](./models/GLM5.1/results/benchmark-results.md) | 2026-05-01 | FP8 | 2 Nodes (16x RTX 6000) | 2785.55 | 3125.35 | 4092.00 | 155.26 |
| [nvidia/DeepSeek-V3.2-NVFP4](./models/DeepSeekv3-2/nvp4/results/benchmark_results.md) | 2026-04-24 | NVFP4 | 1 Node (8x RTX 6000) | 2675.33 | 3012.42 | 2046.00 | 106.03 |
| [deepseek-ai/DeepSeek-V3.2](./models/DeepSeekv3-2/fp8/results/benchmark_results.md) | 2026-04-23 | FP8 | 2 Nodes (16x RTX 6000) | 2962.79 | 3324.21 | 4951.00 | 149.29 |

**[openai/whisper-large-v3](./models/whisper-v3-large/results/benchmark_results.md)** - Since this is ASR model, we did not apply the standard ISL/OSL of 1K/8K and concurrancy of 512.

*Table last updated: September 4, 2026*

*# DeepSeek-V4-Flash-0731 (1-Node) was benchmarked on a single 8× RTX PRO 6000 node using 500GB Hyperdisk Balanced persistent storage (`1k/8k` reasoning sweep, 64 → 512 concurrency). Output throughput reached 3,880.89 tok/s with 6,122.00 peak at C=512. See [DeepSeekV4-Flash-0731/README.md](./models/DeepSeekV4-Flash-0731/README.md).*

*§ GLM-5.3-Flash runs on a single node with FP8 plus custom-tuned SM120 Triton fused-MoE kernels (E=289/N=256) — it would not serve on RTX PRO 6000 without the SM120 tilelang `num_stages=1` and DSA-backend patches described in its README. Its row reports the `1k/8k` reasoning peak output (2,579.8 tok/s @ C=256) and the `8k/1k` peak total (9,481.7 tok/s @ C=128); TPOT is the `1k/8k` @ 256 median ITL. See the [GLM-5.3 sweep report](./models/GLM5.3-Flash/results/benchmark_sweep_results.md).*
 
*Benchmarks conducted using `inf` request rate and 512 max concurrency. Tests utilized a random dataset with 1024 input tokens and 8192 output tokens (1536 total prompts). The load generator was isolated on a dedicated CPU-only node pool to ensure zero interference with GPU performance.*

*\*Kimi-K2.5 and Kimi-K2.6 use native INT4 quantization and KV cache optimization to improve memory efficiency and inference speed.*

**\** datalab-to/chandra-ocr-2 is an VLM model. We have run an image benchmark different for the rest of the models **

*\*\*\* DeepSeek-V4-Flash-0731 numbers come from its concurrency sweep (512 prompts, `1k/8k` @ 512 concurrency) rather than the 1536-prompt standard run. Its Peak column is the balanced `1k/1k` @ 512 result (4,710.94 output tok/s) — see the [sweep report](./models/DeepSeekV4-Flash-0731/results/benchamrk_sweep_report.md).*

*‡ GLM-5.2-NVFP4 was benchmarked at **128** concurrency (384 prompts), not the standard 512 — 256 and 512 have not been run yet. At 128 it delivers **137.6 output tok/s per GPU** on a single node, ~34% better per GPU than the 2-node FP8 build, at less than half the TPOT. A second point exists at C=64 (461.27 output tok/s, 1.0 s median TTFT) — see the [GLM-5.2 README](./models/GLM5.2/README.md).*

*† Kimi-K3 on G4 saturates at **128** concurrency, not 512 — the 32-GPU cluster hits its compute ceiling well before the standard profile's concurrency, so its row is the `1k/8k` peak at C=112. The Peak column is the observed burst peak (816 tok/s). Compare it to the [GB200 row](#performance-benchmarks-latest) for the same model on different silicon, not to the 512-concurrency rows. See the [G4 report](./models/kimik3/g4/BENCHMARK_REPORT.md).*

*\*\*\*\* gemma-4-26B-A4B is served with **vLLM** (not SGLang) on a single GPU, and is benchmarked with `sglang.bench_serving` as the client. Its main-table row is the `1k/8k` @ 512 concurrency point of the sweep; the Peak column is the best observed output throughput across all patterns (`1k/1k` @ 1024). Full sweep: [master_benchmark_summary.md](./models/Gemma4-26B/results/master_benchmark_summary.md).*

## [zai-org/GLM-5.3-Flash](./models/GLM5.3-Flash/README.md) Performance Sweep
[Detailed Configuration & Results](./models/GLM5.3-Flash/)

GLM-5.3-Flash (DeepSeek-style MoE + DSA, FP8, R1-style CoT) served on a **single G4 node** (8x RTX PRO 6000, SM120, TP=8). This model would **not serve on RTX PRO 6000 out of the box** — it required SM120 kernel patches (tilelang `num_stages=1` for the 99 KB shared-memory cap, and forcing the DSA backends to tilelang for GLM-5.3's rope=0 / 256-dim nope KV layout) plus a **custom-tuned Triton fused-MoE kernel config** (E=289/N=256, fp8_w8a8) generated on-node. Full 12-run sweep across `1k/1k`, `1k/8k`, and `8k/1k`, concurrency 32 → 256.

### Benchmark Settings
- **Setup:** 1 Node (8x RTX PRO 6000), TP=8, `moe-runner-backend=triton` (tuned), DSA on tilelang, decode CUDA graphs on.
- **Image:** `glm53-flash:sm120-moe-tuned-v1` (SGLang main + GLM-5.3 support + SM120 patches + tuned MoE config).
- **Workload Patterns:** `1k/8k` (reasoning), `8k/1k` (prefill), `1k/1k` (balanced) at concurrency 32 → 256.

| Workload Pattern | Peak Output Tok/s | Peak Total Tok/s | @ Concurrency | Median ITL |
| :--- | :---: | :---: | :---: | :---: |
| **8k / 1k (Prefill)** | 1,064.2 | **9,481.7** | 128 | 53.7 ms |
| **1k / 1k (Balanced)** | 1,913.1 | 3,870.0 | 256 | 69.4 ms |
| **1k / 8k (Reasoning)** | **2,579.8** | 2,898.7 | 256 | 70.6 ms |

**Operating guidance:** concurrency **64–128** is the sweet spot — sub-second median TTFT (204–695 ms), stable ITL (~30–54 ms), and near-peak saturation. Profiling shows the TP8 AllReduce over PCIe is the dominant GPU cost (~43%), with the tuned MoE at ~25%. See the [model README](./models/GLM5.3-Flash/README.md) for the SM120 patches, profiling breakdown, and the optimization roadmap (prefill CUDA graphs, MoE down-projection TMA tune, fp8 dense GEMM).

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

## [deepseek-ai/DeepSeek-V4-Flash-0731](./models/DeepSeekV4-Flash-0731/README.md) Performance Sweep
[Detailed Configuration & Results](./models/DeepSeekV4-Flash-0731/)

Full concurrency sweep (1 → 512) of DeepSeek's **V4-Flash** (Jul 31, 2026 checkpoint) on the standard 2-node G4 setup. Unlike the other trillion-class recipes, throughput **keeps scaling all the way to 512 concurrent streams** with no saturation plateau.

### Benchmark Settings
- **Setup:** 2 Nodes (16x RTX 6000), TP=8 · PP=2 · DP=8, DP attention.
- **Optimizations:** FlashInfer MXFP4 MoE runner (`--moe-runner-backend flashinfer_mxfp4`), FP8 KV cache.
- **Workload Patterns:** `1k/8k` (reasoning), `8k/1k` (prefill), `1k/1k` (balanced) at concurrency 1 → 512.

| Workload Pattern | Peak Output Throughput | @ Concurrency | TPOT @ 512 | TTFT Mean @ 512 |
| :--- | :---: | :---: | :---: | :---: |
| **1k / 1k (Balanced)** | **4,710.94 tok/s** | 512 | 106.50 ms | 1.23 s |
| **8k / 1k (Prompt)** | 4,209.22 tok/s | 512 | 113.38 ms | 7.19 s |
| **1k / 8k (Reasoning)** | 1,606.27 tok/s | 512 | 107.64 ms | 1.55 s |

TPOT moves only **75.3 → 106–113 ms/tok** from 1 → 512 streams; 100% success rate at every level. Charts: [results/charts/](./models/DeepSeekV4-Flash-0731/results/charts/). The unoptimized 1.6T [DeepSeek-V4-Pro config](./models/DeepSeekv4/) runs on the same topology.

### Single-Node `1k/8k` Reasoning Sweep (8× RTX PRO 6000, Hyperdisk Balanced)

Served on 1 node (8× RTX PRO 6000, `TP=8 · DP=8`, DP attention) with weights mounted via a 500 GB Hyperdisk Balanced volume to eliminate boot disk pressure. Benchmarked from an isolated CPU client pool (`shd-gem-cpu-pool`):

| Concurrency | Output tok/s | Peak tok/s | Mean TTFT | Mean TPOT | Avg Latency | Completed |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **64** | 1,440.93 | — | 1,663.9 ms | 33.98 ms | 148.99 s | 32 / 32 (100%) |
| **128** | 2,230.09 | — | 1,982.0 ms | 41.78 ms | 165.03 s | 48 / 48 (100%) |
| **256** | 2,676.52 | 4,474.00 | 2,892.7 ms | 82.89 ms | 324.25 s | 128 / 128 (100%) |
| **512** | **3,880.89** | **6,122.00** | 10,460.8 ms | 88.71 ms | 346.16 s | 256 / 256 (100%) |

## [moonshotai/Kimi-K3 on G4](./models/kimik3/g4/README.md) — 4 Nodes, 32x RTX PRO 6000
[Detailed Configuration & Results](./models/kimik3/g4/)

**Kimi-K3 now runs on G4.** The ~1.5 TB checkpoint (64 MoE + Linear Attention layers) served across 4× `g4-standard-384` nodes over plain VPC ethernet — no NVLink, no RDMA — using `PP=4 · TP=8`, Marlin MoE with an SM120 patch, Triton radix linear attention, FP8 KV cache, and HiCache host-RAM spillover. Weights load from Hyperdisk ML rather than per-node downloads.

### Benchmark Settings
- **Setup:** 4 Nodes (32x RTX PRO 6000), `PP=4 · TP=8`, 128K context.
- **Image:** `lmsysorg/sglang:nightly-dev-cu13-20260816-4a6dc267`.
- **Key flags:** `--moe-runner-backend marlin`, `--attention-backend triton`, `--triton-attention-num-kv-splits 16`, `--enable-hierarchical-cache --hicache-ratio 1.0`, `--kv-cache-dtype fp8_e4m3`.
- **Storage:** Hyperdisk ML (`ReadOnlyMany`, 2 TB) — weights mounted across all 4 nodes instead of downloaded per node.

### `1k/8k` Concurrency Sweep

| Concurrency | Output Tok/s | Total Tok/s | Median TTFT | P90 TTFT | Mean TPOT | Stream Speed | State |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **8** | 124.59 | 135.73 | 310.9 ms | 0.48 s | **51.96 ms** | **19.24 tok/s** | Zero queueing |
| **16** | 209.88 | 234.83 | 317.6 ms | 0.42 s | 62.55 ms | 15.99 tok/s | Zero queueing |
| **32** | 360.65 | 399.78 | 346.4 ms | 0.45 s | 71.49 ms | 13.99 tok/s | Zero queueing |
| **64** | **480.86** | 537.70 | **529.0 ms** | **0.69 s** | 74.00 ms | 13.51 tok/s | **Full admittance — sweet spot** |
| **80** | 563.51 | 627.32 | 606.3 ms | 64.05 s | 77.69 ms | 12.87 tok/s | Queueing begins |
| **96** | 545.29 | 604.45 | 1,368.5 ms | 221.71 s | 78.25 ms | 12.78 tok/s | Moderate queueing |
| **112** | **583.58** | **649.11** | 723.2 ms | 357.49 s | 78.85 ms | 12.68 tok/s | **Peak saturation** |
| **128** | 579.53 | 644.13 | 3,338.3 ms | 462.43 s | 78.51 ms | 12.74 tok/s | Multi-wave queueing |

**Concurrency 64 is the operating point** — 480.86 output tok/s with zero queueing, sub-530 ms median TTFT and P99 under 696 ms. Past C=80 the cluster holds ~64–72 active 8K-output streams and queues the rest, so P90 TTFT climbs into minutes while aggregate throughput gains only ~20%. Decode itself never degrades: TPOT stays flat at 77.7–78.8 ms/token all the way to 128.

### Workload Patterns (C=32)

| Pattern | Input / Output | Peak Output Tok/s | Total Tok/s @ C=32 | Median TTFT @ C=32 | Mean TPOT @ C=32 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1k / 1k (Balanced)** | 1000 / 1000 | 357.64 | 687.73 | **177.4 ms** | 75.37 ms |
| **1k / 8k (Reasoning)** | 1000 / 8000 | **583.58** | 399.78 | 346.4 ms | 71.49 ms |
| **1k / 500 (Short Chat)** | 1000 / 500 | 298.13 | 840.65 | 382.7 ms | 85.85 ms |
| **8k / 1k (Long Context)** | 8000 / 1000 | 251.04 | **2,194.60** | 1,732.6 ms | 117.86 ms |

Against the earlier G4 baseline: decode roughly **2× faster** (101.7 → 51.49 ms/tok single-stream, ~9.8 → 19.4 tok/s per user), peak output up **+160%**, and prefill 43% more responsive. All 1,104 benchmark requests completed — zero failures, zero CUDA OOMs, zero NCCL socket drops.

This deployment also pairs with the [Gemini CLI harness](./sglang_gemini_cli/README.md) for agentic coding on self-hosted Kimi-K3.

## [google/gemma-4-26B-A4B](./models/Gemma4-26B/results/master_benchmark_summary.md) Single-GPU Sweep (vLLM)
[Detailed Configuration & Results](./models/Gemma4-26B/)

The first **single-GPU** recipe in this repo: Gemma 4 26B (A4B MoE) served with **vLLM** on **one** RTX PRO 6000 Blackwell (`g4-standard-384`, TP=1, FP8), swept from concurrency 32 → 1024 across four workload patterns. Demonstrates that a trillion-class-quality chat model class can be served from a single GPU at >4K output tok/s.

### Benchmark Settings
- **Setup:** 1 Node, 1x RTX PRO 6000 (TP=1), FP8 quantization.
- **Backend:** vLLM OpenAI API server; client is `sglang.bench_serving` on the isolated `cpu-bench-pool`.
- **Workload Patterns:** `1k/512` (fast chat), `1k/1k` (balanced), `1k/8k` (reasoning), `8k/1k` (prompt-heavy).

| Workload Pattern | Peak Output Tok/s | Peak Total Tok/s | @ Concurrency | Median TPOT | Median TTFT @ C=256 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1k / 1k (Balanced)** | **4,054.82** | 8,034.02 | 1024 | 58.88 ms | **229 ms** |
| **1k / 512 (Fast Chat)** | 3,865.45 | **11,508.42** | 1024 | 61.41 ms | 414 ms |
| **1k / 8k (Reasoning)** | 3,830.64 | 4,333.42 | 1024 | 62.60 ms | 430 ms |
| **8k / 1k (Prompt Heavy)** | 2,162.20 | **18,537.77** | 1024 | 110.08 ms | 4,847 ms |

**Operating guidance:** concurrency **256** is the sweet spot — it sustains ~80–85% of peak throughput while keeping median TTFT sub-second and ITL between 28–64 ms. Beyond 512, prefill queueing pushes TTFT into the tens of seconds with no throughput gain, so cap admission per replica and scale horizontally.

### 10K Context Sweep (1 GPU vs. 2 GPU TP=2)
Separate `10k/500` long-context sweep — [1 GPU report](./models/Gemma4-26B/results/sglang_benchmark_10k_500_report.md) · [2 GPU report](./models/Gemma4-26B/results/10k_500_2gpu/sglang_benchmark_10k_500_report.md).

| Concurrency | 1 GPU Output Tok/s | 1 GPU Total Tok/s | 2 GPU Output Tok/s | 2 GPU Total Tok/s | 1 GPU Median TTFT | 2 GPU Median TTFT |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **64** | 729.19 | 16,934.05 | 858.69 | 19,941.52 | 674 ms | 627 ms |
| **128** | 935.84 | 20,496.97 | 1,047.36 | 22,939.65 | 831 ms | 1,106 ms |
| **256** | 1,048.57 | 23,733.47 | 1,149.44 | 26,016.63 | 14.3 s | 4.4 s |
| **512** | **1,146.01** | **24,327.84** | **1,268.02** | **26,917.85** | 68.4 s | 38.8 s |

Even at 10K prompt context, a single GPU holds median TTFT under **835 ms** through concurrency 128 while sustaining ~20.5K total tok/s. TP=2 buys roughly **+10% output throughput** and materially better tail latency, but not linear scaling — single-GPU replicas remain the more cost-efficient unit for this model.

## [moonshotai/Kimi-K3](./models/kimik3/vibench_hard_results/VIBENCH_HARD_KIMIK3_REPORT.md) ViBench Hard (Agentic Coding)
[Detailed Configuration & Results](./models/kimik3/vibench_hard_results/)

Brownfield feature-extension suite run against Kimi-K3 on an **8-node / 32x GB200** SGLang deployment (MNNVL + DCP enabled), driven by 4-way parallel OpenHands agents with Playwright headless-browser assertions. This is the *hard* tier — meaningfully tougher than the 24-app ViBench run that scored 95.6/100.

| Metric | Value |
| :--- | :---: |
| **Overall Normalized Score** | **79.2 / 100** |
| **Zero-Score Artifacts** | **0 (0.0%)** |
| **Perfect-Pass Artifacts** | 9 of 20 (45.0%) |
| **Total Test Plans Evaluated** | 63 |
| **Average Feature Build Time** | 44.7 min |
| **Inference Cost** | $0.00 (self-hosted GKE) |

Scores *rise* with task complexity — Feature 1 (schema additions) averaged 64.2/100 while Feature 3 (full-stack integrations) hit 84.8/100 and Feature 4 (brownfield refactoring) 96.8/100. Every artifact produced working code. The one systematic weak spot was `slack` (33.8/100), where real-time multi-context synchronization produced duplicate DOM state.

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
  - [`DeepSeekV4-Flash-0731/`](./models/DeepSeekV4-Flash-0731/README.md): 1-node and 2-node configs, Hyperdisk Balanced weight disk setup, and 1k/8k reasoning sweep (64–512 concurrency) reaching 3,881 tok/s.
  - `DeepSeekv4/`: DeepSeek-V4-Pro (1.6T) 2-node config — not yet optimized.
  - `GLM5.1/`: Optimized configurations and results for GLM-5.1.
  - [`GLM5.2/`](./models/GLM5.2/README.md): NVFP4 single-node and FP8 two-node recipes with benchmarks, plus the Blackwell (GB300) setup guide.
  - [`GLM5.3/`](./models/GLM5.3/README.md): 2-node FP8 recipe with tuned SM120 MoE kernels, GSM8K validation (0.900/0.925), and Hyperdisk ML provisioning.
  - [`GLM5.3-Flash/`](./models/GLM5.3-Flash/README.md): FP8 single-node (TP8) recipe with tuned SM120 MoE kernels and a 12-run concurrency sweep.
  - `Gemma4-26B/`: Single-GPU (and TP=2) vLLM configs and concurrency sweeps for Gemma 4 26B.
  - `kimik3/`: Performance sweeps and ViBench/ViBench-Hard agentic results for Kimi-K3, on both GB200 and G4.
    - `kimik3/g4/`: 4-node G4 recipe (`PP=4 · TP=8`, Marlin MoE, HiCache), benchmark report, and the Hyperdisk ML weight-provisioning guide.
  - `KimiK2.5/`: Configurations for Kimi-K2.5.
  - `KimiK2.6/`: Agentic benchmark results and HiCache configurations.
  - `Qwen3.5-397B-A17B-FP8/`: Latency benchmarks for ultra-large MoE model.
- `gkecluster/`: Infrastructure-as-Code for GKE provisioning.
- `benchmarking_scripts/`: Global benchmark definitions and performance scripts.
  - `agentic_benchmark/`: Scripts for simulating agentic workloads.
- [`sglang_gemini_cli/`](./sglang_gemini_cli/README.md): Point Gemini CLI at an SGLang endpoint to get an agentic coding harness on self-hosted models.
- `gcp_g4_specs.md`: Detailed hardware and infrastructure specifications.

## Key Updates (July–September 2026)
- **[DeepSeek-V4-Flash Single-Node & Hyperdisk Setup](./models/DeepSeekV4-Flash-0731/README.md)**: Built single-node (8× RTX PRO 6000) serving with dedicated 500 GB Hyperdisk Balanced volume (`dsv4-flash-hyperdisk-balanced`), completely eliminating boot disk pressure and eviction. Completed 1K/8K reasoning sweep from C=64 to 512, reaching **3,880.89 output tok/s** (peak 6,122.00 tok/s).
- **[GLM-5.3 2-Node Validation & MoE Tuning](./models/GLM5.3/README.md)**: Deployed 2-node FP8 recipe with tuned SM120 MoE config (`E=256,N=256,K=6144,fp8_w8a8`) and Hyperdisk ML provisioning. Correctness-verified with GSM8K (0.900/0.925 accuracy). Identified that `flashinfer_sparse_mla` suffers silent corruption and established `--dsa-prefill-backend trtllm --dsa-decode-backend trtllm` as the stable path.
- **[GLM-5.3-Flash on one node](./models/GLM5.3-Flash/README.md)**: Got GLM-5.3-Flash serving on a single G4 node (8x RTX PRO 6000, TP8) — a model that would not run on SM120 out of the box. Required tilelang `num_stages=1` (99 KB smem cap) + DSA-backend patches and a custom-tuned SM120 fused-MoE kernel config (E=289/N=256). Peak **9,481.7 total tok/s** (`8k/1k` @ C=128) and **2,579.8 output tok/s** (`1k/8k` @ C=256); profiling shows TP8 AllReduce over PCIe is the dominant cost (~43%).
- **Kimi-K3 on G4**: Got the ~1.5 TB Kimi-K3 checkpoint serving across 4 G4 nodes (32x RTX PRO 6000) over plain VPC ethernet with `PP=4 · TP=8`, Marlin MoE + SM120 patch, and HiCache host-RAM spillover — 583.58 peak output tok/s, decode 2× faster than the earlier baseline, with weights served from Hyperdisk ML.
- **Gemma 4 26B Single-GPU Recipe**: Added a vLLM-based single-GPU (and TP=2) recipe for `gemma-4-26B-A4B` on G4, with a full 32 → 1024 concurrency sweep across four workload patterns — 4,055 output tok/s peak and sub-250 ms median TTFT through concurrency 256, plus a dedicated 10K-context sweep.
- **Kimi-K3 ViBench Hard**: Ran the harder brownfield feature-extension tier on the 8-node (32x GB200) deployment — 79.2/100 normalized with zero failed artifacts; scores climb with task complexity (96.8/100 on Feature 4 refactors).
- **[Gemini CLI Harness](./sglang_gemini_cli/README.md)**: Full setup guide for connecting Gemini CLI to a self-hosted SGLang endpoint (validated with Kimi-K3 and DeepSeek-V4-Flash-0731) — turns a served checkpoint into an agent that edits files and runs shell commands, with no external API and no per-token cost.
- **[GLM-5.2 NVFP4 on one node](./models/GLM5.2/README.md)**: Benchmarked the single-node NVFP4 build — 1,100.92 output tok/s at 128 concurrency, **137.6 tok/s per GPU** (~34% better per GPU than the 2-node FP8 recipe) at less than half the TPOT. Requires forcing the `trtllm` DSA backends: the auto-selected `flashinfer_sparse_mla` silently emits NaN logits on SM120 with an FP8 KV cache.
- **DeepSeek-V4-Flash Sweep**: Benchmarked the Jul 31 checkpoint on 2 nodes with the FlashInfer MXFP4 MoE runner — 4,711 output tok/s at 512 concurrency with no saturation plateau; added an initial (unoptimized) config for the 1.6T DeepSeek-V4-Pro.
- **Kimi-K3 ViBench**: 24-app agentic coding benchmark on the 8-node deployment — 95.6/100 average with the regular reasoning trace (81.5 with `reasoning_effort: low`).
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

- [zai-org/GLM-5.3-Flash: models/GLM5.3-Flash/results/benchmark_sweep_results.md](./models/GLM5.3-Flash/results/benchmark_sweep_results.md)
- [deepseek-ai/DeepSeek-V4-Flash-0731: models/DeepSeekV4-Flash-0731/results/benchamrk_sweep_report.md](./models/DeepSeekV4-Flash-0731/results/benchamrk_sweep_report.md)
- [moonshotai/Kimi-K3 on G4: models/kimik3/g4/BENCHMARK_REPORT.md](./models/kimik3/g4/BENCHMARK_REPORT.md)
- [moonshotai/Kimi-K3 (GB200): models/kimik3/results/benchamrk_sweep_report.md](./models/kimik3/results/benchamrk_sweep_report.md)
- [moonshotai/Kimi-K3 ViBench Hard: models/kimik3/vibench_hard_results/VIBENCH_HARD_KIMIK3_REPORT.md](./models/kimik3/vibench_hard_results/VIBENCH_HARD_KIMIK3_REPORT.md)
- [google/gemma-4-26B-A4B: models/Gemma4-26B/results/master_benchmark_summary.md](./models/Gemma4-26B/results/master_benchmark_summary.md)
- [nvidia/GLM-5.2-NVFP4 (1 node): models/GLM5.2/nvfp4/resuts/benchmark_results.md](./models/GLM5.2/nvfp4/resuts/benchmark_results.md)
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
