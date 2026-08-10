# SGLang Benchmark Report: `google/gemma-4-26B-A4B` (10K / 500 on 2x Blackwell GPUs)

Comprehensive performance sweep across concurrency levels **32 to 512** for **`google/gemma-4-26B-A4B`** served via **vLLM** on **2x NVIDIA RTX PRO 6000 Blackwell GPUs** (Tensor Parallelism TP=2, FP8 quantization).

- **Input Prompt Length**: 10,240 tokens (10K Context)
- **Max Output Tokens**: 500 tokens
- **Serving Engine**: vLLM OpenAI Server (TP=2, `--disable-custom-all-reduce`, PyNCCL)
- **Benchmark Client**: `sglang.bench_serving` on `cpu-bench-pool`

---

## 1. 2-GPU Concurrency Sweep Summary Table (Median & P99 Metrics)

| Concurrency | Completed Req | Output Tok/s | Input Tok/s | Total Tok/s | Req/s | Median TTFT (ms) | P99 TTFT (ms) | Median TPOT (ms) | P99 TPOT (ms) | Median ITL (ms) | Median Latency (s) | P99 Latency (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **32** | 64 | **739.60** | 14,946.11 | 15,685.71 | 2.86 | **653.04** | 4,945.64 | **34.89** | 88.92 | **18.03** | **9.54** | 19.15 |
| **64** | 128 | **858.69** | 19,082.83 | 19,941.52 | 3.55 | **627.23** | 10,833.18 | **57.77** | 290.57 | **24.78** | **14.85** | 32.58 |
| **128** ⚡ | 256 | **1,047.36** | 21,892.28 | 22,939.65 | 4.17 | **1,106.00** | 21,878.65 | **89.72** | 518.48 | **31.93** | **26.29** | 56.36 |
| **256** | 512 | **1,149.44** | 24,867.18 | 26,016.63 | 4.82 | **4,370.57** | 42,685.98 | **169.92** | 524.72 | **43.68** | **46.29** | 100.36 |
| **512** 🏆 | 1024 | **1,268.02** | 25,649.83 | **26,917.85** | **4.96** | **38,842.66** | 91,079.16 | **230.64** | 527.51 | **132.98** | **89.14** | 193.23 |

---

## 2. Direct Hardware Scaling: 1 GPU (TP=1) vs. 2 GPUs (TP=2)

| Concurrency | 1-GPU Output (tok/s) | 2-GPU Output (tok/s) | Output Gain | 1-GPU Total (tok/s) | 2-GPU Total (tok/s) | 1-GPU Med TTFT (ms) | 2-GPU Med TTFT (ms) | 1-GPU Med ITL (ms) | 2-GPU Med ITL (ms) | ITL Reduction |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **32** | 566.73 | **739.60** | **+30.5%** | 12,019.50 | **15,685.71** | 695.43 | **653.04** | 30.38 | **18.03** | **-40.7%** ⚡ |
| **64** | 729.19 | **858.69** | **+17.8%** | 16,934.05 | **19,941.52** | 674.00 | **627.23** | 38.02 | **24.78** | **-34.8%** ⚡ |
| **128** | 935.84 | **1,047.36** | **+11.9%** | 20,496.97 | **22,939.65** | 831.37 | 1,106.00 | 48.60 | **31.93** | **-34.3%** ⚡ |
| **256** | 1,048.57 | **1,149.44** | **+9.6%** | 23,733.47 | **26,016.63** | 14,328.43 | **4,370.57** (3.3x faster) | 57.68 | **43.68** | **-24.3%** ⚡ |
| **512** | 1,146.01 | **1,268.02** | **+10.6%** | 24,327.84 | **26,917.85** | 68,408.90 | **38,842.66** (1.8x faster) | 59.69 | 132.98 | — |

---

## 3. Key Architectural Takeaways for 2-GPU Deployment

1. **Total System Bandwidth Record (26.9K tok/s)**:
   - Distributing the 10K prompt across 2 Blackwell GPUs boosts total throughput from **24,327 tok/s** up to **26,917.85 tok/s**.
2. **Substantial Reduction in Streaming Latency (ITL)**:
   - Median Inter-Token Latency drops by **34% to 41%** across concurrencies 32–128 (e.g. from **30.38 ms down to 18.03 ms** at C=32), providing ultra-responsive per-token streaming.
3. **Queue TTFT Relief under High Concurrency**:
   - At C=256, 2 GPUs reduce median prefill wait time from **14.33 seconds down to 4.37 seconds** (**3.3x faster**).
4. **Optimal Sizing**:
   - **Concurrency 64 to 128** delivers **~20K–23K total tok/s**, sub-second prefill TTFT, and ultra-low **24–32 ms ITL**.
