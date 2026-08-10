# SGLang Benchmark Report: `google/gemma-4-26B-A4B` (10K Input / 500 Output)

Comprehensive performance sweep across concurrency levels **32 to 512** for **`google/gemma-4-26B-A4B`** served via **vLLM** on a single **NVIDIA RTX PRO 6000 Blackwell GPU** (`g4-standard-384`, TP=1, FP8 quantization).

- **Input Prompt Length**: 10,240 tokens (10K Context)
- **Max Output Tokens**: 500 tokens
- **Backend**: vLLM OpenAI API Server
- **Client**: `sglang.bench_serving` executed from `cpu-bench-pool`

---

## 1. Concurrency Sweep Summary Table (Median & P99 Metrics)

| Concurrency | Completed Req | Output Tok/s | Input Tok/s | Total Tok/s | Req/s | Median TTFT (ms) | P99 TTFT (ms) | Median TPOT (ms) | P99 TPOT (ms) | Median ITL (ms) | Median Latency (s) | P99 Latency (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **32** | 64 | **566.73** | 11,452.76 | 12,019.50 | 2.19 | **695.43** | 4,681.23 | **45.71** | 90.44 | **30.38** | **12.85** | 24.34 |
| **64** | 128 | **729.19** | 16,204.87 | 16,934.05 | 3.02 | **674.00** | 9,942.41 | **68.55** | 283.19 | **38.02** | **17.05** | 36.92 |
| **128** ⚡ | 256 | **935.84** | 19,561.13 | 20,496.97 | 3.73 | **831.37** | 20,741.02 | **98.68** | 498.66 | **48.60** | **28.20** | 60.81 |
| **256** | 512 | **1,048.57** | 22,684.89 | 23,733.47 | 4.40 | **14,328.43** | 45,964.08 | **147.97** | 503.67 | **57.68** | **49.56** | 101.40 |
| **512** 🏆 | 1024 | **1,146.01** | 23,181.83 | **24,327.84** | **4.48** | **68,408.90** | 99,151.98 | **158.90** | 353.98 | **59.69** | **95.95** | 169.58 |

---

## 2. Detailed Performance Comparison: Mean vs. Median

| Concurrency | Output Tok/s | Total Tok/s | Req/s | Mean TTFT (ms) | Median TTFT (ms) | Mean TPOT (ms) | Median TPOT (ms) | Mean ITL (ms) | Median ITL (ms) | Mean Latency (s) | Median Latency (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **32** | 566.73 | 12,019.50 | 2.19 | 1,490.30 | **695.43** | 44.48 | **45.71** | 34.20 | **30.38** | 13.97 | **12.85** |
| **64** | 729.19 | 16,934.05 | 3.02 | 2,767.08 | **674.00** | 79.44 | **68.55** | 49.33 | **38.02** | 18.96 | **17.05** |
| **128** | 935.84 | 20,496.97 | 3.73 | 5,554.32 | **831.37** | 117.98 | **98.68** | 68.64 | **48.60** | 30.64 | **28.20** |
| **256** | 1,048.57 | 23,733.47 | 4.40 | 18,296.84 | **14,328.43** | 162.59 | **147.97** | 108.38 | **57.68** | 56.40 | **49.56** |
| **512** | 1,146.01 | 24,327.84 | 4.48 | 59,022.76 | **68,408.90** | 154.88 | **158.90** | 148.32 | **59.69** | 96.68 | **95.95** |

---

## 3. Key Observations & Sizing Guidance

1. **Massive Total Throughput (24.3K tok/s)**:
   - Due to the large 10K prompt size, the GPU sustains **23,181.83 input tok/s** and **24,327.84 total tok/s** at Concurrency 512.
2. **Sub-Second TTFT up to Concurrency 128**:
   - Even with 10K prompt context, the Blackwell GPU processes prefill with **Median TTFT of ~674–831 ms** through Concurrency 128.
3. **Smooth Inter-Token Streaming**:
   - Median Inter-Token Latency (ITL) remains between **30.38 ms and 59.69 ms** across the entire sweep.
4. **Recommended Production Operating Range**:
   - **Concurrency 64 to 128** provides the optimal balance of **sub-second TTFT (<835 ms)**, fast turnaround (**17–28 s total latency**), and high throughput (**16.9K–20.5K total tok/s**).
