# SGLang Benchmark Report: `google/gemma-4-26B-A4B` (10K Input / 500 Output - 2x GPU TP=2)

Performance sweep across concurrency levels **32 to 512** for **google/gemma-4-26B-A4B** served on **2x NVIDIA RTX PRO 6000 Blackwell GPUs (TP=2, FP8)**.

- **Input Length**: 10,240 tokens (10K)
- **Max Output Length**: 500 tokens
- **Backend**: vLLM OpenAI API Server (TP=2)
- **Client**: `sglang.bench_serving` on `cpu-bench-pool`

---

## 📊 Concurrency Sweep Summary (Median & P99 Metrics)

| Concurrency | Completed Req | Output Tok/s | Input Tok/s | Total Tok/s | Req/s | Median TTFT (ms) | P99 TTFT (ms) | Median TPOT (ms) | P99 TPOT (ms) | Median ITL (ms) | Median Latency (s) | P99 Latency (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **32** | 64 | **739.60** | 14946.11 | 15685.71 | 2.86 | **653.04** | 4945.64 | **34.89** | 88.92 | **18.03** | **9.54** | 19.15 |
| **64** | 128 | **858.69** | 19082.83 | 19941.52 | 3.55 | **627.23** | 10833.18 | **57.77** | 290.57 | **24.78** | **14.85** | 32.58 |
| **128** | 256 | **1047.36** | 21892.28 | 22939.65 | 4.17 | **1106.00** | 21878.65 | **89.72** | 518.48 | **31.93** | **26.29** | 56.36 |
| **256** | 512 | **1149.44** | 24867.18 | 26016.63 | 4.82 | **4370.57** | 42685.98 | **169.92** | 524.72 | **43.68** | **46.29** | 100.36 |
| **512** | 1024 | **1268.02** | 25649.83 | 26917.85 | 4.96 | **38842.66** | 91079.16 | **230.64** | 527.51 | **132.98** | **89.14** | 193.23 |

---
