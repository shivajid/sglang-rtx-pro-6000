# SGLang Benchmark Report: `google/gemma-4-26B-A4B` (10K Input / 500 Output)

Performance sweep across concurrency levels **32 to 512** for **google/gemma-4-26B-A4B** served on single-node single-GPU (Blackwell G4).

- **Input Length**: 10,240 tokens (10K)
- **Max Output Length**: 500 tokens
- **Backend**: vLLM OpenAI API Server
- **Client**: `sglang.bench_serving` on `cpu-bench-pool`

---

## 📊 Concurrency Sweep Summary (Median & P99 Metrics)

| Concurrency | Completed Req | Output Tok/s | Input Tok/s | Total Tok/s | Req/s | Median TTFT (ms) | P99 TTFT (ms) | Median TPOT (ms) | P99 TPOT (ms) | Median ITL (ms) | Median Latency (s) | P99 Latency (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **32** | 64 | **566.73** | 11452.76 | 12019.50 | 2.19 | **695.43** | 4681.23 | **45.71** | 90.44 | **30.38** | **12.85** | 24.34 |
| **64** | 128 | **729.19** | 16204.87 | 16934.05 | 3.02 | **674.00** | 9942.41 | **68.55** | 283.19 | **38.02** | **17.05** | 36.92 |
| **128** | 256 | **935.84** | 19561.13 | 20496.97 | 3.73 | **831.37** | 20741.02 | **98.68** | 498.66 | **48.60** | **28.20** | 60.81 |
| **256** | 512 | **1048.57** | 22684.89 | 23733.47 | 4.40 | **14328.43** | 45964.08 | **147.97** | 503.67 | **57.68** | **49.56** | 101.40 |
| **512** | 1024 | **1146.01** | 23181.83 | 24327.84 | 4.48 | **68408.90** | 99151.98 | **158.90** | 353.98 | **59.69** | **95.95** | 169.58 |

---
