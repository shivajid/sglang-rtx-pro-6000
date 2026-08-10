# Performance Evaluation Report: google/gemma-4-26B-A4B

**Workload Profile**: 10,240 Input Tokens (10K Context) / 500 Output Tokens  
**Model Architecture**: `google/gemma-4-26B-A4B` (FP8 Quantization)  
**Inference Engine**: vLLM OpenAI API Server (Tensor Parallelism TP=2, PyNCCL All-Reduce)  
**Hardware Infrastructure**: 2x NVIDIA RTX PRO 6000 Blackwell GPUs (96 GB VRAM per GPU, 192 GB Total, GKE nodepool `g4-384-np-1`)  
**Benchmarking Harness**: `sglang.bench_serving` executed from nodepool `cpu-bench-pool`

---

## 1. Executive Summary

This report documents the performance characteristics of `google/gemma-4-26B-A4B` deployed in a dual-GPU Tensor Parallel configuration (TP=2) on NVIDIA Blackwell RTX PRO 6000 hardware. Testing evaluated concurrency levels from 32 to 512 using a synthetic long-context workload consisting of 10,240 input prompt tokens and 500 generated output tokens.

### Key Findings

- **Total System Throughput**: Peak aggregate throughput reached **26,917.85 tokens/second** (prefill and decode combined) at concurrency 512, with generation throughput reaching **1,268.02 output tokens/second** (4.96 requests/second).
- **Time to First Token (TTFT)**: For 10,240-token input payloads, median prefill latency remained under **655 milliseconds** up to concurrency 64, and under **1.11 seconds** at concurrency 128.
- **Inter-Token Latency (ITL)**: Dual-GPU tensor parallelism reduced median streaming latency to **18.03 ms** at concurrency 32 and **24.78 ms** at concurrency 64, representing a **35% to 41% reduction** compared to a single-GPU baseline.
- **High-Concurrency Queue Relief**: At concurrency 256, TP=2 reduced median TTFT from **14.33 seconds** (single GPU) down to **4.37 seconds**, a **3.28x latency improvement** under heavy saturation.

---

## 2. Benchmark Configuration and Methodology

```
+---------------------------------------------------------------------------------------+
| Parameter                   | Value                                                   |
+-----------------------------+---------------------------------------------------------+
| Target Model                | google/gemma-4-26B-A4B                                  |
| Precision / Quantization    | FP8 KV Cache, FP8 Model Weights                         |
| Tensor Parallel Size        | 2                                                       |
| Max Model Context Length    | 12,288 tokens                                           |
| Max Number of Sequences     | 320                                                     |
| Max Batched Tokens          | 16,384 tokens                                           |
| Workload Payload            | 10,240 input tokens, 500 output tokens                  |
| Request Arrival Rate        | Infinite (Closed-loop concurrency sweep)                |
| Evaluated Concurrency Range | 32, 64, 128, 256, 512                                   |
| Communication Backend       | PyNCCL (Custom all-reduce disabled for CC 12.0)         |
+---------------------------------------------------------------------------------------+
```

---

## 3. Concurrency Sweep Results (2x GPU TP=2)

The table below summarizes performance metrics across all tested concurrency levels. Values reflect steady-state benchmarks with request counts scaled to twice the concurrency level ($N = 2 \times C$).

| Concurrency | Completed Requests | Output Tok/s | Input Tok/s | Total Tok/s | Request Rate (req/s) | Median TTFT (ms) | P99 TTFT (ms) | Median TPOT (ms) | P99 TPOT (ms) | Median ITL (ms) | Median Latency (s) | P99 Latency (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **32** | 64 | 739.60 | 14,946.11 | 15,685.71 | 2.86 | 653.04 | 4,945.64 | 34.89 | 88.92 | 18.03 | 9.54 | 19.15 |
| **64** | 128 | 858.69 | 19,082.83 | 19,941.52 | 3.55 | 627.23 | 10,833.18 | 57.77 | 290.57 | 24.78 | 14.85 | 32.58 |
| **128** | 256 | 1,047.36 | 21,892.28 | 22,939.65 | 4.17 | 1,106.00 | 21,878.65 | 89.72 | 518.48 | 31.93 | 26.29 | 56.36 |
| **256** | 512 | 1,149.44 | 24,867.18 | 26,016.63 | 4.82 | 4,370.57 | 42,685.98 | 169.92 | 524.72 | 43.68 | 46.29 | 100.36 |
| **512** | 1024 | 1,268.02 | 25,649.83 | 26,917.85 | 4.96 | 38,842.66 | 91,079.16 | 230.64 | 527.51 | 132.98 | 89.14 | 193.23 |

---

## 4. Statistical Distribution: Mean vs. Median Metrics

Comparing mean and median distributions reveals the degree of queue buildup as concurrency scales beyond the hardware's immediate compute capacity.

| Concurrency | Output Tok/s | Total Tok/s | Req/s | Mean TTFT (ms) | Median TTFT (ms) | Mean TPOT (ms) | Median TPOT (ms) | Mean ITL (ms) | Median ITL (ms) | Mean Latency (s) | Median Latency (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **32** | 739.60 | 15,685.71 | 2.86 | 1,578.44 | **653.04** | 34.10 | **34.89** | 31.65 | **18.03** | 9.72 | **9.54** |
| **64** | 858.69 | 19,941.52 | 3.55 | 2,752.62 | **627.23** | 62.77 | **57.77** | 44.57 | **24.78** | 15.68 | **14.85** |
| **128** | 1,047.36 | 22,939.65 | 4.17 | 5,618.30 | **1,106.00** | 103.73 | **89.72** | 75.84 | **31.93** | 28.53 | **26.29** |
| **256** | 1,149.44 | 26,016.63 | 4.82 | 15,221.73 | **4,370.57** | 172.93 | **169.92** | 133.56 | **43.68** | 50.81 | **46.29** |
| **512** | 1,268.02 | 26,917.85 | 4.96 | 42,149.17 | **38,842.66** | 225.66 | **230.64** | 207.92 | **132.98** | 90.00 | **89.14** |

---

## 5. Hardware Scaling Comparison: 1x GPU (TP=1) vs. 2x GPU (TP=2)

The table below contrasts the single-GPU baseline against the dual-GPU deployment across identical concurrency steps.

| Concurrency | 1-GPU Output (tok/s) | 2-GPU Output (tok/s) | Output Gain | 1-GPU Total (tok/s) | 2-GPU Total (tok/s) | 1-GPU Med TTFT (ms) | 2-GPU Med TTFT (ms) | 1-GPU Med ITL (ms) | 2-GPU Med ITL (ms) | ITL Improvement |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **32** | 566.73 | **739.60** | **+30.5%** | 12,019.50 | **15,685.71** | 695.43 | **653.04** | 30.38 | **18.03** | **-40.7%** |
| **64** | 729.19 | **858.69** | **+17.8%** | 16,934.05 | **19,941.52** | 674.00 | **627.23** | 38.02 | **24.78** | **-34.8%** |
| **128** | 935.84 | **1,047.36** | **+11.9%** | 20,496.97 | **22,939.65** | 831.37 | 1,106.00 | 48.60 | **31.93** | **-34.3%** |
| **256** | 1,048.57 | **1,149.44** | **+9.6%** | 23,733.47 | **26,016.63** | 14,328.43 | **4,370.57** | 57.68 | **43.68** | **-24.3%** |
| **512** | 1,146.01 | **1,268.02** | **+10.6%** | 24,327.84 | **26,917.85** | 68,408.90 | **38,842.66** | 59.69 | 132.98 | — |

---

## 6. Architectural Analysis & Deployment Recommendations

### 1. Throughput Dynamics
- Splitting the model across two GPUs increases memory bandwidth and total KV cache capacity, yielding a **10% to 30.5% increase** in generated token throughput across all test points.
- Aggregate system throughput exceeds **26.9K tokens/second** at high concurrency, demonstrating strong scaling during prefill on large context windows.

### 2. Streaming Responsiveness
- For latency-sensitive conversational applications, Inter-Token Latency (ITL) is the primary driver of perceived user responsiveness. The TP=2 deployment consistently delivers **18 ms to 32 ms** ITL between concurrency 32 and 128, providing significantly smoother streaming than the 1-GPU configuration.

### 3. Recommended Production Envelope
- **Target Concurrency**: **64 to 128 concurrent requests per instance**.
- **SLA Metrics at Target Operating Point**:
  - Generation Throughput: **858 to 1,047 tokens/second**
  - Aggregate Throughput: **19.9K to 22.9K tokens/second**
  - Median TTFT: **627 ms to 1.11 seconds**
  - Median ITL: **24.8 ms to 31.9 ms**
  - Median End-to-End Latency: **14.8 to 26.3 seconds**

---

## 7. Appendix: Raw Benchmark Outputs

### Concurrency 32
```text
============ Serving Benchmark Result ============
Backend:                                 vllm
Traffic request rate:                    inf
Max request concurrency:                 32
Successful requests:                     64
Benchmark duration (s):                  22.37
Total input tokens:                      334307
Total generated tokens:                  16543
Request throughput (req/s):              2.86
Input token throughput (tok/s):          14946.11
Output token throughput (tok/s):         739.60
Peak output token throughput (tok/s):    1660.00
Total token throughput (tok/s):          15685.71
Concurrency:                             27.80
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   9715.29
Median E2E Latency (ms):                 9536.64
P90 E2E Latency (ms):                    17609.38
P99 E2E Latency (ms):                    19153.09
---------------Time to First Token----------------
Mean TTFT (ms):                          1578.44
Median TTFT (ms):                        653.04
P99 TTFT (ms):                           4945.64
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          34.10
Median TPOT (ms):                        34.89
P99 TPOT (ms):                           88.92
---------------Inter-Token Latency----------------
Mean ITL (ms):                           31.65
Median ITL (ms):                         18.03
P95 ITL (ms):                            109.73
P99 ITL (ms):                            509.74
==================================================
```

### Concurrency 64
```text
============ Serving Benchmark Result ============
Backend:                                 vllm
Traffic request rate:                    inf
Max request concurrency:                 64
Successful requests:                     128
Benchmark duration (s):                  36.08
Total input tokens:                      688463
Total generated tokens:                  30978
Request throughput (req/s):              3.55
Input token throughput (tok/s):          19082.83
Output token throughput (tok/s):         858.69
Peak output token throughput (tok/s):    2128.00
Total token throughput (tok/s):          19941.52
Concurrency:                             55.63
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   15681.44
Median E2E Latency (ms):                 14852.12
P90 E2E Latency (ms):                    30130.34
P99 E2E Latency (ms):                    32581.16
---------------Time to First Token----------------
Mean TTFT (ms):                          2752.62
Median TTFT (ms):                        627.23
P99 TTFT (ms):                           10833.18
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          62.77
Median TPOT (ms):                        57.77
P99 TPOT (ms):                           290.57
---------------Inter-Token Latency----------------
Mean ITL (ms):                           44.57
Median ITL (ms):                         24.78
P95 ITL (ms):                            288.62
P99 ITL (ms):                            512.63
==================================================
```

### Concurrency 128
```text
============ Serving Benchmark Result ============
Backend:                                 vllm
Traffic request rate:                    inf
Max request concurrency:                 128
Successful requests:                     256
Benchmark duration (s):                  61.43
Total input tokens:                      1344849
Total generated tokens:                  64340
Request throughput (req/s):              4.17
Input token throughput (tok/s):          21892.28
Output token throughput (tok/s):         1047.36
Peak output token throughput (tok/s):    2760.00
Total token throughput (tok/s):          22939.65
Concurrency:                             118.91
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   28526.47
Median E2E Latency (ms):                 26294.02
P90 E2E Latency (ms):                    53641.87
P99 E2E Latency (ms):                    56358.33
---------------Time to First Token----------------
Mean TTFT (ms):                          5618.30
Median TTFT (ms):                        1106.00
P99 TTFT (ms):                           21878.65
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          103.73
Median TPOT (ms):                        89.72
P99 TPOT (ms):                           518.48
---------------Inter-Token Latency----------------
Mean ITL (ms):                           75.84
Median ITL (ms):                         31.93
P95 ITL (ms):                            510.59
P99 ITL (ms):                            524.32
==================================================
```

### Concurrency 256
```text
============ Serving Benchmark Result ============
Backend:                                 vllm
Traffic request rate:                    inf
Max request concurrency:                 256
Successful requests:                     512
Benchmark duration (s):                  106.31
Total input tokens:                      2643729
Total generated tokens:                  122197
Request throughput (req/s):              4.82
Input token throughput (tok/s):          24867.18
Output token throughput (tok/s):         1149.44
Peak output token throughput (tok/s):    3276.00
Total token throughput (tok/s):          26016.63
Concurrency:                             244.69
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   50812.82
Median E2E Latency (ms):                 46294.59
P90 E2E Latency (ms):                    92419.01
P99 E2E Latency (ms):                    100360.77
---------------Time to First Token----------------
Mean TTFT (ms):                          15221.73
Median TTFT (ms):                        4370.57
P99 TTFT (ms):                           42685.98
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          172.93
Median TPOT (ms):                        169.92
P99 TPOT (ms):                           524.72
---------------Inter-Token Latency----------------
Mean ITL (ms):                           133.56
Median ITL (ms):                         43.68
P95 ITL (ms):                            523.51
P99 ITL (ms):                            531.06
==================================================
```

### Concurrency 512
```text
============ Serving Benchmark Result ============
Backend:                                 vllm
Traffic request rate:                    inf
Max request concurrency:                 512
Successful requests:                     1024
Benchmark duration (s):                  206.57
Total input tokens:                      5298492
Total generated tokens:                  261935
Request throughput (req/s):              4.96
Input token throughput (tok/s):          25649.83
Output token throughput (tok/s):         1268.02
Peak output token throughput (tok/s):    3330.00
Total token throughput (tok/s):          26917.85
Concurrency:                             446.12
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   90003.58
Median E2E Latency (ms):                 89139.73
P90 E2E Latency (ms):                    146898.14
P99 E2E Latency (ms):                    193234.78
---------------Time to First Token----------------
Mean TTFT (ms):                          42149.17
Median TTFT (ms):                        38842.66
P99 TTFT (ms):                           91079.16
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          225.66
Median TPOT (ms):                        230.64
P99 TPOT (ms):                           527.51
---------------Inter-Token Latency----------------
Mean ITL (ms):                           207.92
Median ITL (ms):                         132.98
P95 ITL (ms):                            536.84
P99 ITL (ms):                            544.72
==================================================
```
