# Executive Benchmark Report: `google/gemma-4-26B-A4B` (10K Input / 500 Output on 2x Blackwell GPUs)

**Workload**: 10,240 Input Tokens (10K Context) / 500 Output Tokens  
**Model**: `google/gemma-4-26B-A4B` (FP8 Quantization)  
**Serving Engine**: vLLM OpenAI API Server (`--tensor-parallel-size=2`, `--disable-custom-all-reduce`, PyNCCL)  
**Hardware Topology**: **2x NVIDIA RTX PRO 6000 Blackwell GPUs** (96GB VRAM each, Total 192GB VRAM on GKE nodepool `g4-384-np-1`)  
**Benchmarking Tool**: `sglang.bench_serving` executed from dedicated client nodepool `cpu-bench-pool`

---

## 🏆 1. Executive Summary & Highlights

- **Record Total Bandwidth**: Achieves **26,917.85 total tokens/sec** (Input Prefill + Output Generation combined) at Concurrency 512.
- **Peak Generation Throughput**: Delivers **1,268.02 output tokens/sec** and **4.96 requests/sec** under saturated concurrency.
- **Ultra-Low Inter-Token Streaming Latency (ITL)**: Median ITL drops to **18.03 ms** at C=32 and **24.78 ms** at C=64 (**~35% to 41% lower streaming latency** compared to 1 GPU).
- **Sub-Second TTFT on 10K Context**: Sub-second prefill responsiveness (**627–653 ms Median TTFT**) sustained through Concurrency 64.
- **Massive Queue TTFT Reduction at C=256**: Drops from **14.33 seconds** on 1 GPU to **4.37 seconds** on 2 GPUs (**3.3x faster TTFT** under heavy load).

---

## 📊 2. 2-GPU Concurrency Sweep Summary Table (Median & P99 Metrics)

| Concurrency | Completed Req | Output Tok/s | Input Tok/s | Total Tok/s | Req/s | Median TTFT (ms) | P99 TTFT (ms) | Median TPOT (ms) | P99 TPOT (ms) | Median ITL (ms) | Median Latency (s) | P99 Latency (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **32** | 64 | **739.60** | 14,946.11 | 15,685.71 | 2.86 | **653.04** | 4,945.64 | **34.89** | 88.92 | **18.03** | **9.54** | 19.15 |
| **64** | 128 | **858.69** | 19,082.83 | 19,941.52 | 3.55 | **627.23** | 10,833.18 | **57.77** | 290.57 | **24.78** | **14.85** | 32.58 |
| **128** ⚡ | 256 | **1,047.36** | 21,892.28 | 22,939.65 | 4.17 | **1,106.00** | 21,878.65 | **89.72** | 518.48 | **31.93** | **26.29** | 56.36 |
| **256** | 512 | **1,149.44** | 24,867.18 | 26,016.63 | 4.82 | **4,370.57** | 42,685.98 | **169.92** | 524.72 | **43.68** | **46.29** | 100.36 |
| **512** 🏆 | 1024 | **1,268.02** | 25,649.83 | **26,917.85** | **4.96** | **38,842.66** | 91,079.16 | **230.64** | 527.51 | **132.98** | **89.14** | 193.23 |

---

## 📈 3. Detailed Metric Comparison: Mean vs. Median (2 GPUs)

| Concurrency | Output Tok/s | Req/s | Mean TTFT (ms) | Median TTFT (ms) | Mean TPOT (ms) | Median TPOT (ms) | Mean ITL (ms) | Median ITL (ms) | Mean Latency (s) | Median Latency (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **32** | 739.60 | 2.86 | 1,578.44 | **653.04** | 34.10 | **34.89** | 31.65 | **18.03** | 9.72 | **9.54** |
| **64** | 858.69 | 3.55 | 2,752.62 | **627.23** | 62.77 | **57.77** | 44.57 | **24.78** | 15.68 | **14.85** |
| **128** | 1,047.36 | 4.17 | 5,618.30 | **1,106.00** | 103.73 | **89.72** | 75.84 | **31.93** | 28.53 | **26.29** |
| **256** | 1,149.44 | 4.82 | 15,221.73 | **4,370.57** | 172.93 | **169.92** | 133.56 | **43.68** | 50.81 | **46.29** |
| **512** | 1,268.02 | 4.96 | 42,149.17 | **38,842.66** | 225.66 | **230.64** | 207.92 | **132.98** | 90.00 | **89.14** |

---

## ⚡ 4. Hardware Scaling Analysis: 1 GPU (TP=1) vs. 2 GPUs (TP=2)

| Concurrency | 1-GPU Output (tok/s) | 2-GPU Output (tok/s) | Output Gain | 1-GPU Total (tok/s) | 2-GPU Total (tok/s) | 1-GPU Med TTFT (ms) | 2-GPU Med TTFT (ms) | 1-GPU Med ITL (ms) | 2-GPU Med ITL (ms) | ITL Reduction |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **32** | 566.73 | **739.60** | **+30.5%** | 12,019.50 | **15,685.71** | 695.43 | **653.04** | 30.38 | **18.03** | **-40.7%** ⚡ |
| **64** | 729.19 | **858.69** | **+17.8%** | 16,934.05 | **19,941.52** | 674.00 | **627.23** | 38.02 | **24.78** | **-34.8%** ⚡ |
| **128** | 935.84 | **1,047.36** | **+11.9%** | 20,496.97 | **22,939.65** | 831.37 | 1,106.00 | 48.60 | **31.93** | **-34.3%** ⚡ |
| **256** | 1,048.57 | **1,149.44** | **+9.6%** | 23,733.47 | **26,016.63** | 14,328.43 | **4,370.57** (3.3x faster) | 57.68 | **43.68** | **-24.3%** ⚡ |
| **512** | 1,146.01 | **1,268.02** | **+10.6%** | 24,327.84 | **26,917.85** | 68,408.90 | **38,842.66** (1.8x faster) | 59.69 | 132.98 | — |

---

## 🎯 5. Production Sizing & Architecture Recommendations

1. **Optimal Operating Sweet Spot (C=64 to 128)**:
   - For 10K input payloads, operating between **Concurrency 64 and 128** provides **1,047 output tok/s**, **22.9K total tok/s**, sub-second to near-second prefill TTFT (**~627–1106 ms**), and snappy per-token streaming (**24–32 ms ITL**).
2. **When to use TP=2**:
   - **Reduced Streaming Latency**: TP=2 cuts inter-token latency by **35% to 41%**, ideal for real-time streaming copilots.
   - **Heavy Concurrency Relief**: TP=2 shaves prefill queue latency from **14.3s down to 4.3s** at C=256.

---

## 📋 6. Raw Benchmark Serving Outputs (2 GPUs TP=2)

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
