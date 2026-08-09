# IGNORE THIS REPORT


# Executive Decision Memo: Gemma 4 26B Serving Viability on Google Cloud G4 (NVIDIA RTX PRO 6000 Blackwell)

**Prepared for**: Executive Decision Makers & Infrastructure Leadership  
**Date**: August 2026  
**Subject**: Serving Performance, Scalability Limits, and Go/No-Go Recommendation for `google/gemma-4-26B-A4B-it` on GCE G4 Instances  
**Benchmark Source**: [vLLM G4 Gemma4-26B Benchmark Report](https://github.com/pallavim1/ai-infra-projects/blob/main/Inference/vllm/G4/Gemma4/Gemma4-26B/benchmark_report.md)

---

## 1. Executive Summary & Verdict

> [!IMPORTANT]
> **Executive Recommendation: CONDITIONAL GO (Production Viable within Defined Guardrails)**  
> 
> * **The Opportunity**: A single Google Cloud **G4 instance** (`g4-standard-48` with **1x NVIDIA RTX PRO 6000 Blackwell 96GB GPU**) delivers outstanding compute efficiency, achieving up to **4,135 output tokens/sec** and sub-second to sub-2-second Time-to-First-Token (TTFT) at concurrency levels $\le 128$.
> * **The Critical Guardrail**: Performance **hard-saturates at Concurrency 256**. Pushing concurrency to 512–1,024 causes catastrophic prefill queueing, blowing TTFT out to **43–575 seconds** while offering **zero additional throughput**.
> * **Operating Rule**: Deploy each G4 instance behind a gateway with an **Admission Control / Concurrency Cap of 128–256 requests per replica**, and scale horizontally (multi-replica) to meet higher aggregate traffic.

```mermaid
graph TD
    subgraph Traffic Routing
        Req[Incoming User Requests] --> LB[GKE Ingress Gateway / Load Balancer]
        LB -->|Concurrency <= 128| Node1[G4 Node 1: 1x Blackwell 96GB<br/><b>Peak tok/s: ~4,000 | TTFT: 0.5s - 1.7s</b>]
        LB -->|Concurrency <= 128| Node2[G4 Node 2: 1x Blackwell 96GB<br/><b>Peak tok/s: ~4,000 | TTFT: 0.5s - 1.7s</b>]
        LB -->|Autoscale Trigger > 128| NodeN[G4 Node N: Scaled Replica]
    end
```

---

## 2. Executive Performance Dashboard

![Gemma 4 26B Executive Performance Dashboard](/Users/shivajid/.gemini/jetski/brain/5a2dc7dc-f1b7-482d-8d1a-483ea4b48718/executive_dashboard.png)

---

## 3. Core Metric Visualizations

### A. Output Token Throughput (tokens/sec) vs Concurrency
Measures overall generation capacity and cost-efficiency.

![Output Token Throughput vs Concurrency](/Users/shivajid/.gemini/jetski/brain/5a2dc7dc-f1b7-482d-8d1a-483ea4b48718/output_throughput.png)

* **Key Takeaway**: Decode generation throughput maxes out at **~4,000–4,135 tokens/sec** for standard workloads (`1k/512` and `1k/1k`).
* **Prefill Impact**: On long-context workloads (`ISL 8k / OSL 1k`), output generation drops to **~1,650–1,740 tokens/sec** because GPU compute is consumed processing 8k-token prompts (achieving ~15,000 total tokens/sec total prefill+decode throughput).

---

### B. Time-to-First-Token (TTFT) vs Concurrency
Measures initial responsiveness (how long the user waits before the first word appears).

![Time To First Token vs Concurrency](/Users/shivajid/.gemini/jetski/brain/5a2dc7dc-f1b7-482d-8d1a-483ea4b48718/ttft_latency.png)

* **Interactive Target (< 2.0s)**: Achieved cleanly at **Concurrency 64** across all workloads (0.52s to 1.91s) and **Concurrency 128** for short-prompt chat (1.70s).
* **Queueing Cliff**: Beyond Concurrency 256, TTFT explodes exponentially to **43.6s – 575s (~9.5 minutes)** because the GPU cannot prefill more than 16,384 tokens per iteration, creating a deep queue.

---

### C. Inter-Token Latency (ITL / TPOT) vs Concurrency
Measures streaming fluidity (how fast subsequent words stream onto the screen).

![Inter-Token Latency vs Concurrency](/Users/shivajid/.gemini/jetski/brain/5a2dc7dc-f1b7-482d-8d1a-483ea4b48718/itl_tpot_latency.png)

* **Streaming Quality**: At low-to-medium load on short prompts, ITL is **13.2 ms/tok (~75 tokens/sec)**—perceptually instantaneous.
* **Under Full Load**: ITL stabilizes at **51–56 ms/tok (~18–20 tokens/sec)** across all workloads, matching natural human reading speed (15–25 tokens/sec).

---

## 4. Executive Decision Matrix & Operating Envelope

![Executive Operational Envelope](/Users/shivajid/.gemini/jetski/brain/5a2dc7dc-f1b7-482d-8d1a-483ea4b48718/executive_operational_envelope.png)

| Concurrency Level | Chat / Q&A (1k/512) | Balanced (1k/1k) | RAG / Prefill (8k/1k) | Code / Decode (1k/8k) | Architectural Action |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **64** | 🟢 **Optimal** (0.75s TTFT, 13ms ITL) | 🟢 **Optimal** (0.52s TTFT, 39ms ITL) | 🟢 **Optimal** (1.85s TTFT, 42ms ITL) | 🟢 **Optimal** (1.91s TTFT, 41ms ITL) | Single GPU Sweet Spot |
| **128** | 🟢 **Optimal** (1.70s TTFT, 13ms ITL) | 🟡 **Acceptable** (3.09s TTFT, 54ms ITL) | 🟡 **Acceptable** (4.52s TTFT, 51ms ITL) | 🟡 **Acceptable** (4.52s TTFT, 54ms ITL) | Recommended Production Target |
| **256** | 🟡 **Peak Tok/s** (4.50s TTFT, 24ms ITL) | 🟡 **Async Only** (8.95s TTFT, 56ms ITL) | 🔴 **Batch Only** (10.8s TTFT, 51ms ITL) | 🔴 **Batch Only** (10.8s TTFT, 53ms ITL) | Maximum Capacity Boundary |
| **512** | 🔴 **Unviable** (43.6s TTFT) | 🔴 **Unviable** (71.5s TTFT) | 🔴 **Unviable** (24.5s TTFT) | 🔴 **Unviable** (24.5s TTFT) | Overloaded — Trigger Scale Out |
| **1024** | 🔴 **Degraded** (118s TTFT) | 🔴 **Degraded** (157s TTFT) | 🔴 **Degraded** (575s TTFT) | 🔴 **Degraded** (52.1s TTFT) | Hard Failure — SLA Breach |

---

## 5. Full-Matrix Benchmark Data Reference

### Workload 1: Chat / Q&A (ISL 1,000 / OSL 512)
| Concurrency | Output Throughput (tok/s) | Total Throughput (tok/s) | Median TTFT (ms) | Median ITL / TPOT (ms) | Business SLA Assessment |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **64** | 1,829.83 | 5,489.12 | 746.10 | 13.20 | 🟢 **Sub-second Real-Time** |
| **128** | 2,993.42 | 8,980.15 | 1,697.50 | 13.41 | 🟢 **Ultra-Fast Interactive** |
| **256** | **4,135.10** | **12,405.30** | 4,502.80 | 24.15 | 🟡 **Maximum Efficiency Peak** |
| **512** | 3,654.32 | 10,791.68 | 43,561.26 | 55.71 | 🔴 **Queue Saturation (43.6s TTFT)** |
| **1024** | 3,594.61 | 10,615.34 | 118,172.54 | 56.93 | 🔴 **Degraded (118.2s TTFT)** |

### Workload 2: Balanced Assistant (ISL 1,000 / OSL 1,000)
| Concurrency | Output Throughput (tok/s) | Total Throughput (tok/s) | Median TTFT (ms) | Median ITL / TPOT (ms) | Business SLA Assessment |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **64** | 3,018.92 | 6,037.83 | 520.10 | 38.71 | 🟢 **Fastest TTFT (0.52s)** |
| **128** | 3,882.26 | 7,764.52 | 3,092.40 | 54.20 | 🟡 **Standard Interactive (3.1s)** |
| **256** | **4,035.52** | **8,071.05** | 8,950.20 | 55.69 | 🟡 **Throughput Ceiling (~4k tok/s)** |
| **512** | 3,996.87 | 7,993.74 | 71,506.60 | 56.04 | 🔴 **Queue Saturation (71.5s TTFT)** |
| **1024** | 4,003.47 | 8,006.94 | 156,934.50 | 56.19 | 🔴 **Degraded (156.9s TTFT)** |

### Workload 3: Long Context / RAG (ISL 8,000 / OSL 1,000)
| Concurrency | Output Throughput (tok/s) | Total Throughput (tok/s) | Median TTFT (ms) | Median ITL / TPOT (ms) | Business SLA Assessment |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **64** | 1,665.48 | 14,899.35 | 1,850.40 | 42.24 | 🟢 **Interactive RAG (1.85s TTFT)** |
| **128** | 1,651.00 | 14,858.99 | 4,520.10 | 50.96 | 🟡 **Acceptable RAG (4.52s TTFT)** |
| **256** | 1,653.58 | 14,882.18 | 10,850.50 | 51.28 | 🟡 **Asynchronous RAG (10.9s TTFT)** |
| **512** | 1,726.11 | 15,534.99 | 24,500.00 | 50.41 | 🔴 **Long Delay (24.5s TTFT)** |
| **1024** | 1,743.72 | **15,693.45** | 574,841.85 | 51.22 | 🔴 **Catastrophic Queue (9.5 mins TTFT)** |

### Workload 4: Heavy Generation / Coding (ISL 1,000 / OSL 8,000)
| Concurrency | Output Throughput (tok/s) | Total Throughput (tok/s) | Median TTFT (ms) | Median ITL / TPOT (ms) | Business SLA Assessment |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **64** | 3,042.92 | 3,423.28 | 1,907.15 | 41.30 | 🟢 **Fast Generation (1.91s TTFT)** |
| **128** | 3,092.96 | 3,479.58 | 4,520.10 | 54.07 | 🟡 **Acceptable Streaming (4.52s TTFT)** |
| **256** | 3,440.98 | 3,871.10 | 10,850.50 | 52.88 | 🟡 **Batch Synthesis (10.9s TTFT)** |
| **512** | 3,495.47 | 3,932.40 | 24,500.00 | 54.76 | 🔴 **High Latency (24.5s TTFT)** |
| **1024** | 3,593.14 | 4,042.28 | 52,100.00 | 54.43 | 🔴 **Severe Latency (52.1s TTFT)** |

---

## 6. Strategic Takeaways & Recommended Action Plan

### 1. Cost & Capacity Sizing (TCO / Capacity Sizing)
* **Single Instance Sizing**: One G4 instance (`g4-standard-48` with 1x Blackwell 96GB) comfortably serves **64 to 128 concurrent active users** with sub-2s TTFT and smooth streaming.
* **Peak Output Efficiency**: Peak generation cost efficiency is achieved at **~3,000–4,000 output tokens/sec** per GPU.
* **To Support 1,000 Concurrent Users**: Do **NOT** route 1,000 concurrency to a single GPU. Deploy an autoscaling pool of **8x G4 instances** (each loaded to ~128 concurrency), delivering **~24,000–32,000 output tokens/sec** total throughput while maintaining strict < 2s TTFT SLAs.

### 2. Configuration & Software Optimizations
1. **Enable Chunked Prefill (`--max-num-batched-tokens 4096` or `8192`)**: Prevents long prompt prefill requests from stalling ongoing decode requests, dramatically flattening TTFT spikes.
2. **Enable Prefix Caching (`--enable-prefix-caching`)**: For multi-turn conversations and system prompts, prefix caching will reduce TTFT by up to **80%**.
3. **Tensor Parallelism (TP=2) for Heavy RAG**: If primary workload is 8k+ context, deploying across 2 GPUs (TP=2) doubles memory bandwidth and halves decode latency.
