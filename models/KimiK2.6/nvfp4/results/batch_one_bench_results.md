# SGLang bench_one_batch_server Results (Kimi-K2.6 NVFP4)

* **Setup**: 2 Nodes (16x RTX 6000)
* **Quantization**: FP4 (`modelopt_fp4`)
* **Tokenizer**: In-process (zero tokenizer worker processes)

| Batch Size | Prefill Length | Decode Length | Latency (s) | Input Throughput (tok/s) | Output Throughput (tok/s) | Overall Throughput (tok/s) | Slowest TTFT (s) | Avg Gen Speed per Rank (tok/s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **512** | 1024 | 8192 | 1138.99 s | 14,015.66 | 3,807.51 | 4,142.77 | 37.41 s | 487.57 |
| **64** | 1024 | 8192 | 530.94 s | 8,454.34 | 1,002.10 | 1,110.90 | 7.75 s | 125.52 |
