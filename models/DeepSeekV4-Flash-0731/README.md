# DeepSeek-V4-Flash-0731

SGLang serving recipe and benchmarks for **`deepseek-ai/DeepSeek-V4-Flash-0731`** (released July 31, 2026) on **2× GCP G4 nodes** (`g4-standard-384`, 16× NVIDIA RTX PRO 6000 Blackwell, 96 GB each).

## Highlights

- **4,710.94 output tok/s** at concurrency 512 on the balanced `1k/1k` workload — with throughput **still climbing at 512 streams** (no saturation plateau, unlike the other trillion-class recipes in this repo).
- **TPOT barely degrades under load**: 75.3 ms/tok single-stream → 87.6–89.3 ms at 128 streams → 106.5–113.4 ms at 512 streams (~8.8–9.4 tok/s per user at full saturation).
- **Sub-second TTFT through concurrency 256** for 1K prompts (940.8 ms); only 1.23 s at 512.
- **100% success rate** (0 failed requests) at every concurrency level, 1 → 512, across all three workload patterns.

## Configuration

| Item | Value |
|------|-------|
| Model | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| Hardware | 2× `g4-standard-384` (16× RTX PRO 6000, 96 GB) |
| Parallelism | TP=8 · PP=2 · DP=8 with `--enable-dp-attention` |
| MoE runner | `flashinfer_mxfp4` |
| KV cache | `fp8_e4m3` |
| Image | `lmsysorg/sglang:dev-cu13` |
| Context length | 131,072 |

Key launch flags (full manifest: [`sglang-dsv4-flash-2node.yaml`](./sglang-dsv4-flash-2node.yaml)):

```bash
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V4-Flash-0731 --trust-remote-code \
  --tensor-parallel-size 8 --pipeline-parallel-size 2 --dp-size 8 --enable-dp-attention \
  --nnodes 2 --node-rank $POD_INDEX --dist-init-addr sglang-dsv4-flash-master:5000 \
  --moe-runner-backend flashinfer_mxfp4 --kv-cache-dtype fp8_e4m3 \
  --disable-custom-all-reduce \
  --mem-fraction-static 0.80 --cuda-graph-max-bs-decode 32 --max-running-requests 768 \
  --chunked-prefill-size 16384 --context-length 131072 --page-size 256 \
  --enable-mixed-chunk --reasoning-parser deepseek-v3 --tool-call-parser deepseekv32 \
  --enable-metrics --host 0.0.0.0 --port 30000
```

## Benchmark results (concurrency sweep 1 → 512)

Three workload patterns, streaming client on an isolated benchmark node pool:

| Pattern | Peak output tok/s | @ conc | Req/s | TTFT mean @ 512 | TPOT @ 512 |
|---------|------------------:|-------:|------:|----------------:|-----------:|
| `1k/1k` (balanced) | **4,710.94** | 512 | 4.62 | 1.23 s | 106.50 ms |
| `8k/1k` (prefill-heavy) | 4,209.22 | 512 | 4.13 | 7.19 s | 113.38 ms |
| `1k/8k` (reasoning) | 1,606.27 | 512 | 0.68 | 1.55 s | 107.64 ms |

![Output throughput vs concurrency](./results/charts/throughput_vs_concurrency.png)

Full per-concurrency tables, TTFT/TPOT charts, and analysis: **[results/benchamrk_sweep_report.md](./results/benchamrk_sweep_report.md)**

## Files

| File | Purpose |
|------|---------|
| [`sglang-dsv4-flash-2node.yaml`](./sglang-dsv4-flash-2node.yaml) | 2-node StatefulSet serving deployment |
| [`sglang-dsv4-flash-benchmark-runner.yaml`](./sglang-dsv4-flash-benchmark-runner.yaml) | Benchmark client pod (isolated CPU pool) |
| [`results/benchamrk_sweep_report.md`](./results/benchamrk_sweep_report.md) | Full sweep report with charts |
| [`results/charts/`](./results/charts/) | Throughput / TTFT / TPOT PNG charts |
| [`benchmark_runner_full.log`](./benchmark_runner_full.log) | Raw benchmark runner log |

## Reproduce

```bash
# 1. Deploy the server and wait for "The server is fired up and ready!"
kubectl apply -f models/DeepSeekV4-Flash-0731/sglang-dsv4-flash-2node.yaml

# 2. Run the sweep from the isolated benchmark client
kubectl apply -f models/DeepSeekV4-Flash-0731/sglang-dsv4-flash-benchmark-runner.yaml
```

## Related

- [DeepSeek-V4-Pro (1.6T)](../DeepSeekv4/): runs on the same 2-node topology (`--attention-backend dsv4`) — not yet optimized or benchmarked.
- [Docs site page](https://shivajid.github.io/sglang-rtx-pro-6000/#dsv4flash) with interactive chart and comparison to the other recipes.
