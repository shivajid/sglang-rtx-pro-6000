# zai-org/GLM-5.2

Serving recipes and benchmarks for **GLM-5.2** (744B MoE) on Google Cloud — **G4** (RTX PRO 6000,
SM120) in both NVFP4 and FP8, plus a single-host **GB300** (A4X Max) validation.

| Recipe | Hardware | Quantization | Config | Status |
|--------|----------|--------------|--------|--------|
| [**NVFP4 · 1 node**](#nvfp4--1-node-g4) | 8× RTX PRO 6000 | `nvidia/GLM-5.2-NVFP4` | [`sglang-glm52-nvfp4-1node.yaml`](./nvfp4/sglang-glm52-nvfp4-1node.yaml) | ✅ Working — best per-GPU |
| [**FP8 · 2 nodes**](#fp8--2-nodes-g4) | 16× RTX PRO 6000 | `zai-org/GLM-5.2-FP8` | [`sglang-glm52-fp8-2node.yaml`](./fp8/sglang-glm52-fp8-2node.yaml) | ✅ Working — highest aggregate |
| [**GB300 · single host**](#gb300--single-host-a4x-max) | 4× GB300 | `modelopt_fp4` | [`GB300_GLM52_single_host_setup.md`](./GB300_GLM52_single_host_setup.md) | ✅ Working — no throughput run |
| NVFP4 · 2 nodes | 16× RTX PRO 6000 | `modelopt_fp4` | [`..._2node_notworking.yaml`](./nvfp4/sglang-glm52-nvfp4-2node_notworking.yaml) | ❌ **Does not work** |

> 📖 Rendered docs: **[GLM-5.2 on the site](https://shivajid.github.io/sglang-rtx-pro-6000/#glm52)**

---

## Results

Standard profile: 1024 input / 8192 output tokens, `inf` request rate, isolated load generator.

| Recipe | Concurrency | Prompts | Output tok/s | Total tok/s | Peak output | Median TPOT | Median TTFT | tok/s **per GPU** |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **NVFP4 · 1 node** (8 GPU) | 64 | 192 | 461.27 | 518.93 | 640.00 | **108.59 ms** | **1,017 ms** | 57.7 |
| **NVFP4 · 1 node** (8 GPU) | 128 | 384 | 1,100.92 | 1,238.53 | 1,280.00 | 115.00 ms | 12,571 ms | **137.6** |
| **FP8 · 2 nodes** (16 GPU) | 512 | 1,536 | **1,645.21** | **1,855.07** | **2,608.00** | 240.43 ms | 440 ms | 102.8 |

### NVFP4: 64 → 128 concurrency

Doubling concurrency raises output throughput **2.39×** (461 → 1,101 tok/s) while median TPOT barely
moves (108.6 → 115.0 ms). The cost is entirely in prefill latency: median TTFT goes from **1.0 s to
12.6 s**, because the 2,048-token chunked prefill and 9,472-token context window leave little
headroom once 128 streams are admitted.

> ⚠️ **The C=64 run has a stall in it.** Max ITL is 447 s and mean TPOT (136.6 ms) sits well above
> the median (108.6 ms), with P90 end-to-end latency at 1.62 M ms against a 890 K ms median. That
> single-stream hiccup depresses the C=64 aggregate, which is most likely why 64 → 128 looks
> superlinear. Treat the 2.39× as an upper bound and re-run C=64 before quoting it.

**Comparing NVFP4 to FP8 is not like-for-like** — different concurrency, different node counts. What
the numbers do support:

- **NVFP4 on one node is ~34% more efficient per GPU** at C=128 (137.6 vs 102.8 tok/s/GPU) and
  delivers **less than half the TPOT** (115 vs 240 ms). For cost per token or per-user decode speed,
  the single-node NVFP4 build is the better recipe.
- **FP8 across two nodes wins on aggregate throughput** (1,645 vs 1,101 output tok/s) because it has
  2× the GPUs and was driven at 4× the concurrency.
- **On TTFT, pick by operating point.** NVFP4 at C=64 is the most responsive configuration measured
  here (1.0 s median). FP8 at C=512 holds a 440 ms median but with an 8.2 s mean — a long tail.
  NVFP4 at C=128 is the worst of the three at 12.6 s.

The sweep job is configured for 128 / 256 / 512; **256 and 512 have not been run** on NVFP4 yet.

Raw output: [`nvfp4/resuts/benchmark_results.md`](./nvfp4/resuts/benchmark_results.md) ·
[`fp8/results/benchmark_results.yaml`](./fp8/results/benchmark_results.yaml)

---

## NVFP4 · 1 node (G4)

Eight RTX PRO 6000s, TP8 with DP8 attention, NVIDIA's NVFP4 checkpoint. Weights are mounted from
Hyperdisk ML at `/models/GLM-5.2-NVFP4`.

**Config: [`nvfp4/sglang-glm52-nvfp4-1node.yaml`](./nvfp4/sglang-glm52-nvfp4-1node.yaml)**

| Setting | Value |
|---------|-------|
| Image | `lmsysorg/sglang:dev-cu13` |
| Node pool | `g4-384-pool-pm-crwd` (`g4-standard-384`) |
| Parallelism | `--tp 8 --dp-size 8 --enable-dp-attention --ep-size 1 --moe-a2a-backend none` |
| MoE runner | `flashinfer_cutlass` |
| Attention | `flashinfer`, DSA prefill/decode forced to `trtllm` |
| KV cache | `fp8_e4m3` |
| Memory | `--mem-fraction-static 0.975`, `--chunked-prefill-size 2048` |
| Context | `--context-length 9472`, `--max-running-requests 1024` |
| CUDA graph | `--cuda-graph-bs 16 32 48 64`, `--disable-piecewise-cuda-graph` |
| Parsers | `--reasoning-parser glm45 --tool-call-parser glm47` |

### Deploy

```bash
# 1. Provision weights on Hyperdisk ML (once)
kubectl apply -f nvfp4/glm52-hdml-writer.yaml
kubectl apply -f nvfp4/glm52-downloader-job.yaml
kubectl logs -f job/glm52-downloader

# 2. Rebind the volume ReadOnlyMany and start the server
kubectl apply -f nvfp4/glm52-hdml-ro.yaml
kubectl apply -f nvfp4/sglang-glm52-nvfp4-1node.yaml
kubectl port-forward svc/sglang-glm52-nvfp4 8000:8000
```

### Correctness gate — run this before benchmarking

This recipe has a silent-corruption failure mode (see below), so verify accuracy first:

```bash
python3 -m sglang.test.few_shot_gsm8k --num-questions 50 --port 8000
# Expect: Accuracy ~0.900, Invalid 0.000
```

If accuracy collapses or completions degenerate to `!`, the DSA backend flags did not take effect.

### Benchmark

```bash
# the two points benchmarked so far (prompts = 3× concurrency)
python3 -m sglang.bench_serving --backend sglang --model nvidia/GLM-5.2-NVFP4 \
  --dataset-name random --random-input-len 1024 --random-output-len 8192 \
  --random-range-ratio 1.0 --max-concurrency 64 --num-prompts 192

python3 -m sglang.bench_serving --backend sglang --model nvidia/GLM-5.2-NVFP4 \
  --dataset-name random --random-input-len 1024 --random-output-len 8192 \
  --random-range-ratio 1.0 --max-concurrency 128 --num-prompts 384

# or the full sweep job (128 / 256 / 512, 3× prompts per level)
kubectl apply -f nvfp4/sglang-glm52-benchmark-sweep-job.yaml
```

Scripts: [`run_benchmark_sweep.sh`](./nvfp4/run_benchmark_sweep.sh) ·
[`benchmark_sweep_1k8k.py`](./nvfp4/benchmark_sweep_1k8k.py)

### Four fixes this config depends on

Each of these was a real failure, not a precaution:

1. **Pass `--cuda-graph-bs` values as separate array elements** (`"16"`, `"32"`, `"48"`, `"64"`).
   A single space-separated string trips argparse integer conversion and the pod enters
   `CrashLoopBackOff`.
2. **`--disable-piecewise-cuda-graph`** — prefill CUDA-graph capture unconditionally reads DSA
   indexer metadata, which is `None` under `--attention-backend flashinfer`. Without this flag the
   rank schedulers die during init with
   `AttributeError: 'NoneType' object has no attribute 'get_seqlens_expanded'`.
3. **Force `--dsa-prefill-backend trtllm --dsa-decode-backend trtllm`** — this is the silent one.
   The auto-selected `flashinfer_sparse_mla` on SM120 with an FP8 KV cache emits **NaN logits during
   decode**, collapsing every completion to token 0 (`!`). The server stays up and throughput looks
   fine. Forcing `trtllm` restores GSM8K to ~0.900 and increases KV-cache token capacity.
4. **`SGLANG_DISABLE_DSA_INDEXER_FUSION=1`** plus
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, with NCCL tuned for the single-node 8-GPU
   interconnect.

---

## FP8 · 2 nodes (G4)

Sixteen RTX PRO 6000s across two nodes, `TP=8 · PP=2` with DP8 attention, on the stock
`zai-org/GLM-5.2-FP8` checkpoint.

**Config: [`fp8/sglang-glm52-fp8-2node.yaml`](./fp8/sglang-glm52-fp8-2node.yaml)**

```bash
python3 -m sglang.launch_server \
  --model zai-org/GLM-5.2-FP8 --quantization fp8 \
  --tensor-parallel-size 8 --pipeline-parallel-size 2 \
  --nnodes 2 --node-rank $POD_INDEX --dist-init-addr sglang-master-pod:5000 \
  --dp-size 8 --enable-dp-attention \
  --attention-backend flashinfer --kv-cache-dtype fp8_e4m3 \
  --moe-a2a-backend none --ep-size 1 --moe-runner-backend triton \
  --disable-shared-experts-fusion --disable-radix-cache \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  --mem-fraction-static 0.80 --page-size 64 \
  --host 0.0.0.0 --port 8000
```

```bash
kubectl apply -f fp8/sglang-glm52-fp8-2node.yaml
```

This is an **early, unoptimized config** — radix cache is disabled and it uses the `triton` MoE
runner rather than a FlashInfer backend. Expect it to improve; the NVFP4 recipe above is the more
tuned of the two.

---

## GB300 · single host (A4X Max)

Validation of the Blackwell-optimized stack on **GCP A4X Max**, 4 GPUs at `TP=4`, run under Docker on
a bare Ubuntu 24.04 ARM64 host rather than GKE. This deployment adds **EAGLE speculative decoding**,
which the G4 recipes do not use.

**Guide: [`GB300_GLM52_single_host_setup.md`](./GB300_GLM52_single_host_setup.md)** — a phased,
repeatable script covering system inspection, NVMe scratch mount, HuggingFace weight download, and
server launch.

```bash
python3 -m sglang.launch_server \
  --model-path /model --tensor-parallel-size 4 \
  --quantization modelopt_fp4 \
  --max-running-requests 16 --max-prefill-tokens 8192 \
  --chunked-prefill-size 8192 --cuda-graph-max-bs 16 \
  --mem-fraction-static 0.87 --kv-cache-dtype fp8_e4m3 \
  --bf16-gemm-backend cutedsl \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  --speculative-algorithm EAGLE --speculative-num-steps 5 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 \
  --trust-remote-code --host 0.0.0.0 --port 8002
```

**No throughput benchmark was run here** — this was qualitative validation that the stack comes up
and serves correctly on GB300. The numbers in the results table above are all from G4.

Prerequisites: Ubuntu 24.04 ARM64, an NVMe device to format for scratch (`/dev/nvme1n1` in the
guide), sudo, and a HuggingFace token with access to `nvidia/GLM-5.2-NVFP4`.

For the rack-scale architecture behind GB300 — NVL72, Fabric Manager, NVLSM, IMEX, partitions — see
[`GB300_Reading_Guide.md`](./GB300_Reading_Guide.md).

---

## Known not working

[`nvfp4/sglang-glm52-nvfp4-2node_notworking.yaml`](./nvfp4/sglang-glm52-nvfp4-2node_notworking.yaml)
— NVFP4 across 2 nodes (`TP=8 · PP=2 · DP=8`, `modelopt_fp4`, image
`europe-west4-docker.pkg.dev/northam-ce-mlai-tpu/sglang-repo/sglang:glm-opt`). Kept in the repo for
reference. Use the 1-node NVFP4 or 2-node FP8 recipe instead.

---

## Files

| Path | Purpose |
|------|---------|
| [`nvfp4/sglang-glm52-nvfp4-1node.yaml`](./nvfp4/sglang-glm52-nvfp4-1node.yaml) | 1-node NVFP4 serving Deployment |
| [`nvfp4/glm52-hdml-writer.yaml`](./nvfp4/glm52-hdml-writer.yaml) · [`glm52-downloader-job.yaml`](./nvfp4/glm52-downloader-job.yaml) · [`glm52-hdml-ro.yaml`](./nvfp4/glm52-hdml-ro.yaml) | Hyperdisk ML provisioning: write → download → rebind read-only |
| [`nvfp4/sglang-glm52-benchmark-sweep-job.yaml`](./nvfp4/sglang-glm52-benchmark-sweep-job.yaml) | Benchmark sweep Job (128 / 256 / 512) |
| [`nvfp4/run_benchmark_sweep.sh`](./nvfp4/run_benchmark_sweep.sh) · [`benchmark_sweep_1k8k.py`](./nvfp4/benchmark_sweep_1k8k.py) | Sweep driver scripts |
| [`nvfp4/resuts/benchmark_results.md`](./nvfp4/resuts/benchmark_results.md) | NVFP4 raw benchmark output |
| [`nvfp4/README.md`](./nvfp4/README.md) | NVFP4 config notes and fix rationale |
| [`fp8/sglang-glm52-fp8-2node.yaml`](./fp8/sglang-glm52-fp8-2node.yaml) | 2-node FP8 serving manifest |
| [`fp8/results/benchmark_results.yaml`](./fp8/results/benchmark_results.yaml) | FP8 raw benchmark output |
| [`GB300_GLM52_single_host_setup.md`](./GB300_GLM52_single_host_setup.md) | GB300 single-host setup script |
| [`GB300_Reading_Guide.md`](./GB300_Reading_Guide.md) | NVL72 / GB300 architecture reading guide |
