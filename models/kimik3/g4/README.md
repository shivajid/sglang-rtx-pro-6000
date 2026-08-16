# Moonshot AI Kimi-K3 on G4 (SM120)

SGLang serving recipe, Hyperdisk ML multi-node weight provisioning, and comprehensive low-to-medium concurrency performance benchmarks for **`moonshotai/Kimi-K3`** (64 MoE + Linear Attention layers, ~1.5 TB real weights) across **4× GCP G4 nodes** (`g4-standard-384`, 32 GPUs).

---

## Note

The server can take up to 30 mins to start. **Also note that on first request, it will kick of a pre-compilations. This can take upto another 10 mins to finish.** Post that it should respond in a normal time frame. G4 nodes are connected to each other over standard ethernet network. For better performance please use GB200 or GB300.

## Configuration

| Item | Value |
|------|-------|
| Model | `moonshotai/Kimi-K3` (Full Real Checkpoint) |
| Hardware | 4× `g4-standard-384` (32× RTX PRO 6000 Ada SM120, 48 GB each) |
| Parallelism | Pipeline Parallelism $PP=4$, Tensor Parallelism $TP=8$ |
| Storage | GCP Hyperdisk ML (`ReadOnlyMany` 2,000 GB, ext4) |
| MoE Runner | `marlin` GEMM backend |
| Attention Backend | `triton` Radix Linear Attention |
| KV Cache | `fp8_e4m3` with `--mamba-full-memory-ratio 0.6` |
| Image | `lmsysorg/sglang:kimi-k3` |
| Context Length | 131,072 |

Key launch flags (full manifest: [`g4_4node_kimik3.yaml`](./g4_4node_kimik3.yaml)):

```bash
sglang serve \
  --model-path /data/model \
  --served-model-name moonshotai/Kimi-K3 \
  --tp-size 8 \
  --pp-size 4 \
  --nnodes 4 \
  --node-rank ${POD_INDEX} \
  --dist-init-addr sglang-kimi-k3-master:20000 \
  --moe-runner-backend marlin \
  --attention-backend triton \
  --cuda-graph-backend-decode=disabled \
  --mem-fraction-static 0.8 \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-full-memory-ratio 0.6 \
  --context-length 131072 \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --host 0.0.0.0 \
  --port 30000 \
  --watchdog-timeout 3600 \
  --trust-remote-code \
  --enable-metrics
```

---

## Highlights

- **Fast & Consistent TPOT**: Baseline single-stream TPOT is **101.7 ms/tok** (~9.8 tok/s). Scaling from 8 to 32 concurrent streams introduces only ~4–8 ms of decode overhead (**105.7–110.4 ms/tok** at 32 streams / ~9.1–9.5 tok/s per user).
- **Sub-360ms TTFT**: For standard 1K input prompts, Time to First Token remains sub-**360 ms** across all concurrencies (311 ms @ C=8, 335 ms @ C=16, 354 ms @ C=32).
- **Lightning-Fast Hyperdisk ML Loading**: Loaded all 96 Safetensor shards (1.53 TB) across 4 nodes in **42.8 seconds total** (**2.97 shards/sec / ~35.7 GB/s aggregate read bandwidth**).
- **Linear Throughput Scaling**: System output throughput scales from **59.1 tok/s at C=8** to **228.6 tok/s at C=32** on balanced `1k/1k` workloads. Peak system ingestion reaches **1,551.48 tok/s** on `8k/1k`.
- **100% Reliability**: Zero dropped requests, zero CUDA OOMs, and zero NCCL socket drops across all 624 benchmark requests.

---

## Connect to Gemini CLI Agent Harness

You can connect the [Gemini CLI](https://github.com/google-gemini/gemini-cli) agent harness directly to your self-hosted Kimi-K3 SGLang server to get an autonomous coding agent (file editing, tool execution, multi-turn reasoning) powered 100% on your own GPUs.

### 1. Forward the Server Port
```bash
# Forward port 30000 from the SGLang cluster service to localhost
kubectl port-forward svc/sglang-kimi-k3-serving 30000:30000
```

### 2. Configure Environment Variables
```bash
export SGLANG_BASE_URL="http://localhost:30000/v1"
export GEMINI_MODEL="moonshotai/Kimi-K3"
export GEMINI_DEFAULT_AUTH_TYPE="sglang"
```

### 3. Launch the Agent
```bash
# Launch the patched Gemini CLI in the target repository
npx @shivajidnpm2026/gemini-cli
```

> 📖 For full setup details, Node.js prerequisites, and troubleshooting, see **[`sglang_gemini_cli/README.md`](../../../sglang_gemini_cli/README.md)**.

---

## Storage & Hyperdisk ML Setup

Model weights (~1.5 TB) are served directly from **Google Cloud Hyperdisk ML**, eliminating slow per-node downloads and local ephemeral disk constraints:

1. **Phase 1: Ingestion (`ReadWriteOnce`)**: A 2 TB Hyperdisk ML disk is attached to a temporary ingestion job that synchronizes checkpoint weights from GCS via `gcloud storage rsync`.
2. **Phase 2: Serving (`ReadOnlyMany`)**: The volume is rebound as `ReadOnlyMany` and mounted simultaneously at `/data/model` across all 4 SGLang StatefulSet pods.

Full step-by-step instructions and manifests: **[`HYPERDISK_ML_SETUP_GUIDE.md`](./HYPERDISK_ML_SETUP_GUIDE.md)**.

---

## Benchmark Results (Concurrency Sweep 8 → 32)

Evaluated across four distinct workload patterns using the official `python3 -m sglang.bench_serving` client suite on a dedicated 64-vCPU client node pool (`cpu-64-pool`):

| Pattern | Input / Output | Peak Output Tok/s | Total Tok/s @ C=32 | Median TTFT @ C=32 | Mean TPOT @ C=32 | Median ITL |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1k/1k` (Balanced)** | `1000 / 1000` | **228.64 tok/s** | 439.65 tok/s | **353.5 ms** | **110.38 ms** | 103.8 ms |
| **`1k/8k` (Reasoning)** | `1000 / 8000` | **224.77 tok/s** | 249.16 tok/s | **352.8 ms** | **105.65 ms** | 103.9 ms |
| **`1k/500` (Short Chat)** | `1000 / 500` | **205.55 tok/s** | 579.60 tok/s | **360.4 ms** | **118.04 ms** | 104.2 ms |
| **`8k/1k` (Long Context)** | `8000 / 1000` | **177.47 tok/s** | **1,551.48 tok/s** | **1,179.1 ms** | **157.66 ms** | 104.4 ms |

### Visual Scaling Charts

![Output Throughput vs Concurrency](charts/throughput_vs_concurrency.svg)

![TTFT vs Concurrency](charts/ttft_vs_concurrency.svg)

![TPOT vs Concurrency](charts/tpot_vs_concurrency.svg)

Full per-scenario latency tables, P99 breakdowns, and analysis: **[`BENCHMARK_REPORT.md`](./BENCHMARK_REPORT.md)**.

---

## Files

| File | Purpose |
|---|---|
| [`g4_4node_kimik3.yaml`](./g4_4node_kimik3.yaml) | 4-Node G4 StatefulSet + Headless & ClusterIP Service manifest |
| [`sglang-bench-sweep-job.yaml`](./sglang-bench-sweep-job.yaml) | Official `sglang.bench_serving` 12-matrix benchmark sweep Job |
| [`HYPERDISK_ML_SETUP_GUIDE.md`](./HYPERDISK_ML_SETUP_GUIDE.md) | Complete guide for provisioning Hyperdisk ML and weight ingestion |
| [`BENCHMARK_REPORT.md`](./BENCHMARK_REPORT.md) | Full performance benchmark report with charts and analysis |
| [`model_weight_disk/kimik3-hdml-writer.yaml`](./model_weight_disk/kimik3-hdml-writer.yaml) | PersistentVolume & PVC in `ReadWriteOnce` mode for initial download |
| [`model_weight_disk/kimik3-downloader-job.yaml`](./model_weight_disk/kimik3-downloader-job.yaml) | Kubernetes Job to sync checkpoint from GCS to Hyperdisk ML |
| [`model_weight_disk/kimik3-hdml-ro.yaml`](./model_weight_disk/kimik3-hdml-ro.yaml) | PersistentVolume & PVC in `ReadOnlyMany` mode for distributed serving |
| [`charts/`](./charts/) | High-resolution SVG line charts for throughput, TTFT, and TPOT |

---

## Quick Start & Shipping Guide

Follow these high-level steps to ship this recipe in your GKE cluster:

### Step 1: Provision Hyperdisk ML & Sync Model Weights
```bash
# 1. Create the 2TB Hyperdisk ML disk in GCP
gcloud compute disks create kimik3-hyperdisk-ml \
    --project=<YOUR_PROJECT_ID> \
    --zone=<YOUR_ZONE> \
    --type=hyperdisk-ml \
    --size=2000GB

# 2. Mount in ReadWriteOnce mode and download weights from GCS
kubectl apply -f model_weight_disk/kimik3-hdml-writer.yaml
kubectl apply -f model_weight_disk/kimik3-downloader-job.yaml
kubectl logs -f job/kimik3-hdml-downloader

# 3. Once downloaded, switch to ReadOnlyMany mode
kubectl delete job kimik3-hdml-downloader
kubectl delete pvc kimik3-hdml-writer-pvc
kubectl delete pv kimik3-hdml-pv
kubectl apply -f model_weight_disk/kimik3-hdml-ro.yaml
```

### Step 2: Launch the 4-Node SGLang Serving Cluster
```bash
# Apply the 4-Node StatefulSet with SM120 patch and Hyperdisk ML mount
kubectl apply -f g4_4node_kimik3.yaml

# Follow the master node logs until the server is ready:
kubectl logs -f sglang-kimi-k3-g4-0
# Expect: "The server is fired up and ready!"
```

### Step 3: Run the Benchmark Sweep
```bash
# Launch the benchmark job on the CPU node pool
kubectl apply -f sglang-bench-sweep-job.yaml

# Stream live benchmark execution logs
kubectl logs -f job/sglang-bench-sweep-job
```
