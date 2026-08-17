# moonshotai/Kimi-K3

Serving recipes, benchmarks, and agentic evaluations for **Kimi-K3** (released July 27, 2026) — a
2.8T-parameter / ~104B-active MoE reasoning model with 64 MoE + Linear Attention layers and a ~1.5 TB
checkpoint.

This directory covers **two very different deployments**:

| | [GB200](#gb200--a4x-grace-blackwell) | [G4](./g4/) |
|---|---|---|
| Hardware | 4 or 8 × A4X (16 / 32 × GB200) | 4 × `g4-standard-384` (32 × RTX PRO 6000) |
| Interconnect | NVLink + NCCL GIB plugin | **Plain VPC ethernet** — no NVLink, no RDMA |
| Parallelism | `TP=16 · DCP=16` (4n) / `TP=32 · DCP=32` (8n) | `PP=4 · TP=8` |
| Peak output | 1,874.86 tok/s (`1k/8k` @ 128c) | 583.58 tok/s (`1k/8k` @ 112c) |
| Purpose | Throughput sweeps + ViBench agentic runs | Proving K3 runs on commodity G4 at all |

> 📖 Rendered docs: **[Kimi-K3 on the site](https://shivajid.github.io/sglang-rtx-pro-6000/#kimik3)**

---

## Contents

| Path | What's in it |
|------|--------------|
| [`g4/`](./g4/) | **4-node G4 recipe** — configs, benchmark report, Hyperdisk ML weight provisioning, charts |
| [`gb200_4node_kimik3.yaml`](./gb200_4node_kimik3.yaml) | GB200 4-node LeaderWorkerSet (`TP=16 · DCP=16`) |
| [`gb200_8node_kimik3.yaml`](./gb200_8node_kimik3.yaml) | GB200 8-node LeaderWorkerSet (`TP=32 · DCP=32`) |
| [`results/`](./results/) | GB200 concurrency sweep 1 → 512, three workload patterns, with charts |
| [`Vibebench/`](./Vibebench/) | ViBench runner manifests (24-app agentic coding suite) |
| [`vibench_hard_results/`](./vibench_hard_results/) | ViBench **Hard** — brownfield feature extension, per-project scorecards |
| [`vibench_kimik3_final_report.md`](./vibench_kimik3_final_report.md) | ViBench full report |
| [`rev2_prgrs_vibench_kimik3_final_report.md`](./rev2_prgrs_vibench_kimik3_final_report.md) | ViBench rev2 progress report |
| [`chatapp/`](./chatapp/) | K3 Console — FastAPI + JS chat UI, with deployment/service/configmap |

---

## GB200 — A4X (Grace Blackwell)

ARM64, 4 GPUs per node, NVLink within a block, NCCL GIB plugin. Deployed as a LeaderWorkerSet;
the leader exposes port `30100` via ClusterIP service `sglang-kimi-k3-svc`.

```bash
python3 -m sglang.launch_server \
  --model-path /data/model --served-model-name moonshotai/Kimi-K3 \
  --tp-size 16 --dcp-size 16 \
  --nnodes ${SIZE} --node-rank ${RANK} --dist-init-addr ${LEADER_HOST}:20000 \
  --trust-remote-code \
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 \
  --mamba-full-memory-ratio 7.21 \
  --mem-fraction-static 0.85 --host 0.0.0.0 --port 30100
```

The 8-node variant uses `--tp-size 32 --dcp-size 32`, `--mamba-full-memory-ratio 0.62`, and
`--watchdog-timeout 3600`.

### Concurrency sweep — 4 nodes, 16× GB200

| Workload pattern | Peak total throughput | Optimal concurrency | Stream speed |
| :--- | :---: | :---: | :---: |
| `1k / 1k` (balanced) | **2,883.45 tok/s** | 256 | 23.68 tok/s |
| `8k / 1k` (prompt-heavy) | 2,731.09 tok/s | 128 | 24.26 tok/s |
| `1k / 8k` (reasoning) | 1,874.86 tok/s | 128 | 24.88 tok/s |

Throughput scales near-linearly to concurrency 128, then plateaus. TTFT stays in the low seconds
through 128 and jumps one to two orders of magnitude at 256+ as requests queue for KV-cache blocks —
that is queueing, not decode slowdown. A single stream decodes at ~48 tok/s; under 128–512
simultaneous streams per-user speed settles at ~23–25 tok/s. 100% success rate at every level.

Full tables and charts: [`results/benchamrk_sweep_report.md`](./results/benchamrk_sweep_report.md).

---

## G4 — RTX PRO 6000 (SM120)

The harder problem, and the one this repo exists for: the same ~1.5 TB checkpoint across four
commodity G4 nodes over plain VPC ethernet, with no NVLink and no GPU RDMA. Uses `PP=4 · TP=8`,
Marlin MoE with an SM120 patch, Triton radix linear attention, FP8 KV cache, and HiCache spilling to
host RAM. Weights come off Hyperdisk ML rather than per-node downloads — 1.53 TB in 42.8 s.

Peak 583.58 output tok/s at concurrency 112; **concurrency 64 is the operating point** at 480.86
output tok/s with zero queueing and sub-530 ms median TTFT.

Everything for this deployment — configs, benchmark report, storage guide — lives in **[`g4/`](./g4/)**.

---

## Agentic evaluations

Both suites run against the 8-node deployment with 4-way-parallel OpenHands agents and Playwright
browser assertions.

| Suite | Task shape | Score | Report |
|-------|-----------|-------|--------|
| **ViBench** | Build 24 full-stack web apps end to end | **95.6 / 100** (regular reasoning)<br>81.5 with `reasoning_effort: low` | [report](./vibench_kimik3_final_report.md) |
| **ViBench Hard** | Extend 10 existing codebases, 20 feature artifacts, 63 test plans | **79.2 / 100**, zero failed artifacts | [report](./vibench_hard_results/VIBENCH_HARD_KIMIK3_REPORT.md) |

On ViBench Hard, scores *rise* with task complexity — 64.2 on initial schema extensions vs 96.8 on
complex brownfield refactoring. The one systematic failure was `slack` (33.8, 0/7 plans): real-time
multi-context synchronization produced duplicate DOM state.

---

## Operational notes

Hard-won specifics. Most of these cost a debugging session to find.

### Manage the reasoning trace

Kimi-K3 reasons at length. For agentic clients, set `"reasoning_effort": "low"` in the request or
long multi-step runs will exhaust the context window:

```json
{ "reasoning_effort": "low" }
```

This was required for SWE-Bench runs. Be aware of the tradeoff — on ViBench, `low` costs **14 points**
of score (81.5 vs 95.6), and both zero-score apps recovered to 100 with the regular trace. Use `low`
to fit the context, not as a default.

### `--mamba-full-memory-ratio` scales with request length

Set it from your average request length (ISL + OSL):

| Avg request length | `--mamba-full-memory-ratio` |
|---:|---:|
| 11,264 | 7.21 |
| 32,768 | 2.48 |
| 65,536 | 1.24 |
| 131,072 | 0.62 |
| 262,144 | 0.31 |
| 524,288 | 0.16 |
| 1,048,576 | 0.078 |

### NCCL timeouts on 8 nodes

Without these, 8-node runs fail with NCCL errors:

```yaml
- name: TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC
  value: "3600"
- name: NCCL_HEARTBEAT_TIMEOUT_SEC
  value: "3600"
```

### Pin GB200 pods to one NVLink block

All pods must land in the same topology block or you lose NVLink between them:

```yaml
spec:
  nodeSelector:
    cloud.google.com/gce-topology-block: "1acd074d42cd3be9e4486b524db2e9ab"
```

### HiCache behaves differently across platforms

On **GB200**, enabling HiCache with the following config **hangs and crashes** — the shipped GB200
manifests here therefore run without it:

```bash
--enable-hierarchical-cache --page-size 64 --hicache-ratio 2.0 \
--hicache-io-backend direct --hicache-mem-layout page_first_direct \
--hicache-write-policy write_through --hicache-storage-prefetch-policy=timeout
```

On **G4**, HiCache is enabled and working in
[`g4/g4-4node-kk3-agentic.yaml`](./g4/g4-4node-kk3-agentic.yaml) with a different combination —
`--hicache-ratio 1.0`, `--hicache-io-backend direct`, `--hicache-mem-layout page_first`. The
`page_first_direct` memory layout and `hicache-ratio 2.0` are the notable differences from the GB200
attempt that failed.

### Startup is slow

On G4 the server can take **up to 30 minutes** to start, and the first request triggers warmup
pre-compilation before response times normalize. Budget for this in readiness probes —
`--watchdog-timeout 3600` is set for a reason.

---

## Other experiments

- **DSpark configs** — speculative decoding experiments, not yet folded into the shipped recipes.
- **HiCache on GB200** — see above; unresolved.

---

## Connect an agent

Both deployments expose an OpenAI-compatible `/v1` endpoint, so the
[Gemini CLI harness](../../sglang_gemini_cli/README.md) can drive them directly:

```bash
kubectl port-forward svc/sglang-kimi-k3-serving 30000:30000
export SGLANG_BASE_URL="http://localhost:30000/v1"
export GEMINI_MODEL="moonshotai/Kimi-K3"
export GEMINI_DEFAULT_AUTH_TYPE="sglang"
npx @shivajidnpm2026/gemini-cli
```

A lightweight chat UI for the deployment (K3 Console) is in [`chatapp/`](./chatapp/).
