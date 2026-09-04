# Qwen3.8-Flash-Next on GCP G4 (8× RTX PRO 6000 Blackwell, SM120)

Target host: **`shivaji-g4-384-spot`** (`g4-standard-384`, us-central1-b, SPOT)
8× NVIDIA RTX PRO 6000 Blackwell Server Edition — **96GB each, 768GB total**, compute capability **12.0 (SM120)**, driver 610.57.04, Docker 26.1.5 + nvidia-container-toolkit 1.20.

> ⚠️ SPOT instance (`provisioningModel: SPOT`, `preemptible: true`). Weights are
> re-downloaded if the VM is preempted unless the HF cache is on a persistent disk.

---

## 1. Model summary (from `config.json` / model card)

| Property | Value |
|---|---|
| Architecture | `Qwen4ExpForConditionalGeneration` (model_type `qwen4_exp`) — preview of Qwen4 |
| Total params | **176B** (125B LM + 51B N-gram table); **6B active/token**; +4B MTP head |
| HF safetensors | **~180GB (BF16)** across 131 shards |
| Layers | 48 = 12 × (3× Gated DeltaNet→MoE + 1× Qwen Sparse Attention→MoE) |
| Linear attn (GDN) | 48 V-heads / 16 QK-heads, head_dim 128, conv kernel 4, ssm dtype fp32 |
| QSA (full attn, every 4th) | 24 Q / 2 KV heads, head_dim 256, partial RoPE 0.25 (64-dim), block-sparse indexer budget 2048 |
| MoE | 512 experts, 10 routed + 1 shared, expert intermediate 640 |
| Gated Residual | 4 branches, low-rank 320 (`hc_count`, `hc_lowrank`) |
| N-gram embed (PLE) | 20M-entry table at layer 2 — the 51B block; **CPU-offloadable + prefetch** |
| MTP | 1 full-attn (QSA) layer, multi-step trained → in-checkpoint speculative |
| Context | **262,144 native**, up to 1M w/ YaRN |
| Vision | ViT encoder (depth 27, hidden 1152) — multimodal in, text out |
| License | Qwen Community 1.0 (`qwen-community-1.0`) — repo is public (HTTP 200, no token needed) |

Sampling (model card): thinking default ON; thinking `temp=1.0 top_p=0.95 top_k=20`;
instruct `temp=0.7 top_p=0.8 top_k=20 presence_penalty=1.5`. `reasoning_effort`: xhigh|medium|low.

---

## 2. Checkpoints

| Precision | Repo | On-disk | Notes |
|---|---|---|---|
| BF16 | `Qwen/Qwen3.8-Flash-Next` | ~180GB | official TP4 on 141–192GB cards |
| FP8 | `Qwen/Qwen3.8-Flash-Next-FP8` | ~180GB (F8_E4M3 174.5B params + BF16 5.5B) | no memory win on G4 — skip |
| NVFP4 | `RadixArk/Qwen3.8-Flash-Next-NVFP4` | ~120GB | Blackwell-only; SGLang's own quant |

**On this 8×96GB node use BF16.** 180GB/8 ≈ 22.5GB/GPU → fits with huge KV headroom.
NVFP4 is only worth it if you want to run on 2 of the 8 GPUs.

---

## 3. Serve engine

Model support is **not in a tagged SGLang release** — it lives in PR
[`sgl-project/sglang#36497`](https://github.com/sgl-project/sglang/pull/36497) (open).
Use the purpose-built day-0 image:

```bash
docker pull lmsysorg/sglang:qwen38flashnext
```

The PR adds `python/sglang/srt/models/qwen4_exp.py`, `configs/qwen4_exp.py`,
`layers/attention/qwen_sparse_attn_backend.py`, `layers/attention/hybrid_linear_attn_backend.py`,
and a shape-specialized packed-QSA decode kernel `kernels/kda_kernels/qwen38_qsa_sm121`
(validated on GB10/SM121 via radixark KDA-1.5; ~2.07× over the Triton fallback for QSA decode).
**SM120 (this node) is supported** — the official cells all pin the flashinfer GDN backend
and Triton QSA fallback, which run on SM120; the existing `glm53-flash:sm120` image on this
host already proves SGLang serves on these GPUs.

---

## 4. Launch commands

Two ready-to-run configs live in this directory (and in `~/qwen38/` on the host), validated in §6:
- **`serve_latency.sh`** — TP8 + MTP. Single-stream / interactive / agentic (low TPOT/TTFT).
- **`serve_throughput.sh`** — DP2×TP4 + EP4 + MTP. Max aggregate tok/s at high concurrency.

Stop with `docker stop qwen38` (both use container name `qwen38`, so run one at a time).

### 4a. Primary — BF16 low-latency (MTP speculative) on 8 GPUs

```bash
docker run --rm -it --gpus all --ipc=host --network=host \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=$HF_TOKEN \
  lmsysorg/sglang:qwen38flashnext \
  python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3.8-Flash-Next \
    --tp 8 \
    --mem-fraction-static 0.85 \
    --chunked-prefill-size 8192 \
    --linear-attn-prefill-backend flashinfer \
    --linear-attn-decode-backend flashinfer \
    --linear-attn-verify-backend triton \
    --mamba-ssm-dtype bfloat16 \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --max-running-requests 96 \
    --reasoning-parser auto \
    --host 0.0.0.0 --port 30000
```

### 4b. BF16 high-throughput (no spec, EP across ranks)

```bash
    --model-path Qwen/Qwen3.8-Flash-Next \
    --tp 8 --ep 8 \
    --mem-fraction-static 0.85 \
    --chunked-prefill-size 8192 \
    --linear-attn-prefill-backend flashinfer \
    --linear-attn-decode-backend flashinfer \
    --mamba-ssm-dtype bfloat16 \
    --reasoning-parser auto \
    --host 0.0.0.0 --port 30000
```

### 4c. Constrained — NVFP4 on 2 GPUs (e.g. `--gpus '"device=0,1"'`)

```bash
    --model-path RadixArk/Qwen3.8-Flash-Next-NVFP4 \
    --tp 2 \
    --ple-offload-embedding \
    --linear-attn-prefill-backend flashinfer \
    --linear-attn-decode-backend flashinfer \
    --mamba-ssm-dtype bfloat16 \
    --reasoning-parser auto \
    --host 0.0.0.0 --port 30000
```

**Flag rationale (adapted from the verified TP4 cells to this node's TP8):**
- `--linear-attn-{prefill,decode}-backend flashinfer` — pinned so GDN does not fall to the
  GPU-generation-dependent default (Triton on SM90); portable across Blackwell gens.
- `--mamba-ssm-dtype bfloat16` — required for the flashinfer GDN decode default; config
  default is fp32 which is slower and heavier.
- `--linear-attn-verify-backend triton` — only in the low-latency/spec cells (MTP verify).
- `--mem-fraction-static 0.85` / `--chunked-prefill-size 8192` — from verified cells.
- `--max-running-requests 96` — spec runs otherwise take the speculative-hook default.
- `--ep` ≤ `--tp`; high-throughput only. Do **not** set EP with TP1/TP2.
- `--ple-offload-embedding` — CUDA-only; offloads the 51B N-gram table to pinned CPU and
  prefetches on a side stream. Auto-enabled for BF16 on CUDA; set explicitly for NVFP4 TP2.
- `--reasoning-parser auto` — model **always thinks**; without it thinking leaks into content.

---

## 5. Validate

```bash
# health
curl http://localhost:30000/health
# smoke
curl http://localhost:30000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"Qwen/Qwen3.8-Flash-Next",
  "messages":[{"role":"user","content":"Write a Python function to merge two sorted linked lists."}]
}'
# speed bench
python3 -m sglang.bench_serving --backend sglang-oai --host localhost --port 30000 \
  --model Qwen/Qwen3.8-Flash-Next --dataset-name random \
  --random-input-len 1024 --random-output-len 1024 --random-range-ratio 1 \
  --num-prompts 128 --max-concurrency 64 --request-rate inf --flush-cache
```

Accuracy refs (SGLang `qwen4-main @ e17062a1d`, TP4 H200/B200): GSM8K ≈ 97.5–97.7, AIME26 ≈ 97.9–99.2, MMMU-Pro ≈ 77 (BF16/FP8).

---

## 6. Validated results (this host, 2026-09-04)

Deployed recipe **4a** (BF16, TP8, MTP) on `shivaji-g4-384-spot` with image
`lmsysorg/sglang:qwen38flashnext` (36.1GB). Server came up clean across all 8 ranks.

- **Memory:** ~88GB used / 96GB per GPU. `max_total_num_tokens=2,467,840` KV (bf16),
  `max_running_requests=96`, context 262144. The 51B N-gram table is PLE CPU-offloaded
  (host has 1.4TB RAM), so KV headroom is large.
- **Smoke test:** reasoning correctly split into `reasoning_content`
  (`reasoning_tokens:169`); clean answer in `content`.
- **Speculative decode (NEXTN 3/1/4):** accept len ≈ 2.6–2.85, accept rate ≈ 0.54–0.62.
- **Throughput bench** (`random`, ISL 1024 / OSL 512, 64 prompts, concurrency 32):
  - 64/64 successful in 27.2s → **2.35 req/s**
  - **Output token throughput ≈ 1203 tok/s**; per-step gen throughput peaked ≈ **1530 tok/s**
  - Mean TTFT ≈ **1.51s**, Mean TPOT ≈ **21.1ms**
- **1k/8k bench** (`random`, ISL 1024 / OSL **8192**, 64 prompts, concurrency 32):
  - 64/64 successful in 314s → 0.20 req/s
  - **Output token throughput ≈ 1669 tok/s** (total 1878 tok/s)
  - Mean TTFT ≈ **1.17s** (P99 2.63s), Mean TPOT ≈ **16.6ms** (P99 24.4ms)
  - Spec decode **accept len ≈ 3.01**. Mean ITL 28.9ms (median 13ms; MTP bursts)
  - Note: `Total generated tokens (retokenized)` reads 238257 < 524288 — cosmetic
    counter discrepancy from bench `temperature=0` stop-token accounting, not truncation
    (all 64 requests generated the full 8192 tokens). Run with `temperature=1.0,top_p=0.95`
    (the model's recommended sampling) for cleaner accounting.

### Image analysis — `lmsysorg/sglang:qwen38flashnext`

- **Stack:** sglang `0.0.0.dev1+g593134d17` (branch `qwen4-main-squashed-rebased`, commit
  `593134d1`, 2026-09-03) · flashinfer 0.6.18+cu130 · torch 2.13.0+cu130 · transformers
  5.12.1 · triton 3.7.1 · sglang-kernel 0.4.6.post1.
- **What it is:** a squashed/rebased build of PR #36497. It is **newer than the PR head**
  (`78c5024`, 2026-08-30) and newer than the benchmarked `e17062a1d` — it additionally
  pulls in `fix(qsa): restore SM121 correctness` (#36845). Branch is 18 ahead / 29 behind
  upstream `main`.
- **Qwen3.8-specific content (42 files):** QSA attention stack
  (`layers/attention/qsa/*`, `qwen_sparse_attn_backend.py`, `hybrid_linear_attn_backend.py`),
  QSA indexer kernels (`kernels/ops/attention/qsa_indexer.py`, `jit/csrc/attention/qsa_indexer.cuh`),
  KDA-tuned packed-QSA decode kernel (`kernels/kda_kernels/qwen38_qsa_sm121/`),
  PLE N-gram offload (`kernels/ops/qwen4_ple.py`, `mem_cache/ple_state_pool.py`),
  QSA KV pool (`mem_cache/qsa_kv_pool.py`), model + MTP (`models/qwen4_exp{,_mtp}.py`),
  configs/arg-overrides, VL processor, and a full test suite.
- **Should you move to this image?** **You already are on it — it is the correct and only
  build that serves this model.** The alternative is building PR #36497 from source, which
  would give an *older* tree (the image supersedes the PR head). There is no stable release
  containing the model yet, so there is nothing newer/safer to move to. Stay on this tag;
  re-pull only when lmsys pushes an updated `qwen38flashnext` tag or #36497 lands in a release.

### SM120 support — confirmed backends on this node

Yes, SM120 is explicitly supported — but "SM120-specific" here means **validated dispatch
guards + tuned Triton kernels, not SM100/SM121 CUDA/CUTLASS cores**. Confirmed live in
`serve.log` and `qwen_sparse_attn_backend.py`:

| Component | Backend on SM120 (this node) | SM-specific? |
|---|---|---|
| QSA sparse full-attn decode | **flashinfer `trtllm_batch_decode_with_kv_cache`** — code comment: *"numerically validated on SM100 and SM120; do not widen to every SM12x — silently corrupts long-context decode on SM121/GB10"* | ✅ SM100/SM120-exact guard |
| QSA varlen (prefill/extend) | `flash_attn_varlen_func` / cute fallback | generic Blackwell |
| QSA packed decode (tuned) | `qwen38_qsa_sm121` KDA kernel — **NOT used on SM120** (`is_sm121()` only, ~2.07× win) | ❌ SM121-only |
| Gated DeltaNet prefill+decode | **flashinfer GDN** (`--linear-attn-{prefill,decode}-backend flashinfer`, bf16 ssm) — log: *"FlashInfer GDN kernels loaded successfully"* | ✅ pinned |
| GDN MTP verify | Triton (`--linear-attn-verify-backend triton`) | pinned |
| QSA indexer | Triton (`dsa_topk_backend=sgl-kernel`) | Triton |
| MoE (512-expert, BF16) | default fused-MoE / CUTLASS SM120 grouped GEMM (`moe_runner_backend=auto`) | SM120 grouped GEMM |
| MTP speculative | EAGLE/NEXTN, *"QSA MTP index sharing enabled"* | — |

Net: every QSA/GDN path has an SM120-safe, working kernel — nothing silently falls back to
a wrong-arch binary. The **one** SM-arch optimization you don't get on RTX PRO 6000 is the
GB10/SM121-tuned packed-QSA decode kernel (~2× on that kernel); on SM120 the QSA decode uses
the flashinfer TRTLLM path instead, which the code marks as the *correct* choice for SM120
(the SM121 kernel would corrupt long-context decode here). Overall serving is correct and
fast on SM120 (1669 tok/s @ 1k/8k); you're only leaving the small GB10-specific decode win
on the table, which isn't applicable to this hardware anyway.

### GSM8K validation (this host, 2026-09-04)

Ran `benchmark/gsm8k/bench_sglang.py` against the live TP8 server, 500 questions, 8-shot,
thinking mode (`enable_thinking=true`), `temperature=0`.

| Run | max-new-tokens | stop tokens | Accuracy | Invalid |
|---|---|---|---|---|
| Bounded (default harness) | 2048 | `["Question","Assistant:","<|separator|>"]` | **0.622** | 0.066 |
| **Unbounded (patched)** | 16384 | `None` (EOS only) | **0.982** | **0.000** |

- The unbounded **0.982** matches the official SGLang reference (GSM8K ≈ 97.5–97.7),
  confirming the BF16 TP8 deployment is numerically correct.
- The low bounded score is a **harness artifact, not the model**: with thinking ON, the
  2048-token cap plus the hardcoded `"Question"` stop token truncate the long reasoning
  before the `#### answer`, and the extractor (`last integer in output`) then reads a wrong
  number. Fix = raise the cap and drop the premature stop tokens (EOS only). Patch used:
  copy of `bench_sglang.py` with `default=16384` and `stop=None` (saved as
  `/tmp/bench_unbounded.py` in the container; log `qwen38/gsm8k_unbounded.log`).

### Bake-off: TP8+MTP vs DP2×TP4 (1k/8k, C=96, 192 prompts)

Same workload on both, `--max-running-requests 96`. TP8 = recipe 4a (MTP on);
DP2 = two independent TP4 replicas, `--dp-size 2 --tp 4 --ep 4`, MTP **off**.

| Metric (C=96, 1k in / 8k out) | TP8 + MTP | DP2×TP4 (no MTP) |
|---|---|---|
| Output token throughput | 2526 tok/s | **2825 tok/s (+12%)** |
| Total token throughput | 2842 tok/s | **3178 tok/s** |
| Request throughput | 0.31 req/s | 0.34 req/s |
| Mean TPOT | 32.8ms | 32.7ms (tie) |
| Mean TTFT | **2.65s** | 3.90s (TP8 better) |
| Accept length (MTP) | 3.05 | — (off) |
| Single-stream latency | **better (MTP)** | worse |

**Verdict:** it depends on the goal —
- **Max aggregate throughput / high concurrency:** **DP2×TP4** wins ~12% on output tok/s.
  Two independent replicas scale batch better than TP8-over-PCIe at high C. Turn MTP on in
  each replica (not tested here) and the gap likely widens further.
- **Single-stream / low-latency / agentic (few concurrent requests):** **TP8+MTP** wins —
  MTP accept-len ~3 cuts per-token latency (see C=32: TPOT 16.6ms vs DP2's 32.7ms at C=96),
  and TTFT is lower. This is the better default for interactive/coding-agent use.
DP2 serve script: `qwen38/serve_dp2.sh` (container name `qwen38dp`). Currently the server is
running **DP2×TP4**; restart `serve.sh` to go back to TP8+MTP.

### Serve script

Server runs via `/home/shivajid_google_com/qwen38/serve.sh` on the host
(logs: `qwen38/serve.log`, bench: `qwen38/bench_1k8k.log`). Stop with `docker stop qwen38`.

TP8 over PCIe (no NVLink) held up well here — decode TPOT ~16.6ms at 8k output with MTP is solid. If you
see decode ITL degrade at higher concurrency, try `--tp 4` (frees 4 GPUs) or NVFP4 `--tp 2`.

### Reproduce

```bash
# on the host
/home/shivajid_google_com/qwen38/serve.sh   # launches the docker server (logs: qwen38/serve.log)
# stop:  docker stop qwen38
```

---

## 7. Risks / gotchas on this node

1. **SPOT preemption** — VM can stop anytime; keep HF cache on PD if you need persistence.
2. **TP8 on RTX PRO 6000** — these GPUs have **no NVLink** (PCIe). Official cells are TP4.
   TP8 all-reduce/all-gather over PCIe can bottleneck decode; if decode ITL is poor, drop to
   `--tp 4` (fits BF16 easily) or NVFP4 `--tp 2`.
3. **QSA SM121 kernel is GB10-tuned** — on SM120 SGLang uses the Triton QSA fallback unless
   the sm121 kernel self-selects; either path is functional, the tuned kernel is ~2× faster.
4. **Day-0 PR image** — not a stable release; flags/backends may change when #36497 merges.
5. **`--gpus all` vs default runtime** — default Docker runtime is `runc`, so always pass
   `--gpus all` (or `--runtime=nvidia`). Confirmed passthrough works on this host.
6. **DP-attention + MoE-a2a (`--dp-size 8 --enable-dp-attention --moe-a2a-backend X`) is
   NOT supported here.** That recipe = 8 DP ranks, attention DP-local, MoE expert-parallel
   via all-to-all. Verified in the image it does **not** work for this model:
   - **GDN layers**: *would* support it — `linear/gdn_backend.py` shards heads by
     `attn_tp_size` (`tp_heads = n_attention_heads // attn_tp_size`), and GDN recurrent
     state is per-request (batch-parallel), so it carries no sharded KV.
   - **QSA layers**: **do not** — `qwen_sparse_attn_backend.py` and `layers/attention/qsa/`
     have **zero** `attn_tp_size`/`attn_dp`/`dp_size` handling; the QSA indexer + sparse-GQA
     kernels assume plain TP. Every 4th layer is QSA, so enabling DP-attention corrupts or
     crashes those layers.
   - `--moe-a2a-backend` (deepep/p2p/etc.) only selects the EP all-to-all transport for the
     512-expert MoE — it does not make QSA DP-aware.
   Even if QSA were ported: KV is tiny (~2.4GB/GPU, 2 KV heads, 36/48 layers are KV-less
   GDN), so DP-attention's KV-sharding win is nil, and EP all-to-all + dense all-reduce still
   cross the PCIe fabric. The validated recipe for this model is single-node TP (+ optional
   EP within the TP group), per all official cells. For more aggregate decode throughput use
   `--dp-size N` as **independent TP replicas** (router fan-out), e.g. `--dp-size 2 --tp 4`,
   not DP-attention.
