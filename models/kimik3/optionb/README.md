# Option B: SM120 `flashinfer_mxfp4` for Kimi-K3 — Changeset & Build

**Goal:** Opt-in MXFP8-activation × MXFP4-weight path so the native SM120 FP8×FP4
blockscaled grouped GEMM runs with the fused **SiTU** (SoftCap-GLU) epilogue on
GCP G4 (RTX PRO 6000 Blackwell, CC 12.0). **Default OFF** — bf16 (marlin/W4A16)
behavior is byte-identical unless explicitly enabled.

- **Shipped image:** `gcr.io/northam-ce-mlai-tpu/sglang-k3-mxfp8:latest`
- **Base image:** `lmsysorg/sglang:nightly-dev-cu13-20260816-4a6dc267`
- **Validation:** 8/8 unit PASS; real-mode nRMSE ~5%, pearson ≥ 0.9985;
  bs=1024 ≈ 241 TFLOP/s (~2× bf16).
- **Production (ctx=131072, TP=8, PP=4):** 1k_8k C=64 ≈ 745 tok/s (**+55%** vs
  480.86 baseline); ITL +2–8% at low C (converges to +2% at C=32). MXFP8-quant tax
  on decode; FP4 GEMM win at high concurrency.

**Enable (opt-in):** env `SGLANG_SITU_MXFP8_ACT=1` **or**
`MoeRunnerConfig.situ_mxfp8_act=True`.

---

## 1. Patch files (PR-ready, in this directory)

| File | Repo / base | Apply |
|---|---|---|
| `sglang_optionb_sm120.patch` | sglang @ `4a6dc26` | `git apply` (reverse-check passed) |
| `flashinfer_situ_sm120.patch` | flashinfer **0.6.18** (`csrc/` root) | `git apply` (verified identical to working tree) |

> These are convenience copies of changes that live in the two repos' working
> trees. For PRs, prefer committing the changes directly from each repo so you
> get proper commit metadata; the `.patch` files are for review/repro.

### Apply to branches

```bash
# sglang
git checkout -b optionb-sm120-mxfp8 4a6dc26
git apply KimiK3/sglang_optionb_sm120.patch
git commit -am "sglang: opt-in MXFP8-act x MXFP4-wt SiTU MoE for SM120 (Kimi-K3)"

# flashinfer (from pristine 0.6.18)
git checkout -b situ-sm120-epilogue e77a4a0
git apply KimiK3/flashinfer_situ_sm120.patch
git commit -am "flashinfer: re-anchor SiTU (SoftCap-GLU) epilogue for SM120 cutlass MoE"
```

---

## 2. sglang changes (8 modified + 1 new test)

All under `python/sglang/srt/`.

### Group 1 — SM120 kernel-guard fixes (3 files)
SM120 (major 12) was mis-matched by `major >= 10` guards that enable SM100-only
instructions (tcgen05/TMEM/cp.async.bulk). Re-anchored to gate on SM100 exactly →
SM120 falls back to Triton.

- `layers/attention/linear/kernels/gdn_cutedsl.py` — SM100-exact gate (GDN)
- `layers/attention/linear/kernels/kda_cutedsl.py` — SM100-exact gate (KDA)
- `layers/attn_residual.py` — `_FAST_SUPPORTED = is_sm100_supported()` (was
  `major >= 10`); SM120 → Triton pipeline

### Group 2 — Option B opt-in plumbing (5 files)
- `environ.py` — new `SGLANG_SITU_MXFP8_ACT = EnvBool(False)`
- `layers/moe/moe_runner/base.py` — new `MoeRunnerConfig.situ_mxfp8_act: bool = False`
- `layers/quantization/mxfp4.py` — **core change (~128 lines).** When
  `situ_mxfp8_act` and activation=="situ": quantize activations to MXFP8 and route
  to the FP8×FP4 blockscaled grouped GEMM with fused SiTU epilogue.
- `layers/moe/moe_runner/flashinfer_cutlass.py` — pass `situ_mxfp8_act` + SiTU
  params (`situ_beta=4.0`→`swiglu_alpha`, `situ_linear_beta=25.0`→`swiglu_beta`).
- `layers/moe/moe_runner/flashinfer_trtllm.py` — register `"situ": ActivationType.Situ`.

### New test
- `test/registered/unit/layers/test_sm100_kernel_guards.py` — guards against SM120
  mis-enabling SM100 kernels.

---

## 3. flashinfer changes (SiTU epilogue, 3 C++ files)

Re-adds the **SiTU epilogue missing from OSS CUTLASS SM120 kernels**. Formula:
`gate = β·tanh(gate/β)·sigmoid(gate)`, `up = linear_β·tanh(up/linear_β)`,
`out = gate·up` (β=4.0, linear_β=25.0).

- `csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh` — `struct
  SituAdaptor` + gated-dispatch entries (`doGatedActivation`, `doActivation`).
- `csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/common.h` —
  `Situ` enumerator (ActivationType = 10; must keep ordinal to match flashinfer
  python).
- `csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/moe_gemm_kernels.h`
  — `isGatedActivation(... Situ)`.

---

## 4. ⚠️ Version mismatch (read before rebuilding)

The **shipped image patched flashinfer 0.6.17** (bundled in the base container),
but the canonical `flashinfer_situ_sm120.patch` targets **0.6.18**. Same logical
change; line offsets authored against 0.6.18's `csrc/`.

- If your org standardizes on **0.6.18**: PR + patch are consistent — nothing to do.
- The **running container still has 0.6.17-patched**. If you rebuild on a newer base
  that bundles 0.6.18, apply `flashinfer_situ_sm120.patch` directly (no rework).
- Do **not** re-derive the 0.6.17 offsets — just move the base image forward.

---

## 5. Build context — `image_optb/`

```
image_optb/
├── Dockerfile
├── warm_fused_moe.py          # build-time GPU-less JIT warm + SiTU assert
├── sglang/...                 # 8 sglang files (mirror of repo paths) + test
└── flashinfer/...             # 3 C++ files (mirror of bundled 0.6.17 data/csrc paths)
```

The Dockerfile:
1. **Part A** — `COPY` the 8 sglang files over `/sgl-workspace/sglang/...`.
2. **Part B** — `COPY` the 3 C++ files over the image's flashinfer
   (`FI=/usr/local/lib/python3.12/dist-packages/flashinfer`, at `data/csrc/...`).
3. **Warm** — runs `warm_fused_moe.py` (GPU-less) to pre-compile the SM120
   fused-MoE module into the JIT cache; **hard-fails** if `ActivationType.Situ != 10`
   or the C++ SiTU patch is absent.
4. **De-shadow** — `rm -rf .../flashinfer_jit_cache/jit_cache/fused_moe_120` so the
   stale unpatched AOT artifact doesn't shadow the patched module (which would
   silently disable SiTU).

> Note: the Dockerfile does **not** COPY the new unit test into the image (tests
> aren't needed at runtime). It lives in the sglang PR / build context only.

---

## 6. Build & push the container

```bash
cd KimiK3/image_optb
# (if repos changed) refresh overlay files from your repos first:
#   cp <sglang-repo>/python/sglang/srt/<each of 8>   sglang/...
#   cp <flashinfer>/<each of 3 csrc files>            flashinfer/...
docker build -t gcr.io/northam-ce-mlai-tpu/sglang-k3-mxfp8:latest .
docker push gcr.io/northam-ce-mlai-tpu/sglang-k3-mxfp8:latest
```

The warm step self-verifies and exits non-zero on a missing SiTU patch, so a bad
build fails fast.

### Known gap (task #11)
The fp8/mxfp8-quant module is **not** warm in the image → it JIT-compiles once per
pod (~107s) on the first mxfp8 request, contaminating the first benchmark cell /
first user request. Optionally warm it in `warm_fused_moe.py` to eliminate the
cold-start.

---

## Attribution & License

The patched C++ headers under `image_optb/flashinfer/.../csrc/` are derivative
works of **NVIDIA TensorRT-LLM / CUTLASS** source (copyright NVIDIA CORPORATION),
redistributed under the **Apache License, Version 2.0**. Each file retains its
original NVIDIA copyright header. The modifications (the SiTU / SoftCap-GLU
activation epilogue) are documented in `flashinfer_situ_sm120.patch` and in the
file comments.

The sglang Python files under `image_optb/sglang/...` are derivative works of
**SGLang** (Apache License, Version 2.0), © the SGLang authors; modifications are
documented in `sglang_optionb_sm120.patch`.

These changes patch upstream **sglang** and **flashinfer**; the preferred
long-term home is a PR to those upstream repositories. This directory is a
reproducible snapshot of the patched sources + build context used to produce the
`sglang-k3-mxfp8` image.
