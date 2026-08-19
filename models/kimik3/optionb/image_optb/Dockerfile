# Custom sglang image for serving Kimi-K3 (MXFP8) on GCP G4 (SM120 / RTX PRO 6000 Blackwell).
#
#   Part A: overlay 8 sglang "Option B" MXFP8-activation source files
#           (SGLANG_SITU_MXFP8_ACT / situ_mxfp8_act + SM120 is_sm100_supported() kernel guards).
#   Part B: re-anchor the C++ "SiTU" (SoftCap-GLU) activation epilogue into the image's bundled
#           flashinfer 0.6.17 (data/csrc) and pre-compile the SM120 fused-MoE module so no
#           runtime JIT is needed on the cluster.

FROM lmsysorg/sglang:nightly-dev-cu13-20260816-4a6dc267

# Compile for SM120 only; ensure nvcc / ninja are resolvable during the JIT warm step.
ENV TORCH_CUDA_ARCH_LIST="12.0"
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:/usr/local/bin:${PATH}"

# ---- Part A: sglang overlay (8 files), same relative paths under /sgl-workspace/sglang ----
COPY sglang/python/sglang/srt/environ.py \
     /sgl-workspace/sglang/python/sglang/srt/environ.py
COPY sglang/python/sglang/srt/layers/attention/linear/kernels/gdn_cutedsl.py \
     /sgl-workspace/sglang/python/sglang/srt/layers/attention/linear/kernels/gdn_cutedsl.py
COPY sglang/python/sglang/srt/layers/attention/linear/kernels/kda_cutedsl.py \
     /sgl-workspace/sglang/python/sglang/srt/layers/attention/linear/kernels/kda_cutedsl.py
COPY sglang/python/sglang/srt/layers/attn_residual.py \
     /sgl-workspace/sglang/python/sglang/srt/layers/attn_residual.py
COPY sglang/python/sglang/srt/layers/moe/moe_runner/base.py \
     /sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/base.py
COPY sglang/python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py \
     /sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py
COPY sglang/python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py \
     /sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py
COPY sglang/python/sglang/srt/layers/quantization/mxfp4.py \
     /sgl-workspace/sglang/python/sglang/srt/layers/quantization/mxfp4.py

# ---- Part B: SiTU patch into bundled flashinfer 0.6.17 (data/csrc) ----
ARG FI=/usr/local/lib/python3.12/dist-packages/flashinfer
COPY flashinfer/data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/common.h \
     ${FI}/data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/common.h
COPY flashinfer/data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/moe_gemm_kernels.h \
     ${FI}/data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/moe_gemm_kernels.h
COPY flashinfer/data/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh \
     ${FI}/data/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh

# ---- Warm the SM120 JIT cache (GPU-less, best-effort) + assert SiTU wiring ----
COPY warm_fused_moe.py /tmp/warm_fused_moe.py
RUN python3 /tmp/warm_fused_moe.py && rm -f /tmp/warm_fused_moe.py

# Final assertion (hard-fails the build if the Python enum lost Situ).
RUN python3 -c "from flashinfer.tllm_enums import ActivationType; \
    assert hasattr(ActivationType, 'Situ') and int(ActivationType.Situ) == 10, 'Situ enum missing'; \
    print('SiTU ActivationType.Situ=10 confirmed')"

# Remove the stale UNPATCHED AOT fused_moe_120 artifact. flashinfer's try_load() prefers the AOT
# .so when it loads (GPU cluster), which would shadow the warm-patched cached_ops/fused_moe_120
# module and silently disable SiTU. Deleting it forces fallback to the patched cached_ops module
# (kept fresh by ninja). Other AOT modules (xqa/attention) are SiTU-independent and stay.
RUN rm -rf /usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/jit_cache/fused_moe_120 \
    && echo "Removed stale AOT fused_moe_120 so the patched cached_ops module is used"
