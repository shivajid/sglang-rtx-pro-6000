"""Warm the flashinfer SM120 fused-MoE JIT cache at image build time (compilation only,
no GPU required). Also asserts that the SiTU activation is wired up:
  * Python-side (flashinfer.tllm_enums.ActivationType.Situ == 10)
  * C++-side (our patched data/csrc headers contain SituAdaptor / ActivationType::Situ)

The kernel build is best-effort: if the ninja/nvcc invocation succeeds, the SM120
cubins are cached into FLASHINFER_CACHE_DIR so the cluster needs no runtime JIT.
If it fails in this GPU-less environment we log it and exit 0 so the image still
builds; the kernels then compile lazily on first cluster call.
"""
import os
import sys
import traceback

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
# GPU-less build: populate flashinfer's compilation_context TARGET_CUDA_ARCHS so
# get_cutlass_fused_moe_module(supported_major_versions=[12]) has an arch to target.
# "12.0f" -> compute_120f (the documented SM120 suffix in compilation_context).
os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", "12.0f")
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ["PATH"] = "/usr/local/cuda/bin:/usr/local/bin:" + os.environ.get("PATH", "")

FI_BASE = "/usr/local/lib/python3.12/dist-packages/flashinfer"
COMMON = FI_BASE + "/data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/common.h"
CUH = FI_BASE + "/data/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh"


def check(tag, ok):
    print(f"[warm] {tag}: {'OK' if ok else 'FAIL'}", flush=True)
    return ok


hard_fail = False

# 1) Python-side enum must expose Situ == 10 (this ships with the image, no build needed).
try:
    from flashinfer.tllm_enums import ActivationType, is_gated_activation
    hard_fail |= not check("python ActivationType has Situ", hasattr(ActivationType, "Situ"))
    hard_fail |= not check("python ActivationType.Situ == 10", int(ActivationType.Situ) == 10)
    hard_fail |= not check("python Situ is gated", bool(is_gated_activation(ActivationType.Situ)))
except Exception:
    hard_fail = True
    print("[warm] python ActivationType import FAILED", flush=True)
    traceback.print_exc()

# 2) C++ patch must be present in the data/csrc sources that the JIT will compile.
try:
    with open(COMMON) as f:
        common = f.read()
    with open(CUH) as f:
        cuh = f.read()
    hard_fail |= not check("common.h has Situ enumerator", "Situ," in common)
    hard_fail |= not check("cuh has SituAdaptor", "struct SituAdaptor" in cuh)
    hard_fail |= not check("cuh gated branch uses SituAdaptor",
                           "ActivationType::Situ" in cuh and "SituAdaptor>" in cuh)
except Exception:
    hard_fail = True
    print("[warm] csrc read FAILED", flush=True)
    traceback.print_exc()

if hard_fail:
    print("[warm] FATAL: SiTU patch / python enum verification failed", flush=True)
    sys.exit(1)

# 3) Best-effort: actually build the SM120 fused-MoE module so the cache is warm.
try:
    # get_cutlass_fused_moe_module lives in flashinfer.fused_moe.core, not on the
    # flashinfer.fused_moe package namespace; try core first, then fall back.
    builder = None
    try:
        from flashinfer.fused_moe.core import get_cutlass_fused_moe_module as builder
    except Exception:
        import flashinfer.fused_moe as moe
        builder = getattr(moe, "get_fused_moe_module", None) or \
            getattr(moe, "get_cutlass_fused_moe_module", None)
    if builder is None:
        raise RuntimeError("no fused-moe module builder found in flashinfer.fused_moe[.core]")
    try:
        builder(backend="120")          # get_cutlass_fused_moe_module signature
    except TypeError:
        builder()                       # get_fused_moe_module may take no kwargs
    print("[warm] fused-MoE SM120 module built & cached", flush=True)
except Exception:
    print("[warm] WARNING: SM120 fused-MoE build failed in GPU-less build env;", flush=True)
    print("[warm]          kernels will compile on first cluster call instead.", flush=True)
    traceback.print_exc()

# 4) Best-effort: also warm the mxfp8/fp8 quantization module. Option B calls
#    mxfp8_quantize() at runtime, which JIT-builds the fp8-quant module on first use.
try:
    import flashinfer
    warmed = False
    for modname in ("flashinfer.fp8_quantization", "flashinfer.fp4_quantization",
                    "flashinfer.quantization"):
        try:
            mod = __import__(modname, fromlist=["*"])
            fn = getattr(mod, "get_fp8_quantization_module", None) or \
                getattr(mod, "get_module", None)
            if callable(fn):
                try:
                    fn(backend="120")
                except TypeError:
                    fn()
                warmed = True
                print(f"[warm] {modname} SM120 module built & cached", flush=True)
                break
        except Exception:
            continue
    if not warmed:
        print("[warm] WARNING: fp8/mxfp8 quant module not warmed (no builder found); "
              "it will compile on first mxfp8_quantize() call on the cluster.", flush=True)
except Exception:
    print("[warm] WARNING: fp8/mxfp8 quant warm step errored; continuing.", flush=True)
    traceback.print_exc()

sys.exit(0)
