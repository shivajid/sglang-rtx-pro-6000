"""Feasibility probe: NVFP4 (float4_e2m1fn_x2 + e4m3 1x16 block scales) GEMM on this GPU.

1) raw torch._scaled_mm throughput at Clef GEMM shapes vs BF16 and FP8 rowwise
2) torchao NVFP4DynamicActivationNVFP4WeightConfig on one nn.Linear: works? error vs BF16? end-to-end time
"""

from __future__ import annotations

import time
import traceback

import torch


def bench(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


def ceil(a, b):
    return (a + b - 1) // b * b


def main() -> None:
    dev = "cuda"
    print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability())
    shapes = [(12288, 5120, 17408 * 2), (12288, 17408, 5120), (12288, 5120, 12288), (4096, 5120, 34816), (1536, 5120, 34816)]
    for M, K, N in shapes:
        flops = 2 * M * K * N
        a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
        t_bf16 = bench(lambda: a @ w.t())
        a8 = a.to(torch.float8_e4m3fn)
        w8 = w.to(torch.float8_e4m3fn)
        sa = torch.ones(M, 1, device=dev)
        sb = torch.ones(1, N, device=dev)
        t_fp8 = bench(lambda: torch._scaled_mm(a8, w8.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16))
        line = f"M={M} K={K} N={N}: bf16 {flops / t_bf16 / 1e12:.0f} TF/s, fp8 {flops / t_fp8 / 1e12:.0f} TF/s"
        try:
            a4 = torch.randint(0, 255, (M, K // 2), device=dev, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
            w4 = torch.randint(0, 255, (N, K // 2), device=dev, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
            s_a = torch.full((ceil(M, 128) * ceil(K // 16, 4),), 1.0, device=dev).to(torch.float8_e4m3fn)
            s_b = torch.full((ceil(N, 128) * ceil(K // 16, 4),), 1.0, device=dev).to(torch.float8_e4m3fn)
            t_fp4 = bench(lambda: torch._scaled_mm(a4, w4.t(), scale_a=s_a, scale_b=s_b, out_dtype=torch.bfloat16))
            line += f", nvfp4 {flops / t_fp4 / 1e12:.0f} TF/s ({t_fp8 / t_fp4:.2f}x fp8)"
        except Exception as exc:  # noqa: BLE001
            line += f", nvfp4 FAILED: {type(exc).__name__}: {str(exc)[:300]}"
        print(line, flush=True)

    # torchao end-to-end on one Linear (includes activation quantization)
    try:
        from torchao.prototype.mx_formats.inference_workflow import NVFP4DynamicActivationNVFP4WeightConfig
        from torchao.quantization import quantize_

        for use_triton in (True, False):
            M, K, N = 12288, 5120, 34816
            lin = torch.nn.Linear(K, N, bias=False, device=dev, dtype=torch.bfloat16)
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            ref = lin(x)
            t_ref = bench(lambda: lin(x))
            try:
                quantize_(lin, NVFP4DynamicActivationNVFP4WeightConfig(use_triton_kernel=use_triton, use_dynamic_per_tensor_scale=True))
                with torch.inference_mode():
                    y = lin(x)
                    t_q = bench(lambda: lin(x))
                rel = ((y.float() - ref.float()).norm() / ref.float().norm()).item()
                print(f"torchao nvfp4 use_triton={use_triton}: rel_err {rel:.4f}, linear {t_ref * 1e3:.2f} ms bf16 -> {t_q * 1e3:.2f} ms", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"torchao nvfp4 use_triton={use_triton} FAILED: {type(exc).__name__}: {str(exc)[:400]}", flush=True)
    except Exception:  # noqa: BLE001
        traceback.print_exc()


if __name__ == "__main__":
    main()
