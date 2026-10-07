"""Unit checks + micro-timings for the custom kernels (no model load needed).

python tests/test_kernels.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
from fast_patches import rmsnorm_1p, silu_mul  # noqa: E402
from fp8_linear import FP8, FP8_MAX, FP8Linear, quant_rows  # noqa: E402


def bench(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    torch.manual_seed(0)
    dev = "cuda"
    M = 12288
    for K in (5120, 6144, 17408):
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 3
        q, s = quant_rows(x)
        ref_s = x.float().abs().amax(1, keepdim=True).clamp(min=1e-12) / FP8_MAX
        ref_q = (x.float() / ref_s).clamp(-FP8_MAX, FP8_MAX).to(FP8)
        dq = (q.float() - ref_q.float()).abs().max().item()
        ds = (s - ref_s).abs().max().item()
        mism = (q.view(torch.uint8) != ref_q.view(torch.uint8)).float().mean().item()
        ms = bench(lambda: quant_rows(x))
        gbps = (M * K * 3) / (ms / 1000) / 1e9
        print(f"quant_rows K={K}: max|dq|={dq:.3g} max|ds|={ds:.3g} byte-mismatch={mism:.2e}  {ms:.3f} ms ({gbps:.0f} GB/s)")

    for K, N in ((5120, 16384), (5120, 34816), (17408, 5120), (6144, 5120), (5120, 1024)):
        lin = torch.nn.Linear(K, N, bias=False, device=dev, dtype=torch.bfloat16)
        torch.nn.init.normal_(lin.weight, std=K ** -0.5)
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        ref = lin(x)
        f8 = FP8Linear(lin)
        y = f8(x)
        rel = ((y.float() - ref.float()).norm() / ref.float().norm()).item()
        t_bf16 = bench(lambda: lin(x))
        t_fp8 = bench(lambda: f8(x))
        xq, xs = quant_rows(x)
        t_mm = bench(lambda: torch._scaled_mm(xq, f8.wq.t(), scale_a=xs, scale_b=f8.w_scale, out_dtype=torch.bfloat16))
        tf = 2 * M * K * N / 1e12
        print(f"FP8Linear {K}->{N}: rel err {rel:.4f}; bf16 {t_bf16:.2f} ms ({tf / t_bf16 * 1000:.0f} TF/s)  fp8 {t_fp8:.2f} ms "
              f"(gemm only {t_mm:.2f} ms, {tf / t_mm * 1000:.0f} TF/s)")

    x = torch.randn(M, 2 * 17408, device=dev, dtype=torch.bfloat16)
    g, u = x.chunk(2, dim=-1)
    ref = F.silu(g) * u
    y = silu_mul(x)
    print(f"silu_mul: max|d|={(y.float() - ref.float()).abs().max().item():.3g}  fused {bench(lambda: silu_mul(x)):.3f} ms  "
          f"eager {bench(lambda: F.silu(g) * u):.3f} ms")

    for D in (5120, 256):
        x = torch.randn(M * (1 if D == 5120 else 24), D, device=dev, dtype=torch.bfloat16)
        w = torch.randn(D, device=dev, dtype=torch.bfloat16) * 0.1
        xf = x.float()
        ref = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * (1.0 + w.float())).to(torch.bfloat16)
        y = rmsnorm_1p(x, w, 1e-6)
        print(f"rmsnorm D={D}: max|d|={(y.float() - ref.float()).abs().max().item():.3g}  fused {bench(lambda: rmsnorm_1p(x, w, 1e-6)):.3f} ms")


if __name__ == "__main__":
    main()
