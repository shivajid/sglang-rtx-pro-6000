"""NVFP4 kernel checks: exact decode reference, error vs BF16, timings vs FP8Linear / BF16.

python tests/test_nvfp4.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))

import torch  # noqa: E402

import nvfp4_linear as nv  # noqa: E402
from fp8_linear import FP8Linear  # noqa: E402

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def decode(q: torch.Tensor, s: torch.Tensor, amax_buf: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """Independent torch decode of (packed e2m1, swizzled e4m3 scales, tensor amax) -> fp32 [M, K]."""
    q = q.cpu()
    codes = torch.stack([q & 0xF, q >> 4], dim=-1).reshape(M, K).long()
    vals = E2M1[codes & 7] * torch.where(codes & 8 > 0, -1.0, 1.0)
    r = torch.arange(M)[:, None]
    c = torch.arange(K // 16)[None, :]
    off = ((r // 128) * (K // 64) + c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + (c % 4)
    sc = s.cpu().view(torch.float8_e4m3fn).float()[off]
    g = nv.G_MAX / nv.amax_value(amax_buf).item()
    return vals * sc.repeat_interleave(16, dim=1) / g


def bench(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def main() -> None:
    torch.manual_seed(0)
    dev = "cuda"
    # 1) exact decode check (layout, nibble order, scales) on odd M
    for M, K, N in [(1, 5120, 1024), (130, 5120, 2048), (1000, 17408, 1024)]:
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        x[:, ::97] *= 30  # outlier channels
        lin = torch.nn.Linear(K, N, bias=False, device=dev, dtype=torch.bfloat16)
        torch.nn.init.normal_(lin.weight, std=0.02)
        ref_bf16 = lin(x)
        layer = nv.NVFP4Linear(lin)
        y = layer(x)
        amax = nv.amax_(x)
        q, s = nv.quant_nvfp4(x, amax)
        xd = decode(q, s, amax, M, K)
        wamax = torch.tensor([layer.w_amax], dtype=torch.float32).view(torch.int32)
        wd = decode(layer.wq, layer.ws, wamax, N, K)
        y_ref = xd @ wd.t()
        print(f"M={M} K={K} N={N}: vs exact-decode rel {rel(y.cpu(), y_ref):.5f} | quant err x {rel(xd, x.cpu()):.4f} "
              f"w {rel(wd, lin.weight.cpu()):.4f} | vs bf16 {rel(y, ref_bf16):.4f}", flush=True)

    # 2) fused MLP path vs reference silu(gate)*up -> down
    M, K, I = 2048, 5120, 17408
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    gu = torch.nn.Linear(K, 2 * I, bias=False, device=dev, dtype=torch.bfloat16)
    dn = torch.nn.Linear(I, K, bias=False, device=dev, dtype=torch.bfloat16)
    torch.nn.init.normal_(gu.weight, std=0.02)
    torch.nn.init.normal_(dn.weight, std=0.02)
    g, u = gu(x).chunk(2, dim=-1)
    ref = dn(torch.nn.functional.silu(g) * u)

    class M_:
        pass

    m = M_()
    m.gate_up_proj, m.down_proj = nv.NVFP4Linear(gu), nv.NVFP4Linear(dn)
    y = nv._mlp_nvfp4_forward(m, x)
    print(f"mlp nvfp4 vs bf16 rel {rel(y, ref):.4f}", flush=True)

    # 3) timings at serving shapes
    for M, K, N in [(12288, 5120, 34816), (12288, 17408, 5120), (1536, 5120, 34816)]:
        lin = torch.nn.Linear(K, N, bias=False, device=dev, dtype=torch.bfloat16)
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        t_bf = bench(lambda: lin(x))
        f8 = FP8Linear(lin)
        t_f8 = bench(lambda: f8(x))
        f4 = nv.NVFP4Linear(lin)
        t_f4 = bench(lambda: f4(x))
        amax = nv.amax_(x)
        t_q = bench(lambda: nv.quant_nvfp4(x, amax))
        t_a = bench(lambda: nv.amax_(x))
        gb = M * K * 2.5 / 1e9
        print(f"M={M} K={K} N={N}: bf16 {t_bf:.2f} ms | fp8 {t_f8:.2f} | nvfp4 {t_f4:.2f} ({t_f8 / t_f4:.2f}x fp8) "
              f"| quant {t_q:.3f} ms ({gb / t_q:.0f} GB/s) amax {t_a:.3f} ms", flush=True)


if __name__ == "__main__":
    main()
