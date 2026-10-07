"""FP8 (e4m3) linear layers for prefill on SM120 without torch.compile.

torchao's eager Float8DynamicActivationFloat8WeightConfig spends as long quantizing activations
(abs, amax, fp32 divide, clamp and cast run as separate kernels, ~800 ms per 12K-token batch on
the G4) as the FP8 GEMMs save. Here the per-token activation quantization is one Triton kernel
and the GEMM is the same rowwise-scaled CUTLASS kernel (torch._scaled_mm):

    y = (q(x) * s_x[:, None]) @ (q(W) * s_w[:, None]).T     s_x per token, s_w per output channel

Same numerics as torchao PerRow: scale = amax / 448, values clamped to +-448, RTNE cast.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0


@triton.jit
def _quant_rows_kernel(X, Y, S, stride_x, stride_y, K, BLOCK: tl.constexpr, SINGLE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    x_ptr = X + row * stride_x
    y_ptr = Y + row * stride_y
    if SINGLE:
        cols = tl.arange(0, BLOCK)
        m = cols < K
        x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
        scale = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-12) / 448.0
        y = tl.clamp(x / scale, -448.0, 448.0)
        tl.store(y_ptr + cols, y.to(Y.dtype.element_ty), mask=m)
    else:
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for off in range(0, K, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            x = tl.load(x_ptr + cols, mask=cols < K, other=0.0).to(tl.float32)
            acc = tl.maximum(acc, tl.abs(x))
        scale = tl.maximum(tl.max(acc, axis=0), 1e-12) / 448.0
        for off in range(0, K, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            m = cols < K
            x = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
            y = tl.clamp(x / scale, -448.0, 448.0)
            tl.store(y_ptr + cols, y.to(Y.dtype.element_ty), mask=m)
    tl.store(S + row, scale)


def quant_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row dynamic FP8 quantization. x [M, K] (last dim contiguous) -> (fp8 [M, K], fp32 scale [M, 1])."""
    if x.stride(-1) != 1:
        x = x.contiguous()
    M, K = x.shape
    y = torch.empty((M, K), dtype=FP8, device=x.device)
    s = torch.empty((M, 1), dtype=torch.float32, device=x.device)
    if M == 0:
        return y, s
    if K <= 8192:
        BLOCK = triton.next_power_of_2(K)
        _quant_rows_kernel[(M,)](x, y, s, x.stride(0), y.stride(0), K, BLOCK=BLOCK, SINGLE=True, num_warps=8 if BLOCK >= 4096 else 4)
    else:
        _quant_rows_kernel[(M,)](x, y, s, x.stride(0), y.stride(0), K, BLOCK=2048, SINGLE=False, num_warps=8)
    return y, s


class FP8Linear(torch.nn.Module):
    """Drop-in for nn.Linear: per-channel FP8 weight, per-token dynamic FP8 activation."""

    def __init__(self, lin: torch.nn.Linear):
        super().__init__()
        self.in_features, self.out_features = lin.in_features, lin.out_features
        w = lin.weight.data
        wq = torch.empty(w.shape, dtype=FP8, device=w.device)
        ws = torch.empty((w.shape[0], 1), dtype=torch.float32, device=w.device)
        step = 4096  # bound the fp32 temporary
        for i in range(0, w.shape[0], step):
            blk = w[i : i + step].float()
            sc = blk.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / FP8_MAX
            wq[i : i + step] = (blk / sc).clamp(-FP8_MAX, FP8_MAX).to(FP8)
            ws[i : i + step] = sc
        self.register_buffer("wq", wq, persistent=False)  # [N, K] row-major; .t() is the column-major mat2
        self.register_buffer("w_scale", ws.reshape(1, -1).contiguous(), persistent=False)  # [1, N]
        self.bias = lin.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        xq, xs = quant_rows(x.reshape(-1, shp[-1]))
        y = torch._scaled_mm(xq, self.wq.t(), scale_a=xs, scale_b=self.w_scale, bias=self.bias, out_dtype=x.dtype)
        return y.reshape(*shp[:-1], self.out_features)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, fp8_e4m3 rowwise"


def apply_fp8_fast(model: torch.nn.Module, min_features: int = 1024, prefix: str = "layers.", skip: str | None = None) -> int:
    """Replace decoder nn.Linear layers (both dims >= min_features) with FP8Linear in place.
    skip (or env FP8_SKIP): regex on the module name for layers to keep in BF16, e.g. r"^layers\\.(0|63)\\."."""
    import os
    import re

    skip = os.environ.get("FP8_SKIP", "") if skip is None else skip
    pat = re.compile(skip) if skip else None
    names = [
        name for name, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear) and name.startswith(prefix)
        and mod.in_features >= min_features and mod.out_features >= min_features
        and not (pat and pat.search(name))
    ]
    for name in names:  # one at a time so each BF16 weight is freed right after conversion
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        setattr(parent, attr, FP8Linear(getattr(parent, attr)))
    torch.cuda.empty_cache()
    return len(names)
