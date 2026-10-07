"""NVFP4 (W4A4) linear layers for prefill on SM120 (RTX PRO 6000 Blackwell).

Format (what cuBLASLt's block-scaled FP4 GEMM expects through torch._scaled_mm):
  * values: e2m1 (+-{0, .5, 1, 1.5, 2, 3, 4, 6}), two per byte, even element in the low nibble
  * block scales: e4m3, one per 16 consecutive K elements, stored in the 128x4 tile-swizzled layout
  * tensor scales: fp32, one per operand (g = 448 * 6 / amax), applied after the GEMM

    y = GEMM(q(x), q(W)) * amax_x * amax_w / (448 * 6)^2

Activations use a dynamic per-tensor scale: one atomic-amax pass, then one fused Triton pass
(block amax, e4m3 scale, e2m1 rounding, nibble packing, swizzled scale store). The raw GEMM output
can be returned with its scale (``forward_raw``) so the consumer (the fused SiLU*up kernel) applies it
for free; that kernel also emits amax(h) for the down projection, so h is read only once.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

FP4X2 = torch.float4_e2m1fn_x2
E4M3 = torch.float8_e4m3fn
G_MAX = 448.0 * 6.0  # largest e4m3 block scale x largest e2m1 value


@triton.jit
def _amax_kernel(X, OUT, N, BLOCK: tl.constexpr):
    """OUT (int32[1], zero-initialised) <- max(|X|) as float bits (valid ordering for x >= 0)."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs, mask=offs < N, other=0.0).to(tl.float32)
    tl.atomic_max(OUT, tl.max(tl.abs(x), axis=0).to(tl.int32, bitcast=True))


def amax_(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Global max |x| for a contiguous tensor, as an int32[1] buffer holding float bits (no host sync)."""
    if out is None:
        out = torch.zeros(1, dtype=torch.int32, device=x.device)
    n = x.numel()
    if n:
        BLOCK = 8192
        _amax_kernel[(triton.cdiv(n, BLOCK),)](x, out, n, BLOCK=BLOCK, num_warps=8)
    return out


@triton.jit
def _quant_nvfp4_kernel(X, Q, S, AMAX, stride_x, K, n_col_tiles, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    kb = tl.program_id(1)
    amax_t = tl.load(AMAX).to(tl.float32, bitcast=True)
    g = tl.where(amax_t > 0, 2688.0 / amax_t, 1.0)
    cols = kb * BLOCK_K + tl.arange(0, BLOCK_K)
    x = tl.load(X + row.to(tl.int64) * stride_x + cols).to(tl.float32)
    xb = tl.reshape(x, (BLOCK_K // 16, 16))
    s = tl.minimum(tl.max(tl.abs(xb), axis=1) * (g / 6.0), 448.0)
    s8 = s.to(tl.float8e4nv)
    s_deq = s8.to(tl.float32) / g
    inv = tl.where(s_deq > 0, 1.0 / s_deq, 0.0)
    r = tl.abs(xb) * inv[:, None]
    code = ((r > 0.25).to(tl.int32) + (r >= 0.75).to(tl.int32) + (r > 1.25).to(tl.int32) + (r >= 1.75).to(tl.int32)
            + (r > 2.5).to(tl.int32) + (r >= 3.5).to(tl.int32) + (r > 5.0).to(tl.int32))
    code = code + tl.where(xb < 0, 8, 0)
    lo, hi = tl.split(tl.reshape(code, (BLOCK_K // 2, 2)))
    packed = (lo | (hi << 4)).to(tl.uint8)
    tl.store(Q + row.to(tl.int64) * (K // 2) + kb * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2), packed)
    sc = kb * (BLOCK_K // 16) + tl.arange(0, BLOCK_K // 16)
    off = ((row // 128) * n_col_tiles + sc // 4) * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + (sc % 4)
    tl.store(S + off.to(tl.int64), s8.to(tl.uint8, bitcast=True))


def _block_k(K: int) -> int:
    for b in (1024, 512, 256, 128, 64):
        if K % b == 0:
            return b
    raise ValueError(f"K={K} must be a multiple of 64 for NVFP4")


def quant_nvfp4(x: torch.Tensor, amax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x [M, K] bf16 + amax buffer (from amax_) -> (packed e2m1 uint8 [M, K/2], swizzled e4m3 scales uint8)."""
    if x.stride(-1) != 1:
        x = x.contiguous()
    M, K = x.shape
    q = torch.empty((M, K // 2), dtype=torch.uint8, device=x.device)
    pad_m = triton.cdiv(M, 128) * 128
    s = torch.empty((pad_m * (K // 16),), dtype=torch.uint8, device=x.device)
    if M:
        BK = _block_k(K)
        _quant_nvfp4_kernel[(M, K // BK)](x, q, s, amax, x.stride(0), K, K // 64, BLOCK_K=BK, num_warps=4)
    return q, s


def amax_value(buf: torch.Tensor) -> torch.Tensor:
    """0-dim fp32 view of an amax_ buffer (0-dim so bf16 * it stays bf16)."""
    return buf.view(torch.float32)[0]


class NVFP4Linear(torch.nn.Module):
    """Drop-in for a bias-free nn.Linear: NVFP4 weight, dynamic NVFP4 activation."""

    def __init__(self, lin: torch.nn.Linear):
        super().__init__()
        if lin.bias is not None:
            raise ValueError("NVFP4Linear: bias not supported")
        self.in_features, self.out_features = lin.in_features, lin.out_features
        w = lin.weight.data
        if w.stride(-1) != 1:
            w = w.contiguous()
        w_amax = amax_(w)
        wq, ws = quant_nvfp4(w, w_amax)
        self.register_buffer("wq", wq, persistent=False)  # [N, K/2] packed e2m1
        self.register_buffer("ws", ws, persistent=False)  # swizzled e4m3 block scales
        self.w_amax = float(amax_value(w_amax))  # one host sync at conversion time
        self.inv_g2 = self.w_amax / (G_MAX * G_MAX)

    def forward_raw(self, x2: torch.Tensor, amax: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """x2 [M, K] -> (GEMM output before the tensor scale, 0-dim fp32 scale)."""
        if amax is None:
            if x2.stride(-1) != 1 or not x2.is_contiguous():
                x2 = x2.contiguous()
            amax = amax_(x2)
        q, s = quant_nvfp4(x2, amax)
        y = torch._scaled_mm(q.view(FP4X2), self.wq.view(FP4X2).t(), scale_a=s.view(E4M3), scale_b=self.ws.view(E4M3),
                             out_dtype=torch.bfloat16)
        return y, amax_value(amax) * self.inv_g2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        y, alpha = self.forward_raw(x.reshape(-1, shp[-1]))
        y.mul_(alpha)
        return y.reshape(*shp[:-1], self.out_features)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, nvfp4 (e2m1 + e4m3/16 + fp32 tensor)"


# --------------------------------------------------------------------------------------
# MLP: NVFP4 gate_up / down with the tensor scales folded into the fused SiLU*up kernel
# --------------------------------------------------------------------------------------


@triton.jit
def _silu_mul_scaled_kernel(X, Y, ALPHA, AMAX_OUT, stride_x, stride_y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = cols < N
    a = tl.load(ALPHA)
    g = tl.load(X + row * stride_x + cols, mask=m, other=0.0).to(tl.float32) * a
    u = tl.load(X + row * stride_x + N + cols, mask=m, other=0.0).to(tl.float32) * a
    y = (g * tl.sigmoid(g) * u).to(Y.dtype.element_ty)
    tl.store(Y + row * stride_y + cols, y, mask=m)
    tl.atomic_max(AMAX_OUT, tl.max(tl.abs(y.to(tl.float32)), axis=0).to(tl.int32, bitcast=True))


def silu_mul_scaled(x: torch.Tensor, alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x [M, 2N] raw gate|up GEMM output, alpha 0-dim fp32 -> (silu(a*g) * (a*u) [M, N], amax buffer of it)."""
    M, N2 = x.shape
    N = N2 // 2
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    amax = torch.zeros(1, dtype=torch.int32, device=x.device)
    if M:
        BLOCK = 1024
        a = alpha.reshape(1) if alpha.dim() == 0 else alpha
        _silu_mul_scaled_kernel[(M, triton.cdiv(N, BLOCK))](x, y, a, amax, x.stride(0), y.stride(0), N, BLOCK=BLOCK, num_warps=4)
    return y, amax


def _mlp_nvfp4_forward(self, x):
    shp = x.shape
    x2 = x.reshape(-1, shp[-1])
    gu, a1 = self.gate_up_proj.forward_raw(x2)
    h, h_amax = silu_mul_scaled(gu, a1)
    if isinstance(self.down_proj, NVFP4Linear):
        y, a2 = self.down_proj.forward_raw(h, amax=h_amax)
        y.mul_(a2)
    else:  # down_proj kept in FP8/BF16
        y = self.down_proj(h)
    return y.reshape(*shp[:-1], y.shape[-1])


def apply_nvfp4_mlp(model: torch.nn.Module, layers: str | None = None, parts: str | None = None) -> int:
    """Swap the fused MLPs' gate_up_proj / down_proj (fast_patches.patch_mlp) to NVFP4. Returns linears converted.
    layers (env NVFP4_LAYERS): optional regex on the module name (e.g. "layers.12.mlp") to restrict which MLPs.
    parts (env NVFP4_PARTS, default "gate_up,down"): which projections; the others stay nn.Linear (FP8 later)."""
    import os
    import re
    import types

    layers = os.environ.get("NVFP4_LAYERS", "") if layers is None else layers
    parts = os.environ.get("NVFP4_PARTS", "gate_up,down") if parts is None else parts
    want = set(parts.split(","))
    if "gate_up" not in want:
        raise ValueError("NVFP4_PARTS must include gate_up")
    pat = re.compile(layers) if layers else None
    n = 0
    for name, mod in model.named_modules():
        if type(mod).__name__ != "Qwen3_5MLP" or not hasattr(mod, "gate_up_proj"):
            continue
        if pat and not pat.search(name):
            continue
        mod.gate_up_proj = NVFP4Linear(mod.gate_up_proj)
        n += 1
        if "down" in want:
            mod.down_proj = NVFP4Linear(mod.down_proj)
            n += 1
        mod.forward = types.MethodType(_mlp_nvfp4_forward, mod)
        torch.cuda.empty_cache()
    return n
