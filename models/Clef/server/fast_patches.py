"""Inference-only fusions for the HF Qwen3.5 backbone used by Clef (prefill, no KV cache).

patch_gdn(text_model)
    * one GEMM for in_proj_qkv + in_proj_z (merged weight, originals removed), one for in_proj_b + in_proj_a
    * fla Triton causal_conv1d on the [B, T, C] layout (no transposes / cuDNN depthwise conv)
    * native GVA in fla chunk_gated_delta_rule (no repeat_interleave of q/k to 48 heads)
    * fla fused GDN gate
patch_rmsnorm(text_model)
    * single-pass Triton RMSNorm with Qwen3.5's zero-centred (1 + w) weight, fp32 math
patch_mlp(text_model)
    * one GEMM for gate_proj + up_proj, Triton SiLU(gate) * up in one pass
pin_autotune_keys()
    * makes fla's shape-dependent constexpr autotune keys constant (no per-shape recompile/autotune)
All keep the reference math (up to bf16 rounding order); parity is checked in tests/parity.py.
Apply before FP8 so the merged projections are quantized as single linears.
"""

from __future__ import annotations

import re
import types

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# --------------------------------------------------------------------------------------
# RMSNorm
# --------------------------------------------------------------------------------------


@triton.jit
def _rmsnorm_fwd(X, W, Y, stride_x, stride_y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x * rstd) * (1.0 + w)
    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


def rmsnorm_1p(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    shape = x.shape
    N = shape[-1]
    x2 = x.reshape(-1, N)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    y = torch.empty((x2.shape[0], N), dtype=x.dtype, device=x.device)
    if x2.shape[0] == 0:
        return y.reshape(shape)
    BLOCK = triton.next_power_of_2(N)
    num_warps = 8 if BLOCK >= 4096 else (4 if BLOCK >= 1024 else 2)
    _rmsnorm_fwd[(x2.shape[0],)](x2, weight, y, x2.stride(0), y.stride(0), N, eps, BLOCK=BLOCK, num_warps=num_warps)
    return y.reshape(shape)


def _rmsnorm_forward(self, x):
    return rmsnorm_1p(x, self.weight, self.eps)


def patch_rmsnorm(model: torch.nn.Module) -> int:
    n = 0
    for mod in model.modules():
        if type(mod).__name__ == "Qwen3_5RMSNorm":
            mod.forward = types.MethodType(_rmsnorm_forward, mod)
            n += 1
    return n


# --------------------------------------------------------------------------------------
# Gated DeltaNet prefill
# --------------------------------------------------------------------------------------


def _gdn_prefill_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    if cache_params is not None:
        raise RuntimeError("patched GDN layer is prefill-only (no cache)")
    from fla.modules.conv.causal_conv1d import causal_conv1d
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.ops.gated_delta_rule.gate import fused_gdn_gate

    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)
    B, T, _ = hidden_states.shape
    qkvz = self.in_proj_qkvz(hidden_states)  # [B, T, conv_dim + value_dim]
    qkv, z = qkvz.split([self.conv_dim, self.value_dim], dim=-1)
    b, a = self.in_proj_ba(hidden_states).split([self.num_v_heads, self.num_v_heads], dim=-1)
    qkv, _ = causal_conv1d(x=qkv, weight=self.conv1d.weight.squeeze(1), bias=self.conv1d.bias, activation=self.activation)
    q, k, v = qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
    q = q.reshape(B, T, -1, self.head_k_dim)
    k = k.reshape(B, T, -1, self.head_k_dim)
    v = v.reshape(B, T, -1, self.head_v_dim)
    beta = b.sigmoid()
    g = fused_gdn_gate(a.contiguous(), self.A_log, self.dt_bias)
    core, _ = chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True,
    )
    core = self.norm(core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
    return self.out_proj(core.reshape(B, T, -1))


def _linear_from(w: torch.Tensor) -> torch.nn.Linear:
    lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=False, device="meta", dtype=w.dtype)
    lin.weight = torch.nn.Parameter(w, requires_grad=False)
    return lin


def patch_gdn(model: torch.nn.Module) -> int:
    n = 0
    for mod in model.modules():
        if type(mod).__name__ != "Qwen3_5GatedDeltaNet":
            continue
        mod.in_proj_qkvz = _linear_from(torch.cat([mod.in_proj_qkv.weight.data, mod.in_proj_z.weight.data], dim=0))
        mod.in_proj_ba = _linear_from(torch.cat([mod.in_proj_b.weight.data, mod.in_proj_a.weight.data], dim=0))
        del mod.in_proj_qkv, mod.in_proj_z, mod.in_proj_b, mod.in_proj_a
        mod.forward = types.MethodType(_gdn_prefill_forward, mod)
        n += 1
    torch.cuda.empty_cache()
    return n


# --------------------------------------------------------------------------------------
# MLP: merged gate/up GEMM + fused SiLU(gate) * up
# --------------------------------------------------------------------------------------


@triton.jit
def _silu_mul_kernel(X, Y, stride_x, stride_y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = cols < N
    g = tl.load(X + row * stride_x + cols, mask=m, other=0.0).to(tl.float32)
    u = tl.load(X + row * stride_x + N + cols, mask=m, other=0.0).to(tl.float32)
    y = g * tl.sigmoid(g) * u
    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=m)


def silu_mul(x: torch.Tensor) -> torch.Tensor:
    """x [..., 2N] = [gate | up] -> silu(gate) * up, [..., N]."""
    shp = x.shape
    N = shp[-1] // 2
    x2 = x.reshape(-1, 2 * N)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    y = torch.empty((x2.shape[0], N), dtype=x.dtype, device=x.device)
    if x2.shape[0]:
        BLOCK = 1024
        _silu_mul_kernel[(x2.shape[0], triton.cdiv(N, BLOCK))](x2, y, x2.stride(0), y.stride(0), N, BLOCK=BLOCK, num_warps=4)
    return y.reshape(*shp[:-1], N)


def _mlp_forward(self, x):
    return self.down_proj(silu_mul(self.gate_up_proj(x)))


def patch_mlp(model: torch.nn.Module) -> int:
    n = 0
    for mod in model.modules():
        if type(mod).__name__ != "Qwen3_5MLP":
            continue
        mod.gate_up_proj = _linear_from(torch.cat([mod.gate_proj.weight.data, mod.up_proj.weight.data], dim=0))
        del mod.gate_proj, mod.up_proj
        mod.forward = types.MethodType(_mlp_forward, mod)
        n += 1
    torch.cuda.empty_cache()
    return n


# --------------------------------------------------------------------------------------
# Shape-independent autotune keys for fla kernels
# --------------------------------------------------------------------------------------

PIN_VALUE = 0  # a value the unpinned code never produces, so stale on-disk autotune entries are not reused


def _unwrap(obj, cls):
    for _ in range(5):
        if obj is None or isinstance(obj, cls):
            return obj
        obj = getattr(obj, "fn", None)
    return None


def _body_references(src: str, name: str) -> bool:
    """True if ``name`` appears in the kernel body (conservative: any whole-word hit counts)."""
    depth, end = 0, -1
    start = src.find("(")
    for i in range(start, len(src)):
        if src[i] == "(":
            depth += 1
        elif src[i] == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    body = src[end + 1:] if end > 0 else src
    return re.search(rf"\b{re.escape(name)}\b", body) is not None


def pin_autotune_keys(names: tuple[str, ...] = ("B", "NB")) -> list[str]:
    """Stop per-shape recompiles and autotune sweeps in fla kernels on the serving path.

    Some fla kernels take token-count-dependent values as ``tl.constexpr`` autotune keys without
    using them in the kernel body: ``chunk_local_cumsum`` (``B`` = batch size), ``causal_conv1d``
    (``NB = cdiv(B*T, 1024)``) and ``l2norm`` (``NB``). Every new value means compiling every
    config and benchmarking them, which takes seconds and shows up as multi-second stalls when
    HTTP micro-batches keep producing new shapes. For autotuned kernels whose body never references
    the name, the value is replaced by a constant, so the tuning key and the compiled binary no
    longer depend on the shape. Kernels that use the value are left alone. Returns the patched kernels.
    """
    import importlib
    import sys

    from triton.runtime.autotuner import Autotuner
    from triton.runtime.jit import JITFunction

    for mod in ("fla.ops.utils.cumsum", "fla.modules.l2norm", "fla.modules.conv.triton.kernels"):
        try:
            importlib.import_module(mod)
        except Exception:  # noqa: BLE001  (optional module)
            pass
    patched, seen = [], set()
    for mname, module in list(sys.modules.items()):
        if module is None or not (mname == "fla" or mname.startswith("fla.")):
            continue
        for obj in list(vars(module).values()):
            tuner = _unwrap(obj, Autotuner)
            if tuner is None or id(tuner) in seen:
                continue
            seen.add(id(tuner))
            jit = _unwrap(tuner.fn, JITFunction)
            if jit is None:
                continue
            constexpr = {p.name for p in getattr(jit, "params", []) if getattr(p, "is_constexpr", False)}
            pin = [n for n in names if n in tuner.keys and n in constexpr and not _body_references(jit.src, n)]
            if not pin:
                continue
            index = {n: jit.arg_names.index(n) for n in pin}
            orig_run = tuner.run

            def run(self, *args, _orig=orig_run, _pin=tuple(pin), _index=index, **kwargs):
                if args:
                    args = list(args)
                    for n in _pin:
                        if _index[n] < len(args):
                            args[_index[n]] = PIN_VALUE
                for n in _pin:
                    if n in kwargs:
                        kwargs[n] = PIN_VALUE
                return _orig(*args, **kwargs)

            tuner.run = types.MethodType(run, tuner)
            patched.append(f"{jit.fn.__name__}[{','.join(pin)}]")
    return sorted(patched)
