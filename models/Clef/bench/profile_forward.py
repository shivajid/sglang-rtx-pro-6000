"""Kernel-level profile of one backbone+head forward (where does the time go?).

python bench/profile_forward.py --T 1536 --B 8 [--quant fp8_rowwise]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
sys.path.insert(0, os.path.dirname(__file__))
from bench_engine import record_of_length  # noqa: E402
from clef_engine import ClefEngine  # noqa: E402

CATS = [
    ("gemm", ("gemm", "cutlass", "sm90_", "sm100_", "sm120_", "cublas", "nvjet", "xmma", "matmul", "scaled_mm", "f8f8", "_mm_")),
    ("attention", ("flash", "fmha", "attention", "sdpa", "efficient")),
    ("gdn_triton", ("chunk", "delta", "recompute", "fwd_h", "fwd_o", "kkt", "solve_tril", "l2norm", "cumsum", "gated")),
    ("conv", ("conv",)),
    ("norm", ("norm", "rms", "layer_norm")),
    ("quant", ("quant", "amax", "scale", "float8", "fp8")),
    ("elementwise", ("elementwise", "vectorized", "unrolled", "reduce", "copy", "cat", "index", "silu", "gelu", "mul", "add")),
]


def cat_of(name: str) -> str:
    n = name.lower()
    for cat, keys in CATS:
        if any(k in n for k in keys):
            return cat
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=1536)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--quant", default="none")
    ap.add_argument("--fused", default="none")
    ap.add_argument("--out", default="/cache/results/profile.jsonl")
    args = ap.parse_args()
    engine = ClefEngine(quant=args.quant, fused=args.fused)
    recs = [record_of_length(engine, args.T, s) for s in range(args.B)]
    for _ in range(3):
        engine.run(recs)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        _, tm = engine.run(recs)
        torch.cuda.synchronize()
    # Only GPU kernel/memcpy events (CPU ops also carry self device time -> double counting).
    rows = []
    for e in prof.key_averages():
        if e.device_type != torch.autograd.DeviceType.CUDA:
            continue
        t = getattr(e, "self_device_time_total", None)
        if t is None:
            t = getattr(e, "self_cuda_time_total", 0)
        if t and t > 0:
            rows.append((e.key, t / 1000.0, e.count))
    rows.sort(key=lambda r: -r[1])
    total = sum(r[1] for r in rows)
    by_cat = collections.Counter()
    for k, t, _ in rows:
        by_cat[cat_of(k)] += t
    print(f"T={args.T} B={args.B} quant={args.quant} fused={args.fused} batch total {tm.total_ms:.1f} ms (backbone {tm.backbone_ms:.1f}, head {tm.head_ms:.1f}); kernel sum {total:.1f} ms")
    for cat, t in by_cat.most_common():
        print(f"  {cat:12s} {t:8.1f} ms  {100 * t / total:5.1f}%")
    print("top kernels:")
    for k, t, c in rows[:30]:
        print(f"  {t:8.2f} ms  {100 * t / total:5.1f}%  x{c:<5d} [{cat_of(k)}] {k[:150]}")
    with open(args.out, "a") as f:
        f.write(json.dumps(dict(T=args.T, B=args.B, quant=args.quant, fused=args.fused, total_ms=tm.total_ms, by_cat=dict(by_cat),
                                top=[(k[:200], round(t, 3), c) for k, t, c in rows[:40]])) + "\n")


if __name__ == "__main__":
    main()
