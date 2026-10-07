"""In-process engine benchmark (no HTTP): forward throughput/latency vs batch shape.

Rows: one per (variant, T, B): median batch latency, backbone/head split, valid tokens/s.
  --variants   comma list of mask_mode:head_impl run on the BF16 model (no reload)
  --then-quant quantize in place afterwards (fp8_rowwise|fp8_tensor) and rerun --quant-variants
  --jitter     batch records get lengths spread over [T*(1-jitter), T] (creates padding)
  --reference  also time the literal reference ``jsm.systemone`` (batch=1) on workload files
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
from clef_engine import ClefEngine, apply_fp8, jsm  # noqa: E402

QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which team should handle this ticket?",
                   "criteria": {"billing": "Payments, invoices, refunds", "technical": "Bugs, outages, performance",
                                "account": "Login, access, profile changes", "sales": "Upgrades and pricing"}},
    "urgency": {"type": "score", "instructions": "How quickly does this need a response?", "criteria": ["Can wait", "This week", "Today", "Immediately"]},
    "outage": {"type": "noul", "instructions": "Is a production service down or degraded?"},
}
WORDS = ["ticket", "customer", "service", "error", "refund", "deploy", "latency", "invoice"]


def record_of_length(engine: ClefEngine, T: int, salt: int) -> jsm.EncodedRecord:
    base, _ = engine.encoder.encode({"state": "", "questions": QUESTIONS})
    k = max(1, T - len(base.input_ids))
    enc = None
    for _ in range(6):
        state = " ".join(WORDS[(i + salt) % len(WORDS)] for i in range(k))
        enc, _ = engine.encoder.encode({"state": state, "questions": QUESTIONS})
        d = T - len(enc.input_ids)
        if d == 0:
            break
        k = max(1, k + d)
    return enc


def bench_shapes(engine, shapes, iters, warm, jitter, out, tag):
    for T, B in shapes:
        lens = [int(T * (1 - jitter * (i / max(B - 1, 1)))) for i in range(B)]
        recs = [record_of_length(engine, L, s) for s, L in enumerate(lens)]
        try:
            for _ in range(warm):
                engine.run(recs)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            lat, bb, hd = [], [], []
            for _ in range(iters):
                _, tm = engine.run(recs)
                lat.append(tm.total_ms)
                bb.append(tm.backbone_ms)
                hd.append(tm.head_ms)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(json.dumps(dict(tag=tag, T=T, B=B, oom=True)), flush=True)
            continue
        valid = sum(len(r.input_ids) for r in recs)
        ms = statistics.median(lat)
        row = dict(tag=tag, quant=engine.quant, fused=engine.fused, mask_mode=engine.mask_mode, head_impl=engine.head_impl, jitter=jitter,
                   T=T, B=B, tokens=valid, padded=B * max(len(r.input_ids) for r in recs),
                   ms=round(ms, 2), backbone_ms=round(statistics.median(bb), 2), head_ms=round(statistics.median(hd), 2),
                   tok_s=round(valid / (ms / 1000), 1), backbone_tok_s=round(valid / (statistics.median(bb) / 1000), 1),
                   peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1))
        print(json.dumps(row), flush=True)
        if out:
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")


def bench_reference(engine, files, n, out, tag):
    for path in files:
        rows = [json.loads(line) for line in open(path)][:n]
        reqs = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
        for r in reqs:
            r.setdefault("model", "clef")
        jsm.systemone(engine.model, engine.processor, reqs[0])
        lat, toks = [], []
        for r in reqs:
            torch.cuda.synchronize()
            t = time.perf_counter()
            resp = jsm.systemone(engine.model, engine.processor, r)
            lat.append((time.perf_counter() - t) * 1000)
            toks.append(resp["usage"]["input_tokens"])
        lat_sorted = sorted(lat)
        row = dict(tag=tag, workload=os.path.basename(path), n=len(lat), mean_tokens=round(statistics.mean(toks)),
                   lat_p50_ms=round(statistics.median(lat), 1), lat_p95_ms=round(lat_sorted[int(0.95 * (len(lat) - 1))], 1),
                   serial_rps=round(1000 / statistics.mean(lat), 3), serial_tok_s=round(sum(toks) / (sum(lat) / 1000), 1))
        print(json.dumps(row), flush=True)
        if out:
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")


def parse_shapes(s):
    return [tuple(int(v) for v in x.split("x")) for x in s.split(",") if x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn-impl", default="sdpa")
    ap.add_argument("--variants", default="none:fast")
    ap.add_argument("--shapes", default="512x1,512x4,512x16,512x32,512x64,1536x1,1536x2,1536x4,1536x8,1536x16,1536x32,4096x1,4096x2,4096x4,4096x8,16384x1,16384x2")
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--then-quant", default="")
    ap.add_argument("--quant-variants", default="none:fast")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--reference", default="")
    ap.add_argument("--ref-n", type=int, default=20)
    ap.add_argument("--out", default="/cache/results/engine_bench.jsonl")
    ap.add_argument("--tag", default="")
    ap.add_argument("--fused", default="none", help="none | 1 | gdn | rmsnorm | gdn,rmsnorm")
    ap.add_argument("--quant", default="none", help="quantize at load: none | fp8_rowwise | fp8_tensor")
    ap.add_argument("--compile", default="none")
    args = ap.parse_args()
    engine = ClefEngine(quant=args.quant, attn_impl=args.attn_impl, fused=args.fused, compile_mode=args.compile)
    print("loaded", engine.config_summary, flush=True)
    if args.reference:
        bench_reference(engine, args.reference.split(","), args.ref_n, args.out, "reference-systemone")
    shapes = parse_shapes(args.shapes)
    for v in args.variants.split(","):
        if not v:
            continue
        engine.mask_mode, engine.head_impl = v.split(":")
        bench_shapes(engine, shapes, args.iters, args.warm, args.jitter, args.out, args.tag)
    if args.then_quant:
        t = time.time()
        engine.apply_quant(args.then_quant)
        n = engine.quantized_linears
        print(f"quantized {n} linears to {args.then_quant} in {time.time() - t:.1f}s; mem {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)
        for v in args.quant_variants.split(","):
            engine.mask_mode, engine.head_impl = v.split(":")
            bench_shapes(engine, shapes, args.iters, args.warm, args.jitter, args.out, args.tag)


if __name__ == "__main__":
    main()
