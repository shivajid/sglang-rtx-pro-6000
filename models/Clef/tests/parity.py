"""Correctness checks for the optimized serving path against the reference release code.

1. Token identity: CachedEncoder vs jsm.encode_record (input_ids + all spans).
2. Logit parity vs the reference (batch=1, masked, reference head) for:
   B1-fast    batch=1, fast (vectorized) head          -> head vectorization only
   BN-mask    padded batches, mask, reference head     -> batching only
   BN-nomask  padded batches, no mask, reference head  -> mask removal
   BN-opt     padded batches, no mask, fast head       -> full optimized path
Usage: python tests/parity.py [--quant fp8_rowwise]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))

from clef_engine import ClefEngine, jsm  # noqa: E402

README_RECORDS = [
    {
        "state": {"invoice": {"vendor": "Acme", "total": 1250.0, "currency": "USD", "status": "overdue"}},
        "questions": {
            "status": {"type": "choice", "instructions": "What is the invoice status?",
                       "criteria": {"paid": "Invoice is paid.", "overdue": "Invoice is past due.", "draft": "Not sent."}},
            "large": {"type": "noul", "instructions": "Is the total above 1000 USD?"},
        },
    },
    {
        "model": "clef",
        "state": "Our checkout started returning errors and orders are blocked.",
        "questions": {
            "department": {"type": "choice", "instructions": "Which team should handle the message?",
                           "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}},
            "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
            "outage": {"type": "noul", "instructions": "Is a service down?"},
        },
    },
]

EDGE_RECORDS = [
    {"state": {"note": "Café prices rose 20% — 顧客は不満です", "tags": ["pricing", "eu"], "n": [1, 2.5, None, True]},
     "questions": {"complaint": {"type": "noul"},
                   "tone": {"type": "choice", "criteria": {"zeta": "Very formal", "alpha": "Casual", "mid": None}}}},
    {"state": "Short.", "questions": {"q": {"type": "noul", "instructions": "Is this short?",
                                            "criteria": {"true": "Yes it is brief", "false": "It is long"}}}},
    {"state": ["list", "state", {"k": 3}], "questions": {"s": {"type": "score", "instructions": "", "criteria": ["a", "b", "c", "d", "e", "f"]}}},
]


def workload_records(path: str, n: int, seed: int):
    if not os.path.exists(path):
        return []
    rows = [json.loads(line) for line in open(path)]
    random.Random(seed).shuffle(rows)
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows[:n]]


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    p = np.exp(z - z.max())
    return p / p.sum()


def compare(ref, got):
    dp, dl, agree, total = [], [], 0, 0
    for rr, gr in zip(ref, got):
        for rq, gq in zip(rr, gr):
            pr, pg = softmax(rq), softmax(gq)
            dp.append(np.abs(pr - pg).max())
            dl.append(np.abs(np.asarray(rq, dtype=np.float64) - np.asarray(gq, dtype=np.float64)).max())
            agree += int(pr.argmax() == pg.argmax())
            total += 1
    return dict(questions=total, argmax_agree=f"{agree}/{total}", max_dprob=float(np.max(dp)), mean_dprob=float(np.mean(dp)),
                p99_dprob=float(np.quantile(dp, 0.99)), max_dlogit=float(np.max(dl)))


def batched(engine, encs, budget):
    order = sorted(range(len(encs)), key=lambda i: len(encs[i].input_ids))
    out = [None] * len(encs)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and len(encs[order[j]].input_ids) * (j - i + 1) <= budget:
            j += 1
        idx = order[i:j]
        logits, _ = engine.run([encs[k] for k in idx])
        for k, lg in zip(idx, logits):
            out[k] = lg
        i = j
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", default="none")
    ap.add_argument("--workloads", default="/cache/workloads")
    ap.add_argument("--budget", type=int, default=32768)
    ap.add_argument("--out", default="/cache/results/parity.json")
    ap.add_argument("--fused", action="store_true", help="also test fast_patches (applied in place after BF16 variants)")
    ap.add_argument("--skip-bf16-variants", action="store_true")
    args = ap.parse_args()

    engine = ClefEngine(quant="none")
    tok, proc = engine.tokenizer, engine.processor
    recs = list(README_RECORDS) + list(EDGE_RECORDS)
    for name, n in (("short", 6), ("medium", 6), ("long", 4), ("xlong", 1), ("banking77", 6)):
        recs += workload_records(f"{args.workloads}/{name}.jsonl", n, 7)
    print(f"{len(recs)} records")

    # 1. token identity
    mismatches = 0
    encs = []
    for r in recs:
        ref = jsm.encode_record(tok, r, processor=proc)
        got, _ = engine.encoder.encode(r)
        same = ref.input_ids == got.input_ids and [(q.question_span, q.option_spans, q.option_ids, q.question_type) for q in ref.questions] == [
            (q.question_span, q.option_spans, q.option_ids, q.question_type) for q in got.questions]
        mismatches += int(not same)
        encs.append(got)
    lengths = [len(e.input_ids) for e in encs]
    print(f"token identity: {len(recs) - mismatches}/{len(recs)} identical; lengths min={min(lengths)} max={max(lengths)}")

    # 2. reference logits (BF16, unpatched, batch=1, masked, reference head)
    results = {"records": len(recs), "token_identity_mismatches": mismatches, "quant": args.quant, "fused": args.fused}
    engine.mask_mode, engine.head_impl = "mask", "reference"
    t = time.time()
    ref_logits = [engine.run([e])[0][0] for e in encs]
    print(f"reference done in {time.time() - t:.1f}s")

    # Also the literal reference path (jsm ClefModel.forward via collate_records) for a few records
    lit = []
    with torch.inference_mode():
        for e in encs[:5]:
            batch = jsm.collate_records([e], engine.pad_id, engine.device)
            lit.append([q.float().cpu().numpy() for q in engine.model(batch)[0]])
    results["engine_ref_vs_literal_ref"] = compare(lit, ref_logits[:5])

    def run_variant(name, mask_mode, head_impl, budget):
        engine.mask_mode, engine.head_impl = mask_mode, head_impl
        torch.cuda.synchronize()
        t = time.time()
        got = [engine.run([e])[0][0] for e in encs] if budget == 1 else batched(engine, encs, budget)
        results[name] = compare(ref_logits, got)
        results[name]["wall_s"] = round(time.time() - t, 2)
        print(name, json.dumps(results[name]), flush=True)

    if not args.skip_bf16_variants:
        run_variant("B1-fast", "none", "fast", 1)
        run_variant("BN-mask", "mask", "reference", args.budget)
        run_variant("BN-nomask", "none", "reference", args.budget)
        run_variant("BN-opt", "none", "fast", args.budget)
    if args.fused:
        from fast_patches import patch_gdn, patch_mlp, patch_rmsnorm

        print("patched", patch_gdn(engine.text_model), "GDN layers,", patch_rmsnorm(engine.text_model), "RMSNorms,",
              patch_mlp(engine.text_model), "MLPs", flush=True)
        engine.fused = "1"
        results["fused"] = True
        run_variant("B1-fused", "none", "fast", 1)
        run_variant("BN-fused", "none", "fast", args.budget)
        run_variant("BN-fused-mask", "mask", "fast", args.budget)
    if args.quant != "none":
        from clef_engine import apply_fp8

        print("quantized", apply_fp8(engine.text_model, args.quant), "linears", flush=True)
        engine.quant = args.quant
        run_variant(f"B1-{args.quant}", "none", "fast", 1)
        run_variant(f"BN-{args.quant}", "none", "fast", args.budget)
    print(json.dumps(results, indent=1))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
