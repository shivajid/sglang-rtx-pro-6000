"""BANKING77 accuracy, calibration and numerical drift of the serving variants.

Every record goes through the in-process engine (length-sorted batches under a token budget,
fast head, no mask = the serving path). Steps run in order on ONE loaded model and are
cumulative (patches and FP8 are applied in place):

  ref          literal reference forward (ClefModel.forward, B=1, masked) on the first --ref-n records
  bf16         serving path, unmodified BF16 weights
  fused        apply server/fast_patches (GDN + RMSNorm), then evaluate
  fp8_rowwise  quantize decoder linears (torchao dynamic FP8, per-row), then evaluate
  fp8_tensor   same, per-tensor scales

Metrics per step: accuracy, macro-F1, NLL, Brier, ECE (15 bins, top-1), mean confidence, and
drift vs the first evaluated step: argmax agreement, flips, max / p99 / mean |dp|.
Probabilities are saved to <out_dir>/probs_<tag>_<step>.npy so runs can be compared offline.

    python bench/accuracy.py --steps ref,bf16,fp8_rowwise --tag a
    python bench/accuracy.py --steps bf16,fused,fp8_rowwise --tag b
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
from clef_engine import ClefEngine, jsm  # noqa: E402,F401


def softmax(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.float64)
    p = np.exp(z - z.max())
    return p / p.sum()


def run_serving(engine: ClefEngine, recs: list, budget: int) -> tuple[np.ndarray, float]:
    order = sorted(range(len(recs)), key=lambda i: len(recs[i].input_ids))
    probs: list[np.ndarray | None] = [None] * len(recs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    i = 0
    while i < len(order):
        j, max_len = i, 0
        while j < len(order):
            n = len(recs[order[j]].input_ids)
            if max(max_len, n) * (j - i + 1) > budget and j > i:
                break
            max_len = max(max_len, n)
            j += 1
        logits, _ = engine.run_safe([recs[k] for k in order[i:j]])
        for k, lg in zip(order[i:j], logits):
            probs[k] = softmax(lg[0])
        i = j
    torch.cuda.synchronize()
    return np.stack(probs), time.perf_counter() - t0


def run_reference(engine: ClefEngine, recs: list) -> tuple[np.ndarray, float]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = []
    with torch.inference_mode():
        for r in recs:
            logits, _ = engine._run_reference([r], time.perf_counter())
            out.append(softmax(logits[0][0]))
    torch.cuda.synchronize()
    return np.stack(out), time.perf_counter() - t0


def metrics(P: np.ndarray, y: np.ndarray) -> dict:
    pred = P.argmax(1)
    correct = pred == y
    f1 = []
    for c in np.union1d(np.unique(y), np.unique(pred)):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        f1.append(2 * tp / (2 * tp + fp + fn))
    onehot = np.eye(P.shape[1])[y]
    conf = P.max(1)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 16)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return dict(
        n=int(len(y)), accuracy=round(float(correct.mean()), 4), macro_f1=round(float(np.mean(f1)), 4),
        nll=round(float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None)).mean()), 4),
        brier=round(float(((P - onehot) ** 2).sum(1).mean()), 4), ece=round(float(ece), 4),
        mean_conf=round(float(conf.mean()), 4),
    )


def drift(P: np.ndarray, P0: np.ndarray) -> dict:
    d = np.abs(P - P0)
    flips = P.argmax(1) != P0.argmax(1)
    return dict(
        agree=round(float(1 - flips.mean()), 4), flips=int(flips.sum()), max_dp=round(float(d.max()), 5),
        p99_dp=round(float(np.quantile(d.max(1), 0.99)), 5), mean_dp=round(float(d.mean()), 7),
        mean_dconf=round(float(np.abs(P.max(1) - P0.max(1)).mean()), 5),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/cache/workloads/banking77.jsonl")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--steps", default="ref,bf16,fp8_rowwise")
    ap.add_argument("--ref-n", type=int, default=200)
    ap.add_argument("--budget", type=int, default=16384)
    ap.add_argument("--tag", default="acc")
    ap.add_argument("--out", default="/cache/results/accuracy.jsonl")
    ap.add_argument("--apply", default="", help="transforms applied before the steps, without evaluating (e.g. fused)")
    ap.add_argument("--baseline", default="", help="saved probs .npy used for drift instead of the first evaluated step")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.data)][: args.n]
    engine = ClefEngine(quant="none", fused="none", head_impl="fast", mask_mode="none")
    print("loaded", engine.config_summary, flush=True)
    recs = [engine.encoder.encode({k: v for k, v in r.items() if not k.startswith("_")})[0] for r in rows]
    option_ids = list(recs[0].questions[0].option_ids)
    assert all(list(r.questions[0].option_ids) == option_ids for r in recs)
    y = np.array([option_ids.index(r["_label"]) for r in rows])
    tokens = int(sum(len(r.input_ids) for r in recs))
    out_dir = os.path.dirname(args.out)
    base_P = np.load(args.baseline)[: len(recs)] if args.baseline else None
    ref_P = None

    def transform(step: str) -> None:
        if step == "fused":
            engine.apply_fused("1")
        elif step.startswith(("fp8", "nvfp4")):
            engine.apply_quant(step)
        elif step != "bf16":
            raise ValueError(step)

    for step in [s for s in args.apply.split(",") if s]:
        transform(step)
    for step in [s for s in args.steps.split(",") if s]:
        if step == "ref":
            P, wall = run_reference(engine, recs[: args.ref_n])
            ref_P = P
            row = dict(tag=args.tag, step=step, wall_s=round(wall, 2), **metrics(P, y[: args.ref_n]))
        else:
            transform(step)
            run_serving(engine, recs[:16], args.budget)  # warm-up (autotune / compile)
            P, wall = run_serving(engine, recs, args.budget)
            row = dict(tag=args.tag, step=step, fused=engine.fused, quant=engine.quant, fp8_skip=os.environ.get("FP8_SKIP", ""),
                       nvfp4_layers=os.environ.get("NVFP4_LAYERS", ""), nvfp4_parts=os.environ.get("NVFP4_PARTS", ""),
                       wall_s=round(wall, 2), tok_s=round(tokens / wall, 1), **metrics(P, y))
            if base_P is None:
                base_P = P
            else:
                row["vs_first"] = drift(P, base_P)
            if ref_P is not None:
                row["vs_ref_subset"] = drift(P[: len(ref_P)], ref_P)
        np.save(os.path.join(out_dir, f"probs_{args.tag}_{step}.npy"), P)
        print(json.dumps(row), flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
