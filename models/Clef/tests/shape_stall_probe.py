"""Per-batch latency on a stream of previously unseen (B, T) shapes, after the server warmup.

Each shape runs twice back to back: ``first_ms - repeat_ms`` is the one-off cost of a new shape
(Triton recompiles / autotune sweeps). Run once with PIN_AUTOTUNE=0 and once with PIN_AUTOTUNE=1.

  PIN_AUTOTUNE=1 FUSED=1 QUANT=fp8_fast python tests/shape_stall_probe.py --n 40 --tag pin
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))

import torch  # noqa: E402

from app import _warmup  # noqa: E402
from clef_engine import ClefEngine  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="/cache/results/stall_probe.jsonl")
    args = ap.parse_args()

    t0 = time.time()
    engine = ClefEngine()
    print("loaded", engine.config_summary, "pinned:", engine.autotune_pinned, flush=True)
    _warmup(engine)
    print(f"init+warmup {time.time() - t0:.1f}s", flush=True)

    words = ("the customer reported that the service was unavailable and asked for a refund " * 600).split()
    q = {
        "a": {"type": "noul", "instructions": "Is this urgent?"},
        "b": {"type": "choice", "instructions": "Team?", "criteria": {"billing": "Payments", "tech": "Bugs", "other": "Else"}},
    }
    rng = random.Random(args.seed)
    rows, seen = [], set()
    while len(rows) < args.n:
        n_words = rng.randint(150, 2600)
        B = rng.randint(1, 16)
        recs = [engine.encoder.encode({"state": " ".join(words[: n_words + 3 * i]), "questions": q})[0] for i in range(B)]
        T = max(len(r.input_ids) for r in recs)
        if B * T > args.max_tokens or (B, T) in seen:
            continue
        seen.add((B, T))
        times = []
        for _ in range(2):
            torch.cuda.synchronize()
            t = time.perf_counter()
            engine.run(recs)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t) * 1000)
        row = dict(B=B, T=T, tokens=B * T, first_ms=round(times[0], 1), repeat_ms=round(times[1], 1),
                   excess_ms=round(times[0] - times[1], 1))
        rows.append(row)
        print(json.dumps(row), flush=True)

    excess = [r["excess_ms"] for r in rows]
    stalls = [r for r in rows if r["first_ms"] > 1.5 * r["repeat_ms"] + 20]
    summary = dict(
        tag=args.tag, fused=engine.fused, quant=engine.quant, pinned=engine.autotune_pinned, n=len(rows),
        stalls=len(stalls), excess_total_ms=round(sum(excess), 1), excess_max_ms=round(max(excess), 1),
        first_total_ms=round(sum(r["first_ms"] for r in rows), 1),
        repeat_total_ms=round(sum(r["repeat_ms"] for r in rows), 1),
        rows=rows,
    )
    print("SUMMARY", json.dumps({k: v for k, v in summary.items() if k != "rows"}), flush=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
