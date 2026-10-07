"""Markdown tables for the report from loadgen JSONL results.

python bench/report_tables.py ladder results/clef_ablation_v2.jsonl
python bench/report_tables.py final results/final.jsonl
"""

from __future__ import annotations

import collections
import json
import sys

LADDER = ["A", "B", "C", "D", "F", "G", "G8", "G4", "G32"]
LABEL = {
    "A": "A: reference head + mask, batch 1",
    "B": "B: + dynamic batching (16K tokens)",
    "C": "C: + vectorized head",
    "D": "D: + no attention mask (exact for right padding)",
    "F": "F: + fused GDN/RMSNorm/MLP kernels",
    "G": "G: + FP8 (fp8_fast), 16K budget",
    "G8": "G8: FP8, 8K budget",
    "G4": "G4: FP8, 4K budget",
    "G32": "G32: FP8, 32K budget",
}


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.startswith('{"label')]


def ladder(rows: list[dict]) -> None:
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["tag"], r["workload"].replace(".jsonl", ""))][r["concurrency"]] = r
    wls = ["medium", "mixed", "short"]
    print("| Config | " + " | ".join(f"{w} peak tok/s" for w in wls) + " | medium c=1 p50 | medium c=16 p50 / p95 | mixed c=64 p95 |")
    print("|---|" + "---|" * (len(wls) + 3))
    for t in LADDER:
        if (t, "medium") not in by:
            continue
        cells = []
        for w in wls:
            d = by.get((t, w), {})
            cells.append(f"{max(r['input_tok_s'] for r in d.values()) / 1000:.2f}K" if d else "-")
        m = by[(t, "medium")]
        mx = by.get((t, "mixed"), {})
        c1 = m.get(1, {}).get("lat_p50_ms")
        c16 = m.get(16, {})
        p95 = mx.get(64, {}).get("lat_p95_ms")
        print(f"| {LABEL[t]} | " + " | ".join(cells) +
              f" | {c1:.0f} ms | {c16.get('lat_p50_ms', 0):.0f} / {c16.get('lat_p95_ms', 0):.0f} ms | {p95:.0f} ms |")


def final(rows: list[dict]) -> None:
    closed = [r for r in rows if r.get("mode") == "closed"]
    tags = list(dict.fromkeys(r["tag"] for r in closed))
    for wl in dict.fromkeys(r["workload"] for r in closed):
        print(f"\n**{wl.replace('.jsonl', '')}** (closed loop): req/s · tok/s · p50 / p95 ms\n")
        cs = sorted({r["concurrency"] for r in closed if r["workload"] == wl})
        print("| Deployment | " + " | ".join(f"c={c}" for c in cs) + " |")
        print("|---|" + "---|" * len(cs))
        for t in tags:
            d = {r["concurrency"]: r for r in closed if r["workload"] == wl and r["tag"] == t}
            if not d:
                continue
            cells = []
            for c in cs:
                r = d.get(c)
                cells.append(f"{r['rps']:.1f} · {r['input_tok_s'] / 1000:.1f}K · {r['lat_p50_ms']:.0f} / {r['lat_p95_ms']:.0f}"
                             + (f" ({r['errors']} err)" if r["errors"] else "") if r else "-")
            print(f"| {t} | " + " | ".join(cells) + " |")
    opened = [r for r in rows if r.get("mode") == "open"]
    if opened:
        print("\n**Open loop (Poisson arrivals, mixed workload, 60 s per rate)**\n")
        print("| Deployment | Offered req/s | Achieved req/s | tok/s | p50 ms | p95 ms | p99 ms | errors |")
        print("|---|---|---|---|---|---|---|---|")
        for r in opened:
            print(f"| {r['tag']} | {r['rate']} | {r['rps']:.2f} | {r['input_tok_s'] / 1000:.1f}K | {r['lat_p50_ms']:.0f} | "
                  f"{r['lat_p95_ms']:.0f} | {r['lat_p99_ms']:.0f} | {r['errors']} |")


def detail(rows: list[dict]) -> None:
    """Per-level throughput and tail latency (max/p50 > 2 at low concurrency flags a stall)."""
    tags = [a for a in sys.argv[3:]] or list(dict.fromkeys(r["tag"] for r in rows))
    for wl in dict.fromkeys(r["workload"] for r in rows):
        print(f"\n**{wl.replace('.jsonl', '')}**: tok/s · p50 / p99 / max ms\n")
        cs = sorted({r["concurrency"] for r in rows if r["workload"] == wl and r.get("mode") == "closed"})
        print("| Config | " + " | ".join(f"c={c}" for c in cs) + " |")
        print("|---|" + "---|" * len(cs))
        for t in tags:
            d = {r["concurrency"]: r for r in rows if r["workload"] == wl and r["tag"] == t and r.get("mode") == "closed"}
            if not d:
                continue
            cells = []
            for c in cs:
                r = d.get(c)
                cells.append(f"{r['input_tok_s'] / 1000:.2f}K · {r['lat_p50_ms']:.0f} / {r['lat_p99_ms']:.0f} / {r['lat_max_ms']:.0f}"
                             if r else "-")
            print(f"| {t} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    mode, path = sys.argv[1], sys.argv[2]
    {"ladder": ladder, "final": final, "detail": detail}[mode](load(path))
