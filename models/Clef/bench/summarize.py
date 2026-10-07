"""Turn loadgen / engine-bench JSONL results into markdown tables for the report.

python bench/summarize.py results/*.jsonl
"""

from __future__ import annotations

import collections
import json
import sys


def load(paths):
    rows = []
    for p in paths:
        for line in open(p):
            line = line.strip()
            if line.startswith("{"):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def http_tables(rows):
    groups = collections.defaultdict(list)
    for r in rows:
        if "concurrency" in r or r.get("mode") == "open":
            groups[(r["workload"], r["tag"])].append(r)
    by_wl = collections.defaultdict(dict)
    for (wl, tag), rs in groups.items():
        by_wl[wl][tag] = sorted(rs, key=lambda r: r.get("concurrency", r.get("rate", 0)))
    for wl, tags in sorted(by_wl.items()):
        print(f"\n### {wl}\n")
        print("| config | conc | req/s | input tok/s | p50 ms | p95 ms | p99 ms | avg batch | pad eff |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for tag, rs in tags.items():
            for r in rs:
                s = r.get("server", {})
                print(f"| {tag} | {r.get('concurrency', r.get('rate'))} | {r['rps']:.2f} | {r['input_tok_s']:.0f} | {r['lat_p50_ms']:.0f} | "
                      f"{r['lat_p95_ms']:.0f} | {r['lat_p99_ms']:.0f} | {s.get('avg_batch_size', float('nan')):.1f} | {s.get('padding_efficiency', float('nan')):.2f} |")


def peak_table(rows):
    groups = collections.defaultdict(list)
    for r in rows:
        if "concurrency" in r:
            groups[(r["workload"], r["tag"])].append(r)
    wls = sorted({k[0] for k in groups})
    tags = sorted({k[1] for k in groups})
    print("\n### Peak input tokens/s (best concurrency) and c=1 p50 latency\n")
    print("| config | " + " | ".join(wls) + " |")
    print("|---|" + "---:|" * len(wls))
    for t in tags:
        cells = []
        for w in wls:
            rs = groups.get((w, t))
            if not rs:
                cells.append("")
                continue
            best = max(rs, key=lambda r: r["input_tok_s"])
            c1 = [r for r in rs if r.get("concurrency") == 1]
            lat = f"{c1[0]['lat_p50_ms']:.0f} ms" if c1 else "-"
            cells.append(f"{best['input_tok_s']:.0f} (c={best['concurrency']}) / {lat}")
        print(f"| {t} | " + " | ".join(cells) + " |")


def engine_tables(rows):
    eng = [r for r in rows if "T" in r and "B" in r and not r.get("oom")]
    if not eng:
        return
    print("\n### Engine (in-process) throughput by batch shape\n")
    print("| tag | quant | fused | mask | head | T | B | tokens | ms | backbone ms | head ms | tok/s | peak GB |")
    print("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in eng:
        print(f"| {r.get('tag', '')} | {r.get('quant')} | {r.get('fused', 'none')} | {r.get('mask_mode')} | {r.get('head_impl')} | {r['T']} | {r['B']} | "
              f"{r['tokens']} | {r['ms']:.1f} | {r['backbone_ms']:.1f} | {r['head_ms']:.1f} | {r['tok_s']:.0f} | {r.get('peak_gb', '')} |")
    ref = [r for r in rows if r.get("tag") == "reference-systemone"]
    if ref:
        print("\n### Reference systemone() (batch=1, serial)\n")
        print("| workload | mean tokens | p50 ms | p95 ms | req/s | tok/s |")
        print("|---|---:|---:|---:|---:|---:|")
        for r in ref:
            print(f"| {r['workload']} | {r['mean_tokens']} | {r['lat_p50_ms']} | {r['lat_p95_ms']} | {r['serial_rps']} | {r['serial_tok_s']} |")


def engine_pivot(rows):
    eng = [r for r in rows if "T" in r and "B" in r and not r.get("oom") and r.get("mask_mode") == "none" and r.get("head_impl") == "fast"]
    if not eng:
        return
    cfg = lambda r: f"{'fused' if r.get('fused', 'none') not in ('none', None) else 'eager'}+{r.get('quant')}"  # noqa: E731
    cols = sorted({cfg(r) for r in eng})
    cell = {}
    for r in eng:
        cell[((r["T"], r["B"]), cfg(r))] = r
    shapes = sorted({(r["T"], r["B"]) for r in eng})
    print("\n### Engine tok/s (batch ms) by shape and configuration\n")
    print("| T x B | " + " | ".join(cols) + " |")
    print("|---|" + "---:|" * len(cols))
    for s in shapes:
        cells = []
        for c in cols:
            r = cell.get((s, c))
            cells.append(f"{r['tok_s']:.0f} ({r['ms']:.0f} ms)" if r else "")
        print(f"| {s[0]} x {s[1]} | " + " | ".join(cells) + " |")


def accuracy_table(rows):
    acc = [r for r in rows if "accuracy" in r and "step" in r]
    if not acc:
        return
    print("\n### BANKING77 accuracy / calibration\n")
    print("| run | step | n | accuracy | macro-F1 | NLL | Brier | ECE | flips vs BF16 | max dp | mean dp | tok/s |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in acc:
        d = r.get("vs_first", {})
        step = r["step"] + (f" (BF16: `{r['fp8_skip']}`)" if r.get("fp8_skip") else "")
        print(f"| {r['tag']} | {step} | {r['n']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['nll']:.4f} | {r['brier']:.4f} | "
              f"{r['ece']:.4f} | {d.get('flips', '-')} | {d.get('max_dp', '-')} | {d.get('mean_dp', '-')} | {r.get('tok_s', '-')} |")


if __name__ == "__main__":
    rows = load(sys.argv[1:])
    engine_pivot(rows)
    engine_tables(rows)
    accuracy_table(rows)
    peak_table(rows)
    http_tables(rows)
