#!/usr/bin/env python3
"""
SGLang 1K/8K Benchmark Sweep Script for GLM-5.2-NVFP4.
Runs sglang.bench_serving across concurrencies (128, 256, 512) with prompts = 3x concurrency.
"""

import argparse
import json
import os
import subprocess
import sys
import time


def parse_jsonl(filepath, sc_name, in_len, out_len, conc, num_prompts, elapsed):
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        return None

    try:
        summary = json.loads(lines[-1])
    except Exception as e:
        print(f"Error parsing summary JSON line: {e}")
        return None

    req_lines = []
    for line in lines[:-1]:
        try:
            req_lines.append(json.loads(line))
        except Exception:
            pass

    completed = summary.get("completed", len(req_lines) if req_lines else num_prompts)
    duration = summary.get("duration", elapsed)
    total_input = summary.get("total_input_tokens", in_len * completed)
    total_output = summary.get("total_output_tokens", summary.get("total_generated_tokens", out_len * completed))

    out_tps = summary.get("output_throughput", (total_output / duration) if duration > 0 else 0.0)
    total_tps = summary.get("total_throughput", ((total_input + total_output) / duration) if duration > 0 else 0.0)
    req_tps = summary.get("request_throughput", (completed / duration) if duration > 0 else 0.0)

    mean_tpot = summary.get("mean_tpot_ms", 0.0)
    stream_speed = round(1000.0 / mean_tpot, 2) if mean_tpot > 0 else 0.0

    res_row = {
        "scenario": sc_name,
        "concurrency": conc,
        "successful_requests": completed,
        "duration_s": round(duration, 2),
        "total_input_tokens": total_input,
        "total_generated_tokens": total_output,
        "output_throughput": round(out_tps, 2),
        "peak_output_throughput": round(summary.get("peak_output_throughput", 0.0), 2),
        "total_throughput": round(total_tps, 2),
        "request_throughput": round(req_tps, 2),
        "concurrency_measured": round(summary.get("concurrency", float(conc)), 2),
        "e2e_mean_ms": round(summary.get("mean_e2e_latency_ms", 0.0), 2),
        "e2e_median_ms": round(summary.get("median_e2e_latency_ms", 0.0), 2),
        "e2e_p90_ms": round(summary.get("p90_e2e_latency_ms", 0.0), 2),
        "e2e_p95_ms": round(summary.get("p95_e2e_latency_ms", 0.0), 2),
        "e2e_p99_ms": round(summary.get("p99_e2e_latency_ms", 0.0), 2),
        "ttft_mean_ms": round(summary.get("mean_ttft_ms", 0.0), 2),
        "ttft_median_ms": round(summary.get("median_ttft_ms", 0.0), 2),
        "ttft_p90_ms": round(summary.get("p90_ttft_ms", 0.0), 2),
        "ttft_p95_ms": round(summary.get("p95_ttft_ms", 0.0), 2),
        "ttft_p99_ms": round(summary.get("p99_ttft_ms", 0.0), 2),
        "tpot_mean_ms": round(summary.get("mean_tpot_ms", 0.0), 2),
        "tpot_median_ms": round(summary.get("median_tpot_ms", 0.0), 2),
        "tpot_p90_ms": round(summary.get("p90_tpot_ms", 0.0), 2),
        "tpot_p95_ms": round(summary.get("p95_tpot_ms", 0.0), 2),
        "tpot_p99_ms": round(summary.get("p99_tpot_ms", 0.0), 2),
        "itl_mean_ms": round(summary.get("mean_itl_ms", 0.0), 2),
        "itl_median_ms": round(summary.get("median_itl_ms", 0.0), 2),
        "itl_p90_ms": round(summary.get("p90_itl_ms", 0.0), 2),
        "itl_p95_ms": round(summary.get("p95_itl_ms", 0.0), 2),
        "itl_p99_ms": round(summary.get("p99_itl_ms", 0.0), 2),
        "itl_max_ms": round(summary.get("max_itl_ms", 0.0), 2),
        "stream_speed_tok_s": stream_speed,
        "median_latency_s": round(summary.get("median_e2e_latency_ms", 0.0) / 1000.0, 2),
    }
    return res_row


def generate_markdown_report(results, output_file):
    lines = []
    lines.append("# GLM-5.2 NVFP4 1K/8K Serving Benchmark Report")
    lines.append("")
    lines.append("- **Model**: `nvidia/GLM-5.2-NVFP4`")
    lines.append("- **Hardware**: 1x G4 Node (8x RTX PRO 6000 Ada / SM120)")
    lines.append("- **Engine**: SGLang stock `latest-main` (dev-cu13)")
    lines.append("- **Scenario**: 1K Input / 8K Output (`random-input-len 1024`, `random-output-len 8192`, `rrr 1.0`)")
    lines.append("- **Prompts**: 3x Concurrency")
    lines.append("")
    lines.append("## Throughput & Latency Summary")
    lines.append("")
    lines.append("| Concurrency | Prompts | Duration (s) | Output TPS (tok/s) | Total TPS (tok/s) | Req/s | Median TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | Median ITL (ms) | Median Latency (s) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['concurrency']} | {r['successful_requests']} | {r['duration_s']} | "
            f"**{r['output_throughput']}** | {r['total_throughput']} | {r['request_throughput']} | "
            f"{r['ttft_median_ms']} | {r['ttft_p99_ms']} | {r['tpot_mean_ms']} | "
            f"{r['itl_median_ms']} | {r['median_latency_s']} |"
        )
    lines.append("")

    with open(output_file, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"📄 Markdown report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run SGLang 1K/8K benchmark sweep for GLM-5.2-NVFP4")
    parser.add_argument("--host", type=str, default="sglang-glm52-nvfp4-service.default.svc.cluster.local", help="SGLang server host or service name")
    parser.add_argument("--port", type=int, default=8000, help="SGLang server port")
    parser.add_argument("--model", type=str, default="nvidia/GLM-5.2-NVFP4", help="Model name")
    parser.add_argument("--tokenizer", type=str, default="nvidia/GLM-5.2-NVFP4", help="Tokenizer repo for benchmark client")
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[128, 256, 512], help="Concurrencies to sweep (default: 128 256 512)")
    parser.add_argument("--prompts-multiplier", type=int, default=3, help="Multiplier for total prompts (default: 3x concurrency)")
    parser.add_argument("--input-len", type=int, default=1024, help="Input prompt length (default: 1024)")
    parser.add_argument("--output-len", type=int, default=8192, help="Output generation length (default: 8192)")
    parser.add_argument("--random-range-ratio", type=float, default=1.0, help="Random range ratio (default: 1.0)")
    parser.add_argument("--output-dir", type=str, default="/tmp/bench_results_glm52", help="Directory to save JSONL output files")
    parser.add_argument("--warmup", action="store_true", default=True, help="Run a quick 2-prompt warmup before the sweep")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 85)
    print("🚀 GLM-5.2-NVFP4 SGLANG 1K/8K BENCHMARK SWEEP")
    print(f"📍 Target Host: {args.host}:{args.port}")
    print(f"🤖 Model: {args.model}")
    print(f"🔤 Tokenizer: {args.tokenizer}")
    print(f"👥 Concurrencies: {args.concurrencies}")
    print(f"📊 Scenario: In={args.input_len}, Out={args.output_len}, RRR={args.random_range_ratio}")
    print(f"✖️  Prompts: {args.prompts_multiplier}x concurrency ({[c * args.prompts_multiplier for c in args.concurrencies]})")
    print("=" * 85)

    if args.warmup:
        print("\n🔥 Executing initial warm-up (2 requests)...")
        warmup_file = os.path.join(args.output_dir, "warmup.jsonl")
        warmup_cmd = [
            sys.executable, "-m", "sglang.bench_serving",
            "--backend", "sglang",
            "--host", args.host,
            "--port", str(args.port),
            "--model", args.model,
            "--tokenizer", args.tokenizer,
            "--dataset-name", "random",
            "--random-input-len", "1024",
            "--random-output-len", "512",
            "--random-range-ratio", "0.0",
            "--num-prompts", "2",
            "--max-concurrency", "2",
            "--output-file", warmup_file,
            "--output-details"
        ]
        subprocess.run(warmup_cmd, check=False)
        print("✅ Warm-up complete.\n")
        time.sleep(3)

    results_table = []
    sc_name = f"{int(args.input_len/1024)}k_{int(args.output_len/1024)}k"

    for conc in args.concurrencies:
        num_prompts = conc * args.prompts_multiplier
        out_file = os.path.join(args.output_dir, f"glm52_{sc_name}_c{conc}.jsonl")
        if os.path.exists(out_file):
            os.remove(out_file)

        cmd = [
            sys.executable, "-m", "sglang.bench_serving",
            "--backend", "sglang",
            "--host", args.host,
            "--port", str(args.port),
            "--model", args.model,
            "--tokenizer", args.tokenizer,
            "--dataset-name", "random",
            "--random-input-len", str(args.input_len),
            "--random-output-len", str(args.output_len),
            "--random-range-ratio", str(args.random_range_ratio),
            "--num-prompts", str(num_prompts),
            "--max-concurrency", str(conc),
            "--output-file", out_file,
            "--output-details"
        ]

        print(f"\n" + "-" * 85)
        print(f"▶️  Running: Concurrency={conc} | Total Prompts={num_prompts} | Scenario={sc_name}")
        print(f"Command: {' '.join(cmd)}")
        print("-" * 85)

        start_t = time.time()
        try:
            proc = subprocess.run(cmd, check=False)
            elapsed = time.time() - start_t
            data = parse_jsonl(out_file, sc_name, args.input_len, args.output_len, conc, num_prompts, elapsed)
            if data:
                results_table.append(data)
                print("\n" + "=" * 85)
                print(f"✅ Concurrency {conc} Complete in {data['duration_s']}s")
                print(f"   Output Throughput: {data['output_throughput']} tok/s (Total: {data['total_throughput']} tok/s)")
                print(f"   Median TTFT: {data['ttft_median_ms']} ms | Mean TPOT: {data['tpot_mean_ms']} ms | Median Latency: {data['median_latency_s']}s")
                print("=" * 85)
            else:
                print(f"⚠️ Warning: No JSONL summary parsed for concurrency {conc}")
        except Exception as e:
            print(f"❌ Execution failed for concurrency {conc}: {e}")

        time.sleep(5)

    print("\n" + "=" * 125)
    print("🏆 FINAL GLM-5.2-NVFP4 1K/8K BENCHMARK RESULTS")
    print("=" * 125)
    header = f"| {'Scenario':<8} | {'Conc':<5} | {'Prompts':<7} | {'Out TPS':<12} | {'Total TPS':<12} | {'Req/s':<7} | {'TTFT P50':<11} | {'TTFT P99':<11} | {'TPOT Mean':<11} | {'ITL P50':<10} | {'E2E P50':<10} |"
    divider = f"|{'-'*10}|{'-'*7}|{'-'*9}|{'-'*14}|{'-'*14}|{'-'*9}|{'-'*13}|{'-'*13}|{'-'*13}|{'-'*12}|{'-'*12}|"
    print(header)
    print(divider)
    for r in results_table:
        print(
            f"| {r['scenario']:<8} | {r['concurrency']:<5} | {r['successful_requests']:<7} | "
            f"{r['output_throughput']:>9.2f} t/s | {r['total_throughput']:>9.2f} t/s | {r['request_throughput']:>7.2f} | "
            f"{r['ttft_median_ms']:>8.1f} ms | {r['ttft_p99_ms']:>8.1f} ms | {r['tpot_mean_ms']:>8.1f} ms | "
            f"{r['itl_median_ms']:>7.1f} ms | {r['median_latency_s']:>8.2f} s |"
        )
    print("=" * 125)

    # Save summary JSON
    json_path = os.path.join(args.output_dir, "benchmark_data.json")
    with open(json_path, "w") as f:
        json.dump(results_table, f, indent=2)
    print(f"\n💾 Summary JSON saved to: {json_path}")

    # Save summary Markdown
    md_path = os.path.join(args.output_dir, "benchmark_report.md")
    generate_markdown_report(results_table, md_path)

    print("\n### BENCHMARK_ALL_DATA_START ###")
    print(json.dumps(results_table, indent=2))
    print("### BENCHMARK_ALL_DATA_END ###\n")


if __name__ == "__main__":
    main()
