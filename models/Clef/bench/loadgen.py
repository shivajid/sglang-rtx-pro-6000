"""Async load generator for the Clef server (and an SGLang backbone baseline).

Closed loop:  --concurrency C --num N        (C workers, N measured requests)
Open loop:    --rate R --duration S          (Poisson arrivals at R req/s)
Sweep:        --sweep 1,2,4,8,16             (closed-loop concurrency sweep)

--target clef    POST /v1/systemone with the JSONL request bodies
--target sglang  POST /generate with random input_ids of the same token lengths and
                 max_new_tokens=1 (backbone prefill ceiling, same length distribution)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time

import aiohttp


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f, c = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def load_requests(path: str, target: str, seed: int):
    rows = [json.loads(line) for line in open(path)]
    rng = random.Random(seed)
    rng.shuffle(rows)
    out = []
    for r in rows:
        n = r.get("_tokens", 1000)
        if target == "clef":
            body = {k: v for k, v in r.items() if not k.startswith("_")}
            out.append((json.dumps(body).encode(), n))
        else:
            ids = [rng.randint(1000, 150000) for _ in range(n)]
            body = {"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0.0}}
            out.append((json.dumps(body).encode(), n))
    return out


async def one(session, url, payload, target):
    t0 = time.perf_counter()
    try:
        async with session.post(url, data=payload, headers={"Content-Type": "application/json"}) as resp:
            body = await resp.read()
            ok = resp.status == 200
    except Exception:  # noqa: BLE001
        return time.perf_counter() - t0, False, 0
    dt = time.perf_counter() - t0
    toks = 0
    if ok:
        try:
            j = json.loads(body)
            toks = j["usage"]["input_tokens"] if target == "clef" else j["meta_info"]["prompt_tokens"]
        except Exception:  # noqa: BLE001
            pass
    return dt, ok, toks


async def closed_loop(url, reqs, target, concurrency, num, warmup):
    results = []
    idx = 0
    lock = asyncio.Lock()
    conn = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        # warmup
        await asyncio.gather(*[one(session, url, reqs[i % len(reqs)][0], target) for i in range(warmup)])

        async def worker():
            nonlocal idx
            while True:
                async with lock:
                    if idx >= num:
                        return
                    i = idx
                    idx += 1
                payload, _ = reqs[i % len(reqs)]
                res = await one(session, url, payload, target)
                results.append((time.perf_counter(), *res))

        t0 = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(concurrency)])
        wall = time.perf_counter() - t0
    return results, wall


async def open_loop(url, reqs, target, rate, duration, seed):
    rng = random.Random(seed)
    results = []
    tasks = []
    conn = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:

        async def fire(payload):
            res = await one(session, url, payload, target)
            results.append((time.perf_counter(), *res))

        t0 = time.perf_counter()
        i = 0
        next_t = t0
        while next_t - t0 < duration:
            now = time.perf_counter()
            if next_t > now:
                await asyncio.sleep(next_t - now)
            tasks.append(asyncio.create_task(fire(reqs[i % len(reqs)][0])))
            i += 1
            next_t += rng.expovariate(rate)
        await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0
    return results, wall


def summarize(results, wall, label, extra):
    lat = [r[1] * 1000 for r in results if r[2]]
    toks = sum(r[3] for r in results if r[2])
    errors = sum(1 for r in results if not r[2])
    s = dict(label=label, n=len(results), errors=errors, wall_s=round(wall, 2),
             rps=round(len(lat) / wall, 3), input_tok_s=round(toks / wall, 1),
             mean_tokens=round(toks / max(len(lat), 1), 1),
             lat_mean_ms=round(statistics.mean(lat), 1) if lat else None,
             lat_p50_ms=round(pct(lat, 50), 1), lat_p90_ms=round(pct(lat, 90), 1),
             lat_p95_ms=round(pct(lat, 95), 1), lat_p99_ms=round(pct(lat, 99), 1),
             lat_max_ms=round(max(lat), 1) if lat else None)
    s.update(extra)
    return s


async def get_json(url):
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(url) as r:
                return await r.json()
    except Exception:  # noqa: BLE001
        return None


async def post(url):
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(url) as r:
                return await r.json()
    except Exception:  # noqa: BLE001
        return None


async def main_async(args):
    reqs = load_requests(args.workload, args.target, args.seed)
    base = args.url.rstrip("/")
    url = base + ("/v1/systemone" if args.target == "clef" else "/generate")
    out = []
    rates = [float(x) for x in args.rates.split(",")] if args.rates else []
    levels = rates or ([int(x) for x in args.sweep.split(",")] if args.sweep else [args.concurrency])
    for c in levels:
        if args.target == "clef":
            await post(base + "/stats/reset")
        if rates or args.rate:
            rate = c if rates else args.rate
            res, wall = await open_loop(url, reqs, args.target, rate, args.duration, args.seed)
            label = f"rate={rate}"
            extra = dict(mode="open", rate=rate)
        else:
            num = args.num if args.num else max(args.min_num, c * args.per_worker)
            res, wall = await closed_loop(url, reqs, args.target, c, num, args.warmup if c == levels[0] else min(args.warmup, c))
            label = f"c={c}"
            extra = dict(mode="closed", concurrency=c)
        extra.update(workload=args.workload.split("/")[-1], target=args.target, tag=args.tag)
        if args.target == "clef":
            st = await get_json(base + "/stats")
            if st:
                extra["server"] = {k: st[k] for k in ("avg_batch_size", "avg_valid_tokens_per_batch", "padding_efficiency",
                                                      "backbone_ms", "head_ms", "gpu_ms", "batches", "requests", "gpu_mem_peak_gb") if k in st}
                extra["server"]["gpu_busy_frac"] = round(st["gpu_ms"] / 1000 / wall, 3) if wall else None
        s = summarize(res, wall, label, extra)
        print(json.dumps(s), flush=True)
        out.append(s)
    if args.out:
        with open(args.out, "a") as f:
            for s in out:
                f.write(json.dumps(s) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--workload", required=True)
    ap.add_argument("--target", choices=["clef", "sglang"], default="clef")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--sweep", default="")
    ap.add_argument("--num", type=int, default=0)
    ap.add_argument("--per-worker", type=int, default=8)
    ap.add_argument("--min-num", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--rate", type=float, default=0.0)
    ap.add_argument("--rates", default="", help="open-loop sweep, e.g. 2,4,6,8")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
