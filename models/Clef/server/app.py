"""Clef HTTP server: Jev/SystemOne-compatible ``POST /v1/systemone``.

Env config (see clef_engine.py for engine options):
  MODEL_PATH, MAX_LENGTH, QUANT (none|fp8_fast|nvfp4_mlp|fp8_rowwise|fp8_tensor), FUSED (none|1),
  HEAD_IMPL, MASK_MODE, ATTN_IMPL, COMPILE, PIN_AUTOTUNE (1: shape-independent fla autotune keys)
  PAD_MULTIPLE (1; 16 rounds the batch length up so Triton kernels never JIT-compile new variants on live traffic)
  MAX_BATCH_TOKENS (default 16384), MAX_BATCH_SIZE (64), MAX_WAIT_MS (0), MIN_PAD_EFF (0.8)
  PORT (8000), WARMUP (1)
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import orjson
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from clef_engine import ClefEngine, MicroBatcher, env_int, validate_request

log = logging.getLogger("clef")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MAX_BATCH_TOKENS = env_int("MAX_BATCH_TOKENS", 16384)
MAX_BATCH_SIZE = env_int("MAX_BATCH_SIZE", 64)
MAX_WAIT_MS = float(os.environ.get("MAX_WAIT_MS", "0"))

app = FastAPI(title="clef-server")
STATE: dict = {}
ENCODE_POOL = ThreadPoolExecutor(max_workers=env_int("ENCODE_THREADS", 8), thread_name_prefix="encode")


def _orjson(obj, status: int = 200) -> Response:
    return Response(orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY), status_code=status, media_type="application/json")


def _decode_media(request: dict) -> dict:
    """HTTP clients send images as base64 strings (optionally data: URIs)."""
    imgs = request.get("images")
    if not imgs:
        return request
    from PIL import Image

    out = []
    for item in imgs:
        if isinstance(item, str):
            data = item.split(",", 1)[1] if item.startswith("data:") else item
            out.append(Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB"))
        else:
            out.append(item)
    request = dict(request)
    request["images"] = out
    return request


def _warmup(engine: ClefEngine) -> None:
    words = ("the customer reported that the service was unavailable and asked for a refund " * 400).split()
    q = {
        "a": {"type": "noul", "instructions": "Is this urgent?"},
        "b": {"type": "choice", "instructions": "Team?", "criteria": {"billing": "Payments", "tech": "Bugs", "other": "Else"}},
        "c": {"type": "score", "criteria": ["low", "mid", "high"]},
    }
    # Largest batch first: with pinned autotune keys (fast_patches.pin_autotune_keys) the first call
    # decides the config for every later shape, so tune it on a throughput-sized batch.
    for n_words, bs in ((1200, 8), (300, 1), (1200, 1), (1200, 4), (3000, 2), (6000, 1), (300, 16)):
        recs = [engine.encoder.encode({"state": " ".join(words[: n_words + i]), "questions": q})[0] for i in range(bs)]
        engine.run(recs)
    torch.cuda.synchronize()


@app.on_event("startup")
async def startup() -> None:
    engine = ClefEngine()
    log.info("engine loaded: %s", engine.config_summary)
    if os.environ.get("WARMUP", "1") == "1":
        t = time.time()
        _warmup(engine)
        log.info("warmup done in %.1fs", time.time() - t)
    STATE["engine"] = engine
    STATE["batcher"] = MicroBatcher(engine, MAX_BATCH_TOKENS, MAX_BATCH_SIZE, MAX_WAIT_MS)
    STATE["started"] = time.time()


@app.get("/health")
async def health() -> Response:
    if "batcher" not in STATE:
        return _orjson({"status": "loading"}, 503)
    return _orjson({"status": "ok"})


@app.get("/stats")
async def stats() -> Response:
    b: MicroBatcher = STATE["batcher"]
    e: ClefEngine = STATE["engine"]
    snap = b.snapshot()
    snap.update(
        config=e.config_summary,
        max_batch_tokens=b.max_batch_tokens,
        max_batch_size=b.max_batch_size,
        max_wait_ms=b.max_wait * 1000,
        min_pad_eff=b.min_pad_eff,
        uptime_s=time.time() - STATE["started"],
        gpu_mem_alloc_gb=torch.cuda.memory_allocated() / 1e9,
        gpu_mem_peak_gb=torch.cuda.max_memory_allocated() / 1e9,
    )
    return _orjson(snap)


@app.post("/stats/reset")
async def stats_reset() -> Response:
    b: MicroBatcher = STATE["batcher"]
    for k in b.stats:
        b.stats[k] = 0 if isinstance(b.stats[k], int) else 0.0
    torch.cuda.reset_peak_memory_stats()
    return _orjson({"ok": True})


@app.post("/admin/config")
async def admin_config(req: Request) -> Response:
    """Runtime reconfiguration for ablations (no model reload). Quantization is one-way."""
    cfg = orjson.loads(await req.body())
    b: MicroBatcher = STATE["batcher"]
    e: ClefEngine = STATE["engine"]

    def apply() -> dict:
        with b.gpu_lock:
            if "head_impl" in cfg:
                e.head_impl = cfg["head_impl"]
            if "mask_mode" in cfg:
                e.mask_mode = cfg["mask_mode"]
            if "max_batch_tokens" in cfg:
                b.max_batch_tokens = int(cfg["max_batch_tokens"])
            if "max_batch_size" in cfg:
                b.max_batch_size = int(cfg["max_batch_size"])
            if "max_wait_ms" in cfg:
                b.max_wait = float(cfg["max_wait_ms"]) / 1000.0
            if "min_pad_eff" in cfg:
                b.min_pad_eff = float(cfg["min_pad_eff"])
            changed = False
            if cfg.get("fused") and cfg["fused"] not in ("0", "none"):
                before = e.fused
                e.apply_fused(str(cfg["fused"]))
                changed |= e.fused != before
            if cfg.get("quant") and cfg["quant"] != e.quant:
                e.apply_quant(cfg["quant"])
                changed = True
            if changed and cfg.get("warmup", True):
                _warmup(e)
            e.config_summary.update(head_impl=e.head_impl, mask_mode=e.mask_mode, quant=e.quant,
                                    quantized_linears=e.quantized_linears, fused=e.fused)
            return dict(e.config_summary, max_batch_tokens=b.max_batch_tokens, max_batch_size=b.max_batch_size,
                        max_wait_ms=b.max_wait * 1000, min_pad_eff=b.min_pad_eff)

    try:
        out = await asyncio.get_running_loop().run_in_executor(None, apply)
    except Exception as exc:  # noqa: BLE001
        return _orjson({"error": str(exc)}, 400)
    log.info("reconfigured: %s", out)
    return _orjson(out)


@app.post("/v1/systemone")
async def systemone(req: Request) -> Response:
    t0 = time.perf_counter()
    try:
        request = orjson.loads(await req.body())
        validate_request(request)
    except Exception as exc:  # noqa: BLE001
        return _orjson({"error": str(exc)}, 400)
    engine: ClefEngine = STATE["engine"]
    batcher: MicroBatcher = STATE["batcher"]
    loop = asyncio.get_running_loop()
    try:
        request_m = _decode_media(request)
        record, truncated = await loop.run_in_executor(ENCODE_POOL, engine.encoder.encode, request_m)
    except ValueError as exc:
        return _orjson({"error": str(exc)}, 400)
    t1 = time.perf_counter()
    logits, info = await batcher.submit(record, loop)
    answers = engine.answers(request, record, logits)
    body = {
        "model": request["model"],
        "answers": answers,
        "usage": {"input_tokens": len(record.input_ids), "output_tokens": 0},
    }
    if truncated:
        body["warnings"] = ["state truncated to fit max_length"]
    if request.get("debug"):
        body["debug"] = dict(
            info,
            encode_ms=(t1 - t0) * 1000,
            server_ms=(time.perf_counter() - t0) * 1000,
            logits={q.question_id: lg.tolist() for q, lg in zip(record.questions, logits)},
        )
    return _orjson(body)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=env_int("PORT", 8000), log_level="warning", access_log=False)
