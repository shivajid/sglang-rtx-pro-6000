"""Lightweight chat gateway for an SGLang server running Kimi-K3.

Two upstreams, both OpenAI-compatible:
  * chat / vision      -> SGLang autoregressive server (Kimi-K3)
  * image generation   -> SGLang Diffusion server (FLUX / Qwen-Image / GLM-Image)

The process is pure I/O: no model, no GPU, no database. It fits comfortably in
a 256Mi / 200m CPU pod.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


# ---------------------------------------------------------------- config ----

CHAT_BASE_URL = _env("SGLANG_BASE_URL", "http://sglang-kimi-k3:30000/v1").rstrip("/")
CHAT_API_KEY = _env("SGLANG_API_KEY")
CHAT_MODEL = _env("CHAT_MODEL", "moonshotai/Kimi-K3")

IMAGE_BASE_URL = _env("IMAGE_BASE_URL").rstrip("/")
IMAGE_API_KEY = _env("IMAGE_API_KEY") or CHAT_API_KEY
IMAGE_MODEL = _env("IMAGE_MODEL", "Qwen/Qwen-Image")

MAX_REQUEST_MB = float(_env("MAX_REQUEST_MB", "24"))
MAX_IMAGE_MB = float(_env("MAX_IMAGE_MB", "8"))
READ_TIMEOUT = float(_env("READ_TIMEOUT_SECONDS", "900"))
IMAGE_TIMEOUT = float(_env("IMAGE_TIMEOUT_SECONDS", "600"))
DEFAULT_TEMPERATURE = float(_env("DEFAULT_TEMPERATURE", "0.6"))
APP_TOKEN = _env("APP_TOKEN")  # optional shared secret; prefer IAP in front

_ready_cache: dict[str, Any] = {"at": 0.0, "ok": False, "detail": ""}


@asynccontextmanager
async def lifespan(app: FastAPI):
    timeout = httpx.Timeout(connect=10.0, read=READ_TIMEOUT, write=120.0, pool=10.0)
    limits = httpx.Limits(max_connections=64, max_keepalive_connections=16)
    app.state.http = httpx.AsyncClient(timeout=timeout, limits=limits)
    try:
        yield
    finally:
        await app.state.http.aclose()


app = FastAPI(title="Kimi-K3 Chat", lifespan=lifespan, docs_url=None, redoc_url=None)


def _headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.middleware("http")
async def guard(request: Request, call_next):
    if APP_TOKEN and request.url.path.startswith("/api/"):
        supplied = request.headers.get("x-app-token", "")
        if supplied != APP_TOKEN:
            return JSONResponse({"error": "Not authorised for this app."}, status_code=401)
    return await call_next(request)


# ----------------------------------------------------------------- models ---

class ChatRequest(BaseModel):
    messages: list[dict[str, Any]]
    model: str | None = None
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1, le=262144)


class ImageRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    size: str = "1024x1024"
    n: int = Field(default=1, ge=1, le=4)
    model: str | None = None
    seed: int | None = None


# --------------------------------------------------------------- routes -----

@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> JSONResponse:
    now = time.monotonic()
    if now - _ready_cache["at"] < 10.0:
        cached = _ready_cache
        return JSONResponse(
            {"status": "ok" if cached["ok"] else "upstream unavailable", "detail": cached["detail"]},
            status_code=200 if cached["ok"] else 503,
        )
    detail, ok = "", False
    try:
        resp = await app.state.http.get(
            f"{CHAT_BASE_URL}/models", headers=_headers(CHAT_API_KEY), timeout=5.0
        )
        ok = resp.status_code < 500
        detail = f"chat upstream returned {resp.status_code}"
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the operator
        detail = f"chat upstream unreachable: {exc.__class__.__name__}"
    _ready_cache.update(at=now, ok=ok, detail=detail)
    return JSONResponse({"status": "ok" if ok else "upstream unavailable", "detail": detail},
                        status_code=200 if ok else 503)


@app.get("/api/config")
async def config() -> dict[str, Any]:
    return {
        "chat_model": CHAT_MODEL,
        "image_model": IMAGE_MODEL if IMAGE_BASE_URL else None,
        "image_enabled": bool(IMAGE_BASE_URL),
        "max_image_mb": MAX_IMAGE_MB,
        "default_temperature": DEFAULT_TEMPERATURE,
    }


@app.get("/api/models")
async def models() -> JSONResponse:
    try:
        resp = await app.state.http.get(
            f"{CHAT_BASE_URL}/models", headers=_headers(CHAT_API_KEY), timeout=10.0
        )
        return JSONResponse(resp.json(), status_code=resp.status_code)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"Model list unavailable: {exc}"}, status_code=502)


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request) -> StreamingResponse:
    body = {
        "model": req.model or CHAT_MODEL,
        "messages": req.messages,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if req.max_tokens:
        body["max_tokens"] = req.max_tokens

    raw = json.dumps(body)
    if len(raw.encode()) > MAX_REQUEST_MB * 1024 * 1024:
        async def too_big() -> AsyncIterator[str]:
            yield sse({"type": "error", "message":
                       f"Conversation exceeds the {MAX_REQUEST_MB:g} MB request limit. "
                       f"Remove some attached images or start a new chat."})
        return StreamingResponse(too_big(), media_type="text/event-stream")

    async def relay() -> AsyncIterator[str]:
        started = time.monotonic()
        first_token_ms: float | None = None
        usage: dict[str, Any] | None = None
        try:
            async with app.state.http.stream(
                "POST", f"{CHAT_BASE_URL}/chat/completions",
                content=raw, headers=_headers(CHAT_API_KEY),
            ) as resp:
                if resp.status_code >= 400:
                    detail = (await resp.aread()).decode(errors="replace")[:600]
                    yield sse({"type": "error",
                               "message": f"SGLang returned {resp.status_code}. {detail}"})
                    return
                async for line in resp.aiter_lines():
                    if await request.is_disconnected():
                        return
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    thinking = delta.get("reasoning_content")
                    if thinking:
                        yield sse({"type": "reasoning", "text": thinking})
                    text = delta.get("content")
                    if text:
                        if first_token_ms is None:
                            first_token_ms = (time.monotonic() - started) * 1000
                        yield sse({"type": "delta", "text": text})
                    if choices[0].get("finish_reason") == "length":
                        yield sse({"type": "notice",
                                   "message": "Output stopped at the token limit."})
        except httpx.ReadTimeout:
            yield sse({"type": "error", "message":
                       f"No response from SGLang within {READ_TIMEOUT:g}s."})
            return
        except httpx.HTTPError as exc:
            yield sse({"type": "error",
                       "message": f"Cannot reach SGLang at {CHAT_BASE_URL}: {exc}"})
            return
        except asyncio.CancelledError:
            raise
        yield sse({
            "type": "done",
            "usage": usage,
            "elapsed_ms": round((time.monotonic() - started) * 1000),
            "ttft_ms": round(first_token_ms) if first_token_ms else None,
        })

    return StreamingResponse(
        relay(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform",
                 "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
    )


@app.post("/api/images")
async def images(req: ImageRequest) -> JSONResponse:
    if not IMAGE_BASE_URL:
        return JSONResponse(
            {"error": "Image generation is off. Set IMAGE_BASE_URL to an SGLang Diffusion server."},
            status_code=501,
        )
    body: dict[str, Any] = {
        "model": req.model or IMAGE_MODEL,
        "prompt": req.prompt,
        "n": req.n,
        "size": req.size,
        "response_format": "b64_json",
    }
    if req.seed is not None:
        body["seed"] = req.seed
    started = time.monotonic()
    try:
        resp = await app.state.http.post(
            f"{IMAGE_BASE_URL}/images/generations",
            json=body, headers=_headers(IMAGE_API_KEY), timeout=IMAGE_TIMEOUT,
        )
    except httpx.HTTPError as exc:
        return JSONResponse(
            {"error": f"Cannot reach the image server at {IMAGE_BASE_URL}: {exc}"}, status_code=502)
    if resp.status_code >= 400:
        return JSONResponse(
            {"error": f"Image server returned {resp.status_code}. "
                      f"{resp.text[:400]}"}, status_code=resp.status_code)

    data = resp.json().get("data", [])
    out: list[dict[str, str]] = []
    for item in data:
        if item.get("b64_json"):
            out.append({"url": f"data:image/png;base64,{item['b64_json']}"})
        elif item.get("url"):
            out.append({"url": item["url"]})
    return JSONResponse({
        "images": out,
        "model": body["model"],
        "elapsed_ms": round((time.monotonic() - started) * 1000),
    })


# --------------------------------------------------------------- static -----

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
