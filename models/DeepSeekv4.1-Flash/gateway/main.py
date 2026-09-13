"""
DeepSeek-V4.1-Flash API Gateway using deepseek-recipe
Connects universal API protocols (OpenAI, Anthropic, Responses) to SGLang inference engine.
"""

import json
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from time import time
from uuid import uuid4

import httpx
from anyio import CancelScope
from deepseek_recipe import (
    THINKING_END_TOKEN,
    THINKING_START_TOKEN,
    CalcResizeError,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ConversationRequest,
    ConversionError,
    ConversionOptions,
    DeepseekV4Encoding,
    DeepseekV41Encoding,
    ImageError,
    ImageInfo,
    ImageQuota,
    ImageResolver,
    InferenceChunk,
    InferenceFinishReason,
    InferenceOptions,
    MessagesRequest,
    MessagesResponse,
    OpenCvImagePreprocessor,
    ReqwestImageFetcher,
    ResponsesRequest,
    ResponsesResponse,
    StreamProcessor,
)
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool
from starlette.types import Receive, Scope, Send

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("deepseek_recipe_gateway")

SGLANG_UPSTREAM_URL = os.getenv(
    "SGLANG_UPSTREAM_URL",
    "http://sglang-dsv41-flash-svc.default.svc.cluster.local:30000",
).rstrip("/")

PROTOCOL_TYPES = {
    "chat_completions": (ChatCompletionRequest, ChatCompletionResponse),
    "responses": (ResponsesRequest, ResponsesResponse),
    "messages": (MessagesRequest, MessagesResponse),
}


@dataclass(frozen=True)
class PreparedRequest:
    """The gateway's rendered input and protocol response settings."""

    protocol: str
    conversation_request: ConversationRequest
    prompt: str
    images: list[ImageInfo]
    image_token_adjustment: int
    include_usage: bool = False
    custom_tool_names: frozenset[str] = frozenset()

    @property
    def model(self) -> str | None:
        return self.conversation_request.model

    @property
    def stream(self) -> bool:
        return self.conversation_request.stream

    @property
    def inference_options(self) -> InferenceOptions:
        return self.conversation_request.inference_options


Inference = Callable[[PreparedRequest], AsyncIterator[InferenceChunk]]


class RequestError(Exception):
    """A request preparation failure with an HTTP status chosen by the gateway."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def prepare_request(
    protocol: str,
    body: bytes,
    *,
    model_version: str = "v4.1",
    options: ConversionOptions | None = None,
) -> PreparedRequest:
    """Convert protocol request, render DeepSeek-V4.1 prompt, and resolve images."""
    request_type, _ = PROTOCOL_TYPES[protocol]
    try:
        request = request_type(body)
    except ValueError as error:
        raise RequestError(str(error)) from error

    include_usage = request.include_usage() if protocol == "chat_completions" else False
    custom_tool_names = (
        frozenset(request.custom_tool_names()) if protocol == "responses" else frozenset()
    )
    converted = request.convert(options if options is not None else ConversionOptions())
    encoding_type = {"v4": DeepseekV4Encoding, "v4.1": DeepseekV41Encoding}[model_version]
    rendered = encoding_type().render_conversation(converted.conversation)

    images = []
    image_token_adjustment = 0
    if rendered.image_sources:
        try:
            resolver = ImageResolver(ReqwestImageFetcher(), OpenCvImagePreprocessor())
            multimodal = resolver.resolve(rendered.image_sources, ImageQuota())
        except ImageError as error:
            status_code = 500 if error.kind in {"Client", "TokenBudget"} else 400
            raise RequestError(str(error), status_code) from error
        images = multimodal.images
        try:
            image_token_adjustment = multimodal.image_token_adjustment()
        except CalcResizeError as error:
            raise RequestError(str(error), 500) from error

    return PreparedRequest(
        protocol,
        converted,
        rendered.prompt,
        images,
        image_token_adjustment,
        include_usage,
        custom_tool_names,
    )


class InferenceStreamingResponse(StreamingResponse):
    """Close the response iterator after a disconnect or a failed send."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            with CancelScope(shield=True):
                await self.body_iterator.aclose()


MOCK_INFERENCE = os.getenv("MOCK_INFERENCE", "0").lower() in ("1", "true", "yes")


async def sglang_inference(request: PreparedRequest) -> AsyncIterator[InferenceChunk]:
    """Execute inference against the upstream SGLang serving cluster, or mock if configured."""
    opts = request.inference_options
    max_tokens = opts.max_tokens if opts.max_tokens is not None else 2048
    temperature = opts.temperature if opts.temperature is not None else 0.6
    top_p = opts.top_p if opts.top_p is not None else 0.95

    # Check for mock inference mode for pipeline testing
    if MOCK_INFERENCE:
        logger.info("Executing mock inference via deepseek-recipe stream processor...")
        import asyncio
        yield InferenceChunk.ready()
        mock_response = (
            "Quantum entanglement is a physical phenomenon where two or more particles become "
            "intertwined such that the quantum state of each particle cannot be described independently "
            "of the others, regardless of the distance separating them. Any measurement performed on one "
            "particle instantaneously determines the outcome of a measurement on the other."
        )
        words = mock_response.split(" ")
        for i, word in enumerate(words):
            token = word if i == 0 else " " + word
            yield InferenceChunk.text(token, content_tokens=1)
            await asyncio.sleep(0.02)
        yield InferenceChunk.finish(finish_reason=InferenceFinishReason.Stop)
        return

    prompt = request.prompt
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": ["<｜end of sentence｜>"],
        },
        "stream": True,
    }

    if request.images:
        import base64
        # Replace the deepseek-recipe placeholder with the token expected by SGLang tokenizer
        payload["text"] = prompt.replace("<｜image｜>", "<｜deepseek_image｜>")
        payload["image_data"] = [
            f"data:image/webp;base64,{base64.b64encode(img.data).decode('utf-8')}"
            for img in request.images
        ]
        logger.info(f"Forwarding request with {len(request.images)} image(s) to SGLang upstream")

    url = f"{SGLANG_UPSTREAM_URL}/generate"
    timeout = httpx.Timeout(600.0, connect=5.0)

    client = httpx.AsyncClient(timeout=timeout)
    try:
        try:
            req = client.build_request("POST", url, json=payload)
            resp = await client.send(req, stream=True)
            resp.raise_for_status()
        except Exception as exc:
            logger.error(f"Cannot connect to upstream SGLang ({url}): {exc}")
            yield InferenceChunk.ready()
            error_msg = (
                f"\n\n[DeepSeek-Recipe Gateway Error]: Upstream SGLang engine unreachable at {url}.\n"
                f"Details: {exc}\n\n"
                f"Troubleshooting:\n"
                f"1. Deploy the SGLang GPU server in GKE (sglang-dsv41-flash-svc:30000).\n"
                f"2. Or set MOCK_INFERENCE=1 in the gateway Deployment to test deepseek-recipe translation."
            )
            yield InferenceChunk.text(error_msg, content_tokens=1)
            yield InferenceChunk.finish(finish_reason=InferenceFinishReason.Stop)
            return

        yield InferenceChunk.ready()
        last_len = 0
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                full_text = chunk.get("text", "")
                if len(full_text) > last_len:
                    delta = full_text[last_len:]
                    last_len = len(full_text)
                    yield InferenceChunk.text(delta, content_tokens=1)
                elif len(full_text) < last_len:
                    last_len = len(full_text)
            except json.JSONDecodeError:
                continue
        await resp.aclose()
    except Exception as exc:
        logger.error(f"Stream error during SGLang inference: {exc}")
        yield InferenceChunk.text(f"\n\n[Upstream Stream Interrupted: {exc}]", content_tokens=1)
    finally:
        await client.aclose()

    yield InferenceChunk.finish(finish_reason=InferenceFinishReason.Stop)


def sse_frame(event: str | None, data: str) -> str:
    """Format one protocol chunk or completion marker for HTTP streaming."""
    prefix = f"event: {event}\n" if event is not None else ""
    return f"{prefix}data: {data}\n\n"


async def response_body(
    request: PreparedRequest, infer: Inference
) -> AsyncIterator[str]:
    request_type, response_type = PROTOCOL_TYPES[request.protocol]
    response_id = str(uuid4())
    model = request.model or "deepseek-ai/DeepSeek-V4.1-Flash"
    generator = request_type.chunk_generator(request.conversation_request, response_id, model)
    if request.protocol == "chat_completions":
        generator = generator.with_include_usage(request.include_usage)
    elif request.protocol == "responses":
        generator = generator.with_custom_tool_names(request.custom_tool_names)

    processor = StreamProcessor(generator, request.conversation_request.parsing_options)
    response = None if request.stream else response_type(response_id, model, int(time()), 0, 0)
    inference = None

    try:
        inference = aiter(infer(request))
        while True:
            try:
                chunk = await anext(inference)
            except StopAsyncIteration:
                chunks = processor.finish()
            else:
                chunks = processor.push(chunk)
            for output in chunks:
                if request.stream:
                    yield sse_frame(response_type.chunk_event_type(output), output.to_json())
                else:
                    response.append(output)
            if processor.finished:
                break
        if request.stream:
            done = response_type.done_message()
            if done is not None:
                yield sse_frame(None, done)
        else:
            yield response.to_json()
    finally:
        try:
            close = getattr(inference, "aclose", None)
            if close is not None:
                with CancelScope(shield=True):
                    await close()
        finally:
            processor.close()


def create_app(
    infer: Inference = sglang_inference, *, options: ConversionOptions | None = None
) -> FastAPI:
    """Create FastAPI application with deepseek-recipe protocol translation."""
    app = FastAPI(
        title="deepseek-recipe-gateway",
        description="Official DeepSeek-V4.1 protocol gateway fronting SGLang serving engine",
        version="0.1.0",
    )

    @app.exception_handler(ConversionError)
    async def conversion_error_handler(_request: Request, exc: ConversionError) -> Response:
        return Response(
            content=exc.body, status_code=exc.status_code, media_type="application/json"
        )

    @app.exception_handler(RequestError)
    async def request_error_handler(_request: Request, exc: RequestError) -> Response:
        error_type = "internal_error" if exc.status_code >= 500 else "invalid_request_error"
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {
                "message": str(exc), "type": error_type, "param": None, "code": error_type,
            }},
        )

    @app.get("/health")
    async def health() -> dict:
        upstream_status = "unknown"
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{SGLANG_UPSTREAM_URL}/health")
                upstream_status = f"HTTP {res.status_code}"
        except Exception as e:
            upstream_status = f"unreachable: {e}"
        return {
            "status": "ok",
            "gateway": "healthy",
            "upstream_sglang": upstream_status,
            "upstream_url": SGLANG_UPSTREAM_URL,
        }

    @app.get("/v1/models")
    async def list_models() -> dict:
        models_data = []
        created_time = int(time())
        max_model_len = 1048576
        upstream_info = None

        # Fetch detailed upstream model_info if available
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                info_res = await client.get(f"{SGLANG_UPSTREAM_URL}/model_info")
                if info_res.status_code == 200:
                    upstream_info = info_res.json()
        except Exception as e:
            logger.warning(f"Could not retrieve upstream /model_info: {e}")

        # Fetch models from upstream SGLang
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                models_res = await client.get(f"{SGLANG_UPSTREAM_URL}/v1/models")
                if models_res.status_code == 200:
                    data = models_res.json().get("data", [])
                    if data:
                        created_time = data[0].get("created", created_time)
                        max_model_len = data[0].get("max_model_len", max_model_len)
                        for item in data:
                            if item not in models_data:
                                models_data.append(item)
        except Exception as e:
            logger.warning(f"Could not retrieve upstream /v1/models: {e}")

        # Include canonical model IDs and aliases
        alias_ids = [
            "deepseek-ai/DeepSeek-V4.1-Flash",
            "DeepSeek-V4.1-Flash",
            "/models/DeepSeek-V4.1-Flash",
        ]
        existing_ids = {m["id"] for m in models_data}
        for aid in alias_ids:
            if aid not in existing_ids:
                entry = {
                    "id": aid,
                    "object": "model",
                    "created": created_time,
                    "owned_by": "deepseek",
                    "root": aid,
                    "parent": None,
                    "max_model_len": max_model_len,
                }
                if upstream_info:
                    entry["model_info"] = {
                        "model_type": upstream_info.get("model_type"),
                        "architectures": upstream_info.get("architectures"),
                        "reasoning_parser": upstream_info.get("reasoning_parser"),
                        "tool_call_parser": upstream_info.get("tool_call_parser"),
                        "has_image_understanding": upstream_info.get("has_image_understanding"),
                    }
                models_data.append(entry)

        return {"object": "list", "data": models_data}

    @app.get("/v1/models/{model_id:path}")
    async def retrieve_model(model_id: str) -> Response:
        alias_ids = {
            "deepseek-ai/DeepSeek-V4.1-Flash",
            "DeepSeek-V4.1-Flash",
            "/models/DeepSeek-V4.1-Flash",
            "models/DeepSeek-V4.1-Flash",
        }
        clean_id = model_id.strip()

        # Query upstream first
        upstream_data = None
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{SGLANG_UPSTREAM_URL}/v1/models/{clean_id}")
                if res.status_code == 200:
                    upstream_data = res.json()
        except Exception as e:
            logger.warning(f"Could not retrieve upstream /v1/models/{clean_id}: {e}")

        if upstream_data:
            return JSONResponse(status_code=200, content=upstream_data)

        # Match alias or substring
        if clean_id in alias_ids or "DeepSeek" in clean_id or "deepseek" in clean_id:
            upstream_info = None
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    info_res = await client.get(f"{SGLANG_UPSTREAM_URL}/model_info")
                    if info_res.status_code == 200:
                        upstream_info = info_res.json()
            except Exception:
                pass

            model_obj = {
                "id": clean_id,
                "object": "model",
                "created": int(time()),
                "owned_by": "deepseek",
                "root": clean_id,
                "parent": None,
                "max_model_len": 1048576,
            }
            if upstream_info:
                model_obj["model_info"] = upstream_info

            return JSONResponse(status_code=200, content=model_obj)

        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": f"The model '{model_id}' does not exist",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
        )

    @app.get("/model_info")
    @app.get("/v1/model_info")
    async def model_info() -> Response:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{SGLANG_UPSTREAM_URL}/model_info")
                return Response(
                    content=res.content,
                    status_code=res.status_code,
                    media_type="application/json",
                )
        except Exception as e:
            return JSONResponse(
                status_code=502,
                content={"error": f"Upstream SGLang unreachable: {e}"},
            )

    def api_handler(protocol: str) -> Callable[[Request], Awaitable[Response]]:
        async def handler(request: Request) -> Response:
            body = await request.body()
            prepared = await run_in_threadpool(prepare_request, protocol, body, options=options)
            output = response_body(prepared, infer)
            if prepared.stream:
                return InferenceStreamingResponse(output, media_type="text/event-stream")
            content = "".join([part async for part in output])
            return Response(content, media_type="application/json")

        return handler

    for path, protocol in (
        ("/v1/chat/completions", "chat_completions"),
        ("/v1/responses", "responses"),
        ("/v1/messages", "messages"),
    ):
        app.add_api_route(path, api_handler(protocol), methods=["POST"], name=protocol)

    return app


app = create_app()
