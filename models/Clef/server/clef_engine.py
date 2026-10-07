"""Clef serving engine.

Wraps the reference release code (``joint_schema_model.py`` shipped with the model)
and adds the serving optimizations measured in this project:

* ``CachedEncoder``   - token-identical to ``encode_record`` for text records, but the
                        schema/prefix/suffix tokenization is cached (schemas repeat).
* ``ClefEngine.run``  - one padded backbone forward for a micro-batch of records.
                        ``mask_mode="none"`` drops the attention mask for right-padded
                        text batches (pads trail every sequence, positions are an arange,
                        and every layer is causal, so valid tokens are unaffected); this
                        lets SDPA use the causal flash kernel instead of a 4D mask.
                        ``PAD_MULTIPLE`` (e.g. 16) rounds the padded length up, which is equally
                        exact and stops divisibility-specialized Triton kernels from recompiling.
* ``fast_head_forward`` - vectorized JointSchemaHead (no per-record python loop, no
                        ``.item()`` syncs), numerically equivalent up to bf16 reduction order.
* ``MicroBatcher``    - length-aware dynamic batching on a dedicated GPU thread.
* optional FP8 (torchao dynamic activation + weight) for the backbone decoder linears.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import sys
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

MODEL_PATH = os.environ.get("MODEL_PATH", "/cache/models/clef")
if MODEL_PATH not in sys.path:
    sys.path.insert(0, MODEL_PATH)
import joint_schema_model as jsm  # noqa: E402  (reference code shipped with the weights)


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


# --------------------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------------------


@dataclass
class SchemaEncoding:
    schema_ids: list[int]
    questions: tuple[jsm.EncodedQuestion, ...]  # spans relative to schema start


class CachedEncoder:
    """Token-identical replacement for ``jsm.encode_record`` with schema caching."""

    def __init__(self, tokenizer: Any, processor: Any, max_length: int = 16384, cache_size: int = 4096):
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.cache_size = cache_size
        self.prefix_ids = jsm._tokens(
            tokenizer,
            f"<|im_start|>system\n{jsm.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n",
        )
        self.suffix_ids = jsm._tokens(
            tokenizer,
            "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
        )
        self._cache: OrderedDict[str, SchemaEncoding] = OrderedDict()
        self._lock = threading.Lock()
        self._local = threading.local()
        self.hits = 0
        self.misses = 0

    def _tok(self) -> Any:
        """Per-thread tokenizer copy (fast tokenizers are not safe to share across threads)."""
        tok = getattr(self._local, "tok", None)
        if tok is None:
            tok = copy.deepcopy(self.tokenizer)
            self._local.tok = tok
        return tok

    def _schema(self, questions: dict[str, Any]) -> SchemaEncoding:
        key = json.dumps(questions, ensure_ascii=False, default=str)
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                self.hits += 1
                return hit
        # Derive the schema encoding from the reference encoder itself (empty state)
        # so the cached tokens are guaranteed identical to encode_record.
        ref = jsm.encode_record(self._tok(), {"state": "", "questions": questions}, max_length=1 << 30)
        offset = len(self.prefix_ids)
        assert list(ref.input_ids[:offset]) == self.prefix_ids
        schema_ids = list(ref.input_ids[offset : len(ref.input_ids) - len(self.suffix_ids)])
        rel = tuple(
            jsm.EncodedQuestion(
                question_id=q.question_id,
                question_type=q.question_type,
                question_span=(q.question_span[0] - offset, q.question_span[1] - offset),
                option_spans=tuple((s - offset, e - offset) for s, e in q.option_spans),
                option_ids=q.option_ids,
            )
            for q in ref.questions
        )
        enc = SchemaEncoding(schema_ids, rel)
        with self._lock:
            self.misses += 1
            self._cache[key] = enc
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return enc

    def encode(self, record: dict[str, Any]) -> tuple[jsm.EncodedRecord, bool]:
        """Returns (encoded record, state_was_truncated)."""
        if record.get("images") or record.get("videos"):
            enc = jsm.encode_record(self.tokenizer, record, max_length=self.max_length, processor=self.processor)
            return enc, False
        schema = self._schema(record["questions"])
        state_ids = jsm._tokens(self._tok(), jsm.render(record["state"]))
        fixed = len(self.prefix_ids) + len(schema.schema_ids) + len(self.suffix_ids)
        if fixed > self.max_length:
            raise ValueError(f"schema requires {fixed} tokens before state; maximum is {self.max_length}")
        allowed = self.max_length - fixed
        truncated = len(state_ids) > allowed
        state_ids = state_ids[:allowed]
        off = len(self.prefix_ids) + len(state_ids)
        questions = tuple(
            jsm.EncodedQuestion(
                question_id=q.question_id,
                question_type=q.question_type,
                question_span=(q.question_span[0] + off, q.question_span[1] + off),
                option_spans=tuple((s + off, e + off) for s, e in q.option_spans),
                option_ids=q.option_ids,
            )
            for q in schema.questions
        )
        input_ids = tuple(self.prefix_ids + state_ids + schema.schema_ids + self.suffix_ids)
        return jsm.EncodedRecord(input_ids=input_ids, questions=questions, record_id=str(record.get("id", "unknown"))), truncated


# --------------------------------------------------------------------------------------
# Vectorized joint schema head
# --------------------------------------------------------------------------------------


@dataclass
class HeadPlan:
    B: int
    T: int
    nq: int
    no: int
    max_q: int
    max_o: int
    span_tok: torch.Tensor  # flat b*T+t index of every token of every span (questions, then options)
    span_id: torch.Tensor
    span_cnt: torch.Tensor  # float32 [nq+no]
    lex_tok: torch.Tensor  # token ids inside option spans
    lex_id: torch.Tensor
    lex_cnt: torch.Tensor  # float32 [no]
    last_idx: torch.Tensor  # flat index of each record's last valid token
    q_batch: torch.Tensor
    q_slot: torch.Tensor
    q_type: torch.Tensor
    o_batch: torch.Tensor
    o_slot: torch.Tensor
    o_q: torch.Tensor
    mem_pad: torch.Tensor | None  # [B,T] True = padding
    q_pad: torch.Tensor | None  # [B,max_q] True = padding
    option_counts: list[int]  # per question, in order


def build_plan(records: list[jsm.EncodedRecord], lengths: list[int], T: int, device: torch.device) -> HeadPlan:
    B = len(records)
    q_batch, q_slot, q_type = [], [], []
    o_batch, o_slot, o_q = [], [], []
    q_spans, o_spans, lex_chunks = [], [], []
    option_counts = []
    max_q = max_o = 0
    nq = 0
    for b, rec in enumerate(records):
        slot = 0
        base = b * T
        for qi, q in enumerate(rec.questions):
            q_batch.append(b)
            q_slot.append(qi)
            q_type.append(q.question_type)
            q_spans.append((base + q.question_span[0], base + q.question_span[1]))
            for s, e in q.option_spans:
                o_batch.append(b)
                o_slot.append(slot)
                o_q.append(nq)
                o_spans.append((base + s, base + e))
                lex_chunks.append(rec.input_ids[s:e])
                slot += 1
            option_counts.append(len(q.option_spans))
            nq += 1
        max_q = max(max_q, len(rec.questions))
        max_o = max(max_o, slot)
    spans = q_spans + o_spans
    span_len = np.array([e - s for s, e in spans], dtype=np.int64)
    span_tok = np.concatenate([np.arange(s, e, dtype=np.int64) for s, e in spans])
    span_id = np.repeat(np.arange(len(spans), dtype=np.int64), span_len)
    lex_len = np.array([len(c) for c in lex_chunks], dtype=np.int64)
    lex_tok = np.fromiter((t for c in lex_chunks for t in c), dtype=np.int64, count=int(lex_len.sum()))
    lex_id = np.repeat(np.arange(len(lex_chunks), dtype=np.int64), lex_len)
    last_idx = np.array([b * T + lengths[b] - 1 for b in range(B)], dtype=np.int64)

    def t(x, dtype=torch.long):
        return torch.as_tensor(x, dtype=dtype).to(device, non_blocking=True)

    padded = any(L != T for L in lengths)
    mem_pad = None
    if padded:
        mem_pad = torch.arange(T, device=device)[None, :] >= t(lengths)[:, None]
    q_pad = None
    if any(len(r.questions) != max_q for r in records):
        qp = np.ones((B, max_q), dtype=bool)
        for b, r in enumerate(records):
            qp[b, : len(r.questions)] = False
        q_pad = t(qp, torch.bool)
    return HeadPlan(
        B=B, T=T, nq=nq, no=len(o_spans), max_q=max_q, max_o=max_o,
        span_tok=t(span_tok), span_id=t(span_id), span_cnt=t(np.maximum(span_len, 1), torch.float32),
        lex_tok=t(lex_tok), lex_id=t(lex_id), lex_cnt=t(np.maximum(lex_len, 1), torch.float32),
        last_idx=t(last_idx), q_batch=t(q_batch), q_slot=t(q_slot), q_type=t(q_type),
        o_batch=t(o_batch), o_slot=t(o_slot), o_q=t(o_q), mem_pad=mem_pad, q_pad=q_pad,
        option_counts=option_counts,
    )


def fast_head_forward(head: jsm.JointSchemaHead, hidden: torch.Tensor, plan: HeadPlan, lm_weight: torch.Tensor) -> torch.Tensor:
    """Vectorized JointSchemaHead.forward. Returns flat logits [num_options_total]."""
    B, T, Hd = hidden.shape
    dt, dev = hidden.dtype, hidden.device
    H = head.hidden_norm(hidden)
    flat = H.reshape(B * T, Hd)

    sums = torch.zeros(plan.nq + plan.no, Hd, device=dev, dtype=torch.float32)
    sums.index_add_(0, plan.span_id, flat.index_select(0, plan.span_tok).float())
    means = (sums / plan.span_cnt[:, None]).to(dt)
    qvec, ovec = means[: plan.nq], means[plan.nq :]

    lex = torch.zeros(plan.no, Hd, device=dev, dtype=torch.float32)
    lex.index_add_(0, plan.lex_id, lm_weight.index_select(0, plan.lex_tok).float())
    lex = (lex / plan.lex_cnt[:, None]).to(dt)

    glob = flat.index_select(0, plan.last_idx)  # [B, Hd]
    memory = head.memory_projection(H)  # [B, T, W]
    W = memory.shape[-1]

    oq = (
        head.option_context_projection(ovec)
        + head.option_lexical_projection(lex)
        + head.option_question_projection(qvec).index_select(0, plan.o_q)
    )
    Q = oq.new_zeros(B, plan.max_o, W)
    Q[plan.o_batch, plan.o_slot] = oq
    for layer in head.evidence_layers:
        mem_n = layer.memory_norm(memory)
        routed, _ = layer.attention(layer.query_norm(Q), mem_n, mem_n, key_padding_mask=plan.mem_pad, need_weights=False)
        Q = Q + routed
        Q = Q + layer.feedforward(layer.feedforward_norm(Q))
    routed = Q[plan.o_batch, plan.o_slot]  # [no, W]

    base_fields = head.question_projection(qvec)  # [nq, W]
    s = ((routed.float() * base_fields.index_select(0, plan.o_q).float()).sum(-1).to(dt) / math.sqrt(W)).float()
    mx = torch.full((plan.nq,), float("-inf"), device=dev).scatter_reduce(0, plan.o_q, s, reduce="amax", include_self=True)
    e = torch.exp(s - mx.index_select(0, plan.o_q))
    den = torch.zeros(plan.nq, device=dev).index_add_(0, plan.o_q, e)
    w = (e / den.index_select(0, plan.o_q)).to(dt)
    summ = torch.zeros(plan.nq, W, device=dev, dtype=torch.float32)
    summ.index_add_(0, plan.o_q, (w[:, None] * routed).float())
    fields = (
        base_fields
        + head.option_summary_norm(summ.to(dt))
        + head.global_projection(glob).index_select(0, plan.q_batch)
        + head.type_embedding(plan.q_type)
    )
    Fp = fields.new_zeros(B, plan.max_q, W)
    Fp[plan.q_batch, plan.q_slot] = fields
    for layer in head.layers:
        Fp = layer(Fp, memory, tgt_key_padding_mask=plan.q_pad, memory_key_padding_mask=plan.mem_pad)
    fields = head.field_norm(Fp[plan.q_batch, plan.q_slot])  # [nq, W]

    anchor = F.normalize(qvec + glob.index_select(0, plan.q_batch), dim=-1)
    lex_anchor = F.normalize(lex, dim=-1)
    prior_scale = head.prior_logit_scale.clamp(max=math.log(100.0)).exp()
    prior = prior_scale * (lex_anchor.float() * anchor.index_select(0, plan.o_q).float()).sum(-1).to(dt)
    opts = head.option_norm(routed)
    fo = fields.index_select(0, plan.o_q)
    cosine = F.cosine_similarity(fo, opts, dim=-1)
    features = torch.cat([fo, opts, fo * opts, torch.abs(fo - opts)], dim=-1)
    residual = head.residual_scorer(features).squeeze(-1)
    joint_scale = head.joint_logit_scale.clamp(max=math.log(100.0)).exp()
    joint = joint_scale * cosine + residual
    return prior + torch.sigmoid(head.residual_gate) * joint


# --------------------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------------------


def apply_fp8(text_model: torch.nn.Module, mode: str, min_features: int = 1024) -> int:
    """fp8_fast: server/fp8_linear.py (Triton per-token quant + rowwise _scaled_mm).
    nvfp4_mlp: server/nvfp4_linear.py NVFP4 for the fused MLP (needs FUSED mlp), fp8_fast elsewhere.
    fp8_rowwise / fp8_tensor: torchao Float8DynamicActivationFloat8WeightConfig (eager)."""
    if mode == "nvfp4_mlp":
        from fp8_linear import apply_fp8_fast
        from nvfp4_linear import apply_nvfp4_mlp

        n4 = apply_nvfp4_mlp(text_model)
        if n4 == 0:
            raise ValueError("nvfp4_mlp needs the fused MLP (FUSED=1 or fused=mlp)")
        return n4 + apply_fp8_fast(text_model, min_features=min_features)
    if mode == "fp8_fast":
        from fp8_linear import apply_fp8_fast

        return apply_fp8_fast(text_model, min_features=min_features)
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow, PerTensor, quantize_

    granularity = PerRow() if mode == "fp8_rowwise" else PerTensor()
    count = 0

    def filter_fn(module: torch.nn.Module, fqn: str) -> bool:
        nonlocal count
        ok = (
            isinstance(module, torch.nn.Linear)
            and fqn.startswith("layers.")
            and module.in_features >= min_features
            and module.out_features >= min_features
        )
        count += int(ok)
        return ok

    quantize_(text_model, Float8DynamicActivationFloat8WeightConfig(granularity=granularity), filter_fn=filter_fn)
    return count


@dataclass
class BatchTiming:
    batch_size: int
    padded_tokens: int
    valid_tokens: int
    backbone_ms: float
    head_ms: float
    total_ms: float


class ClefEngine:
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        device: str = "cuda",
        attn_impl: str = os.environ.get("ATTN_IMPL", "sdpa"),
        quant: str = os.environ.get("QUANT", "none"),
        head_impl: str = os.environ.get("HEAD_IMPL", "fast"),
        mask_mode: str = os.environ.get("MASK_MODE", "none"),
        max_length: int = env_int("MAX_LENGTH", 16384),
        compile_mode: str = os.environ.get("COMPILE", "none"),
        fused: str = os.environ.get("FUSED", "none"),
    ):
        t0 = time.time()
        self.device = torch.device(device)
        self.model, self.processor = jsm.load_release_model(model_path, device=device, attn_implementation=attn_impl)
        self.load_s = time.time() - t0
        self.tokenizer = self.processor.tokenizer
        self.pad_id = self.tokenizer.pad_token_id
        base = self.model.language_model
        self.full_model = base.model
        self.text_model = base.model.language_model
        self.lm_weight = base.get_output_embeddings().weight
        self.head = self.model.head
        self.head_impl = head_impl
        self.mask_mode = mask_mode
        self.quant = "none"
        self.attn_impl = attn_impl
        self.encoder = CachedEncoder(self.tokenizer, self.processor, max_length=max_length)
        self.fused = "none"
        self.fused_info: dict[str, int] = {}
        self.quantized_linears = 0
        self.apply_fused(fused)
        self.apply_quant(quant)
        self.autotune_pinned: list[str] = []
        if os.environ.get("PIN_AUTOTUNE", "1") == "1":
            import fast_patches

            self.autotune_pinned = fast_patches.pin_autotune_keys()
        self.compile_mode = compile_mode
        if compile_mode != "none":
            for layer in self.text_model.layers:
                layer.forward = torch.compile(layer.forward, mode=None if compile_mode == "default" else compile_mode, dynamic=True)
        self.pad_multiple = max(1, env_int("PAD_MULTIPLE", 1))
        torch.cuda.synchronize()
        self.config_summary = dict(
            attn_impl=attn_impl, quant=self.quant, head_impl=head_impl, mask_mode=mask_mode,
            max_length=max_length, compile=compile_mode, quantized_linears=self.quantized_linears,
            fused=self.fused, autotune_pinned=len(self.autotune_pinned), pad_multiple=self.pad_multiple,
            load_s=round(self.load_s, 1), gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
        )

    def apply_fused(self, fused: str) -> None:
        """Fused prefill kernels (server/fast_patches.py), in place and one-way. Must precede FP8 so
        the merged projections are quantized as single linears. "1" = gdn,rmsnorm,mlp."""
        if fused in ("", "0", "none"):
            return
        if self.quant != "none":
            raise ValueError("fused patches must be applied before quantization")
        import fast_patches

        parts = {"gdn", "rmsnorm", "mlp"} if fused in ("1", "all") else set(fused.split(","))
        for p in sorted(parts - set(self.fused_info)):
            self.fused_info[p] = getattr(fast_patches, f"patch_{p}")(self.text_model)
        self.fused = ",".join(sorted(self.fused_info))

    def apply_quant(self, quant: str) -> None:
        """FP8 for the decoder linears, in place and one-way (see apply_fp8)."""
        if quant in ("", "none") or quant == self.quant:
            return
        if self.quant != "none":
            raise ValueError("already quantized; restart to change quantization")
        self.quantized_linears = apply_fp8(self.text_model, quant)
        self.quant = quant
        torch.cuda.empty_cache()

    # ---- core forward -------------------------------------------------------------
    @torch.inference_mode()
    def run(self, records: list[jsm.EncodedRecord]) -> tuple[list[list[np.ndarray]], BatchTiming]:
        t_start = time.perf_counter()
        if any(r.media for r in records):
            if len(records) != 1:
                raise ValueError("media records must be run one at a time")
            return self._run_reference(records, t_start)
        lengths = [len(r.input_ids) for r in records]
        B, T = len(records), max(lengths)
        if self.pad_multiple > 1 and self.head_impl == "fast":
            # Exact (trailing pads never reach valid tokens; the head masks them via plan.mem_pad). Keeps T and
            # B*T multiples of 16, so Triton kernels that specialize on integer divisibility (fla's causal_conv1d
            # takes B and T as plain ints) see no new variants after warmup -> no JIT compiles on live traffic.
            T = -(-T // self.pad_multiple) * self.pad_multiple
        ids_np = np.full((B, T), self.pad_id, dtype=np.int64)
        for i, r in enumerate(records):
            ids_np[i, : lengths[i]] = r.input_ids
        ids = torch.from_numpy(ids_np).to(self.device, non_blocking=True)
        need_mask = self.mask_mode == "mask" or self.head_impl == "reference"
        mask = None
        if need_mask:
            mask = (torch.arange(T)[None, :] < torch.tensor(lengths)[:, None]).long().to(self.device)
        plan = build_plan(records, lengths, T, self.device) if self.head_impl == "fast" else None
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        ev[0].record()
        out = self.text_model(
            input_ids=ids,
            attention_mask=mask if self.mask_mode == "mask" else None,
            use_cache=False,
            return_dict=True,
        )
        hidden = out.last_hidden_state
        ev[1].record()
        if self.head_impl == "fast":
            flat = fast_head_forward(self.head, hidden, plan, self.lm_weight)
            counts = plan.option_counts
        else:
            res = self.head(hidden, ids, mask, records, self.lm_weight)
            flat = torch.cat([torch.cat(r) for r in res])
            counts = [len(q.option_spans) for r in records for q in r.questions]
        ev[2].record()
        flat = flat.float().cpu().numpy()
        out_logits = self._split(records, flat, counts)
        timing = BatchTiming(
            batch_size=B, padded_tokens=B * T, valid_tokens=sum(lengths),
            backbone_ms=ev[0].elapsed_time(ev[1]), head_ms=ev[1].elapsed_time(ev[2]),
            total_ms=(time.perf_counter() - t_start) * 1000,
        )
        return out_logits, timing

    def _run_reference(self, records, t_start):
        batch = jsm.collate_records(records, self.pad_id, self.device)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        ev[0].record()
        res = self.model(batch)
        ev[1].record()
        flat = torch.cat([torch.cat(r) for r in res]).float().cpu().numpy()
        counts = [len(q.option_spans) for r in records for q in r.questions]
        L = len(records[0].input_ids)
        timing = BatchTiming(1, L, L, ev[0].elapsed_time(ev[1]), 0.0, (time.perf_counter() - t_start) * 1000)
        return self._split(records, flat, counts), timing

    @staticmethod
    def _split(records, flat: np.ndarray, counts: list[int]) -> list[list[np.ndarray]]:
        out, pos, k = [], 0, 0
        for r in records:
            per_q = []
            for _ in r.questions:
                c = counts[k]
                per_q.append(flat[pos : pos + c])
                pos += c
                k += 1
            out.append(per_q)
        return out

    def run_safe(self, records: list[jsm.EncodedRecord]) -> tuple[list[list[np.ndarray]], list[BatchTiming]]:
        """run() with OOM fallback: split the batch in half and retry."""
        try:
            logits, timing = self.run(records)
            return logits, [timing]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(records) == 1:
                raise
            mid = len(records) // 2
            a, ta = self.run_safe(records[:mid])
            b, tb = self.run_safe(records[mid:])
            return a + b, ta + tb

    # ---- response formatting ------------------------------------------------------
    @staticmethod
    def answers(request: dict[str, Any], record: jsm.EncodedRecord, logits: list[np.ndarray]) -> dict[str, Any]:
        questions = request["questions"]
        out = {}
        for q, ql in zip(record.questions, logits):
            z = ql.astype(np.float64)
            p = np.exp(z - z.max())
            p /= p.sum()
            out[q.question_id] = jsm.systemone_answer(questions[q.question_id], dict(zip(q.option_ids, p.tolist())))
        return out


def validate_request(request: dict[str, Any]) -> None:
    """Same validation as jsm.systemone."""
    questions = request.get("questions")
    if not isinstance(request.get("model"), str) or "state" not in request:
        raise ValueError("model and state are required")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("at least one question is required")
    for question_id, question in questions.items():
        if question.get("type") not in jsm.QUESTION_TYPES:
            raise ValueError(f"{question_id}: type must be noul, choice, or score")
        if question["type"] != "noul" and not question.get("criteria"):
            raise ValueError(f"{question_id}: criteria must not be empty")


# --------------------------------------------------------------------------------------
# Micro-batching
# --------------------------------------------------------------------------------------


@dataclass
class _Pending:
    record: jsm.EncodedRecord
    n: int
    media: bool
    fut: asyncio.Future
    loop: asyncio.AbstractEventLoop
    t_enq: float
    info: dict = field(default_factory=dict)


def _resolve(fut: asyncio.Future, value: Any, exc: BaseException | None) -> None:
    if fut.done():
        return
    if exc is not None:
        fut.set_exception(exc)
    else:
        fut.set_result(value)


class MicroBatcher:
    """Length-aware dynamic batching. A single GPU worker thread repeatedly takes the
    oldest queued request as the anchor and greedily adds the queued requests closest
    in length while padded_tokens = max_len * count <= max_batch_tokens and
    valid_tokens / padded_tokens >= min_pad_eff."""

    def __init__(self, engine: ClefEngine, max_batch_tokens: int, max_batch_size: int, max_wait_ms: float,
                 min_pad_eff: float = float(os.environ.get("MIN_PAD_EFF", "0.8"))):
        self.engine = engine
        self.max_batch_tokens = max_batch_tokens
        self.max_batch_size = max_batch_size
        self.max_wait = max_wait_ms / 1000.0
        self.min_pad_eff = min_pad_eff
        self.queue: list[_Pending] = []
        self.cv = threading.Condition()
        self.gpu_lock = threading.Lock()
        self.stats = dict(batches=0, requests=0, valid_tokens=0, padded_tokens=0, backbone_ms=0.0, head_ms=0.0, gpu_ms=0.0, oom_splits=0)
        self.recent = deque(maxlen=2048)
        self.thread = threading.Thread(target=self._loop, name="clef-gpu", daemon=True)
        self.thread.start()

    def submit(self, record: jsm.EncodedRecord, loop: asyncio.AbstractEventLoop) -> asyncio.Future:
        fut = loop.create_future()
        p = _Pending(record, len(record.input_ids), bool(record.media), fut, loop, time.perf_counter())
        with self.cv:
            self.queue.append(p)
            self.cv.notify()
        return fut

    def queue_depth(self) -> int:
        with self.cv:
            return len(self.queue)

    def _take(self) -> list[_Pending]:
        anchor = self.queue[0]
        if anchor.media:
            batch = [anchor]
        else:
            cands = sorted((p for p in self.queue[1:] if not p.media), key=lambda p: abs(p.n - anchor.n))
            batch, max_len, valid = [anchor], anchor.n, anchor.n
            for p in cands:
                if len(batch) >= self.max_batch_size:
                    break
                new_max = max(max_len, p.n)
                padded = new_max * (len(batch) + 1)
                if padded > self.max_batch_tokens or (valid + p.n) / padded < self.min_pad_eff:
                    continue
                batch.append(p)
                max_len = new_max
                valid += p.n
        chosen = {id(p) for p in batch}
        self.queue = [p for p in self.queue if id(p) not in chosen]
        return batch

    def _loop(self) -> None:
        while True:
            with self.cv:
                while not self.queue:
                    self.cv.wait()
                if self.max_wait > 0:
                    deadline = self.queue[0].t_enq + self.max_wait
                    while True:
                        queued = sum(p.n for p in self.queue)
                        remaining = deadline - time.perf_counter()
                        if remaining <= 0 or queued >= self.max_batch_tokens or len(self.queue) >= self.max_batch_size:
                            break
                        self.cv.wait(timeout=remaining)
                batch = self._take()
            t_take = time.perf_counter()
            try:
                with self.gpu_lock:
                    logits, timings = self.engine.run_safe([p.record for p in batch])
                t_done = time.perf_counter()
                st = self.stats
                st["batches"] += len(timings)
                st["requests"] += len(batch)
                st["oom_splits"] += len(timings) - 1
                for tm in timings:
                    st["valid_tokens"] += tm.valid_tokens
                    st["padded_tokens"] += tm.padded_tokens
                    st["backbone_ms"] += tm.backbone_ms
                    st["head_ms"] += tm.head_ms
                    st["gpu_ms"] += tm.total_ms
                    self.recent.append((t_done, tm.batch_size, tm.valid_tokens, tm.padded_tokens, tm.backbone_ms, tm.head_ms))
                for p, lg in zip(batch, logits):
                    p.info = dict(queue_ms=(t_take - p.t_enq) * 1000, compute_ms=(t_done - t_take) * 1000, batch_size=len(batch))
                    p.loop.call_soon_threadsafe(_resolve, p.fut, (lg, p.info), None)
            except Exception as exc:  # noqa: BLE001
                for p in batch:
                    p.loop.call_soon_threadsafe(_resolve, p.fut, None, exc)

    def snapshot(self) -> dict[str, Any]:
        st = dict(self.stats)
        b = max(st["batches"], 1)
        st["avg_batch_size"] = st["requests"] / b
        st["avg_valid_tokens_per_batch"] = st["valid_tokens"] / b
        st["padding_efficiency"] = st["valid_tokens"] / max(st["padded_tokens"], 1)
        st["queue_depth"] = self.queue_depth()
        st["schema_cache_hits"] = self.engine.encoder.hits
        st["schema_cache_misses"] = self.engine.encoder.misses
        return st
