# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TTFT timing monkey patches for vLLM inference pipeline profiling.

Patches key methods across the full request lifecycle to emit per-stage
timing checkpoints with a ``[TTFT_TRACE]`` prefix.  All checkpoints use
``time.time()`` (wall clock) so logs from different processes can be
correlated.

Activate by importing this module (side-effect import):
    import vllm_ascend.patch.platform.patch_ttft_timing  # noqa

Or add it to ``platform/__init__.py`` to auto-activate.

Log format::

    [TTFT_TRACE] request_id=<id> stage=<stage_name> timestamp=<secs_since_epoch>
"""

from __future__ import annotations
import time
import uuid
from typing import Any
from vllm.logger import logger  # pre-configured vllm logger

logger.warning("TTFT: patch module loaded — 10 patches will be attempted")

# =============================================================================
#  Inline timing helpers (self-contained, no dependency on vllm source changes)
# =============================================================================


def _sync_device() -> None:
    """Flush pending NPU work so wall-clock timestamps are accurate."""
    try:
        import torch
        torch.npu.synchronize()
    except Exception:
        pass


def _ttft_log(request_id: object, stage: str) -> None:
    """Emit a TTFT timing checkpoint (always via stdout)."""
    ts = time.time()
    # Defensive: request_id may accidentally be a non-string object
    # (e.g. ProcessorInputs) if a wrapper has the wrong parameter order.
    try:
        if not isinstance(request_id, str):
            request_id = str(request_id)
    except Exception:
        request_id = f"<bad-req-id-{type(request_id).__name__}>"
    safe_req = request_id.replace("%", "%%")
    safe_stage = stage.replace("%", "%%")
    logger.info(
        "[TTFT_TRACE] request_id=%s stage=%s timestamp=%.6f",
        safe_req, safe_stage, ts,
    )


def _ttft_start(request_id: str, stage: str) -> None:
    _ttft_log(request_id, f"{stage}_start")


def _ttft_end(request_id: str, stage: str) -> None:
    _ttft_log(request_id, f"{stage}_end")


# =============================================================================
#  Patch: OpenAIServingChat.create_chat_completion  (API Server entry)
# =============================================================================
#
#  create_chat_completion is a regular coroutine (``async def`` with
#  ``return``).  It is called via ``await`` at api_router.py:55.  We wrap
#  it as a coroutine too, logging api_request_start before the original
#  begins executing.
#
#  render_chat_start / render_chat_end are covered by the
#  OpenAIServingRender.render_chat patch below.
# =============================================================================

try:
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

    _orig_create_chat_completion = OpenAIServingChat.create_chat_completion

    async def _patched_create_chat_completion(
        self: OpenAIServingChat, request: Any, raw_request: Any = None
    ) -> Any:
        req_id = request.request_id or f"req-{uuid.uuid4().hex[:12]}"
        # Stash the raw request_id so _patched_base_request_id (below)
        # can log the mapping when the chatcmpl-xxx id is generated.
        self._ttft_raw_req_id = req_id
        _ttft_start(req_id, "api_request")
        return await _orig_create_chat_completion(self, request, raw_request)

    # ---- _base_request_id (id mapping) ----
    # The original creates the chatcmpl-xxx request_id by calling
    # _base_request_id(raw_request, request.request_id).  We wrap it
    # to emit an id_mapped event, linking the raw UUID used in
    # API-server patches to the engine-side chatcmpl-xxx id.
    from vllm.entrypoints.openai.engine.serving import OpenAIServing

    _orig_base_request_id = OpenAIServing._base_request_id

    def _patched_base_request_id(
        self: OpenAIServing, raw_request: Any, default: str | None = None
    ) -> str | None:
        result = _orig_base_request_id(raw_request, default)
        # Use the stashed raw_req_id (set in _patched_create_chat_completion)
        # instead of `default` because `default` may be None or differ.
        raw_req_id = getattr(self, "_ttft_raw_req_id", default)
        if result and raw_req_id:
            _ttft_log(f"chatcmpl-{result}", f"id_mapped_from_{raw_req_id}")
        return result

    OpenAIServing._base_request_id = _patched_base_request_id

    OpenAIServingChat.create_chat_completion = _patched_create_chat_completion
    logger.warning("TTFT: patched OpenAIServingChat.create_chat_completion")
except ImportError as e:
    logger.warning("TTFT: OpenAIServingChat NOT available, skipping")

# =============================================================================
#  Patch: OpenAIServingRender.render_chat  (render + HF processor, API Server)
# =============================================================================
#
#  Called by OpenAIServingChat.render_chat_request() which does
#  engine/model checks then delegates here.  The ``request.request_id``
#  is the user-supplied or server-generated UUID, *not* the final
#  ``chatcmpl-xxx`` id.  This is consistent with api_request_start
#  logged in the create_chat_completion patch above.
# =============================================================================

try:
    from vllm.entrypoints.serve.render.serving import OpenAIServingRender

    _orig_render_chat = OpenAIServingRender.render_chat

    async def _patched_render_chat(self: OpenAIServingRender, request: Any) -> Any:
        req_id = request.request_id or f"req-{uuid.uuid4().hex[:12]}"
        # Stash the request_id on the renderer so that the tokenization
        # patch (BaseRenderer._tokenize_prompt) can pick it up.
        self.renderer._ttft_req_id = req_id
        _ttft_start(req_id, "render_chat")
        result = await _orig_render_chat(self, request)
        _ttft_end(req_id, "render_chat")
        return result

    OpenAIServingRender.render_chat = _patched_render_chat
    logger.warning("TTFT: patched OpenAIServingRender.render_chat")
except ImportError as e:
    logger.warning("TTFT: OpenAIServingRender NOT available, skipping")


# =============================================================================
#  Patch: MediaConnector  (Image download)
# =============================================================================

try:
    from vllm.multimodal.media.connector import MediaConnector

    # --- Sync variant ---
    _orig_load_from_url = MediaConnector.load_from_url

    def _patched_load_from_url(
        self: MediaConnector,
        url: str,
        media_io: Any,
        *,
        fetch_timeout: int | None = None,
    ) -> Any:
        from urllib3.util import parse_url

        url_spec = parse_url(url)
        if url_spec.scheme and url_spec.scheme.startswith("http"):
            _ttft_start("media_connector", "http_download")
            data = self.connection.get_bytes(
                url_spec.url,
                timeout=fetch_timeout,
                allow_redirects=True,
            )
            _ttft_end("media_connector", "http_download")

            _ttft_start("media_connector", "image_decode")
            result = media_io.load_bytes(data)
            _ttft_end("media_connector", "image_decode")
            return result

        if url_spec.scheme == "data":
            return _orig_load_from_url(
                self, url, media_io, fetch_timeout=fetch_timeout
            )
        if url_spec.scheme == "file":
            return _orig_load_from_url(
                self, url, media_io, fetch_timeout=fetch_timeout
            )
        return _orig_load_from_url(
            self, url, media_io, fetch_timeout=fetch_timeout
        )

    MediaConnector.load_from_url = _patched_load_from_url

    # --- Async variant ---
    _orig_load_from_url_async = MediaConnector.load_from_url_async

    async def _patched_load_from_url_async(
        self: MediaConnector,
        url: str,
        media_io: Any,
        *,
        fetch_timeout: int | None = None,
    ) -> Any:
        import asyncio
        from urllib3.util import parse_url

        url_spec = parse_url(url)
        loop = asyncio.get_running_loop()

        if url_spec.scheme and url_spec.scheme.startswith("http"):
            _ttft_start("media_connector", "http_download_async")
            data = await self.connection.async_get_bytes(
                url_spec.url,
                timeout=fetch_timeout,
                allow_redirects=True,
            )
            _ttft_end("media_connector", "http_download_async")

            _ttft_start("media_connector", "image_decode_async")
            # Use the global thread pool for decoding
            from vllm.multimodal.media.connector import global_thread_pool

            future = loop.run_in_executor(global_thread_pool, media_io.load_bytes, data)
            result = await future
            _ttft_end("media_connector", "image_decode_async")
            return result

        return await _orig_load_from_url_async(
            self, url, media_io, fetch_timeout=fetch_timeout
        )

    MediaConnector.load_from_url_async = _patched_load_from_url_async
    logger.warning("TTFT: patched MediaConnector.load_from_url / load_from_url_async")
except ImportError as e:
    logger.warning("TTFT: MediaConnector NOT available, skipping")


# =============================================================================
#  Patch: Phase4 HTTP download (VLLM_ASCEND_API_OPT_PHASE >= 3 bypasses
#         MediaConnector.load_from_url_async; HTTP images go through
#         _phase4_materialize_http_url in phase1.py instead)
# =============================================================================

try:
    from vllm_ascend.patch.platform.phase import phase1

    _orig_phase4_materialize = phase1._phase4_materialize_http_url

    def _patched_phase4_materialize(image_url: str | None):
        _ttft_start("media_connector", "http_download_phase4")
        result = _orig_phase4_materialize(image_url)
        _ttft_end("media_connector", "http_download_phase4")
        return result

    phase1._phase4_materialize_http_url = _patched_phase4_materialize
    logger.warning(
        "TTFT: patched phase1._phase4_materialize_http_url (phase4 HTTP download)"
    )
except ImportError:
    # Phase1 module not loaded — no phase optimisation active, nothing to do.
    pass

except Exception:
    logger.warning("TTFT: phase1._phase4_materialize_http_url patch failed, skipping")


# =============================================================================
#  Patch: BaseRenderer._process_multimodal  (HF processor)
# =============================================================================

try:
    from vllm.renderers.base import BaseRenderer

    _orig_process_multimodal = BaseRenderer._process_multimodal

    def _patched_process_multimodal(
        self: BaseRenderer,
        prompt: Any,
        mm_data: Any,
        mm_uuids: Any = None,
        mm_processor_kwargs: Any = None,
        tokenization_kwargs: Any = None,
    ) -> Any:
        mm_req_id = getattr(self, "_mm_req_id", None)
        if mm_req_id is None:
            mm_req_id = f"renderer-mm-{uuid.uuid4().hex[:8]}"

        # The original method creates a TimingContext and calls mm_processor.apply()
        # which is the HF processor call.
        _ttft_start(mm_req_id, "hf_multimodal_processor")
        result = _orig_process_multimodal(
            self, prompt, mm_data, mm_uuids, mm_processor_kwargs, tokenization_kwargs
        )
        _ttft_end(mm_req_id, "hf_multimodal_processor")
        return result

    BaseRenderer._process_multimodal = _patched_process_multimodal
    logger.warning("TTFT: patched BaseRenderer._process_multimodal")

    # ---- _tokenize_prompt (text → token_ids, inside render_chat) ----
    # The request_id is passed via self.renderer._ttft_req_id, set by the
    # OpenAIServingRender.render_chat patch above.
    _orig_tokenize_prompt = BaseRenderer._tokenize_prompt

    def _patched_tokenize_prompt(
        self: BaseRenderer, prompt: Any, params: Any
    ) -> Any:
        req_id = getattr(
            self, "_ttft_req_id", f"tokenizer-{uuid.uuid4().hex[:8]}"
        )
        _ttft_start(req_id, "tokenization")
        result = _orig_tokenize_prompt(self, prompt, params)
        _ttft_end(req_id, "tokenization")
        return result

    BaseRenderer._tokenize_prompt = _patched_tokenize_prompt

    # ---- _tokenize_prompt_async (async variant) ----
    _orig_tokenize_prompt_async = BaseRenderer._tokenize_prompt_async

    async def _patched_tokenize_prompt_async(
        self: BaseRenderer, prompt: Any, params: Any
    ) -> Any:
        req_id = getattr(
            self, "_ttft_req_id", f"tokenizer-{uuid.uuid4().hex[:8]}"
        )
        _ttft_start(req_id, "tokenization_async")
        result = await _orig_tokenize_prompt_async(self, prompt, params)
        _ttft_end(req_id, "tokenization_async")
        return result

    BaseRenderer._tokenize_prompt_async = _patched_tokenize_prompt_async
    logger.warning("TTFT: patched BaseRenderer._tokenize_prompt / _tokenize_prompt_async")
except ImportError as e:
    logger.warning("TTFT: BaseRenderer NOT available, skipping")


# =============================================================================
#  Patch: AsyncLLM.add_request & generate  (Engine frontend)
# =============================================================================

try:
    from vllm.v1.engine.async_llm import AsyncLLM

    # ---- add_request (input processing) ----
    _orig_add_request = AsyncLLM.add_request

    async def _patched_add_request(
        self: AsyncLLM,
        request_id: str,
        prompt: Any,
        params: Any,
        arrival_time: float | None = None,
        lora_request: Any = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Any = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        prompt_text: str | None = None,
        reasoning_ended: bool | None = None,
    ) -> Any:
        _ttft_start(request_id, "input_processing")
        result = await _orig_add_request(
            self,
            request_id,
            prompt,
            params,
            arrival_time=arrival_time,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
            trace_headers=trace_headers,
            priority=priority,
            data_parallel_rank=data_parallel_rank,
            prompt_text=prompt_text,
            reasoning_ended=reasoning_ended,
        )
        _ttft_end(request_id, "input_processing")
        return result

    AsyncLLM.add_request = _patched_add_request

    # ---- _add_request (engine core dispatch) ----
    _orig_add_request_private = AsyncLLM._add_request

    async def _patched_add_request_private(
        self: AsyncLLM,
        request: Any,
        prompt: str | None,
        parent_req: Any,
        index: int,
        queue: Any,
    ) -> None:
        _ttft_start(request.request_id, "engine_core_dispatch")
        await _orig_add_request_private(
            self, request, prompt, parent_req, index, queue
        )
        _ttft_end(request.request_id, "engine_core_dispatch")

    AsyncLLM._add_request = _patched_add_request_private

    # ---- generate (engine_generate entry + api_server_dispatch) ----
    # Original signature: generate(self, prompt, sampling_params, request_id, *, ...)
    _orig_generate = AsyncLLM.generate

    async def _patched_generate(
        self: AsyncLLM,
        prompt: Any,
        sampling_params: Any,
        request_id: str,
        *,
        lora_request: Any = None,
        prompt_text: str | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Any = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        reasoning_ended: bool | None = None,
    ) -> Any:
        # API Server → Engine boundary.  This fires right after
        # create_chat_completion finishes setup and calls
        # engine_client.generate().  The elapsed time since
        # api_request_start is the API Server processing time.
        _ttft_log(request_id, "api_server_dispatch")
        _ttft_start(request_id, "engine_generate")

        first_token = True
        async for output in _orig_generate(
            self,
            prompt,
            sampling_params,
            request_id,
            lora_request=lora_request,
            prompt_text=prompt_text,
            tokenization_kwargs=tokenization_kwargs,
            trace_headers=trace_headers,
            priority=priority,
            data_parallel_rank=data_parallel_rank,
            reasoning_ended=reasoning_ended,
        ):
            if first_token:
                first_token = False
                _ttft_log(request_id, "first_token_api_yield")
            yield output

    AsyncLLM.generate = _patched_generate
    logger.warning("TTFT: patched AsyncLLM.add_request / _add_request / generate")
except ImportError as e:
    logger.warning("TTFT: AsyncLLM NOT available, skipping")


# =============================================================================
#  Patch: EngineCore  (Engine core process)
# =============================================================================

try:
    from vllm.v1.engine.core import EngineCore

    # ---- add_request ----
    _orig_ec_add_request = EngineCore.add_request

    def _patched_ec_add_request(
        self: EngineCore, request: Any, request_wave: int = 0
    ) -> None:
        _ttft_log(request.request_id, "engine_core_add_request")
        _orig_ec_add_request(self, request, request_wave)

    EngineCore.add_request = _patched_ec_add_request

    # ---- step ----
    # Simple wrapper -- finer-grained timing (scheduler, model runner,
    # encoder runner) is covered by their respective patches.
    _orig_ec_step = EngineCore.step

    def _patched_ec_step(self: EngineCore) -> tuple[Any, bool]:
        _ttft_log("batch", "engine_core_step_start")
        result = _orig_ec_step(self)
        _ttft_log("batch", "engine_core_step_end")
        return result

    EngineCore.step = _patched_ec_step
    logger.warning("TTFT: patched EngineCore.step")
except ImportError as e:
    logger.warning("TTFT: EngineCore NOT available, skipping")


# =============================================================================
#  Patch: Scheduler  (Scheduling and queuing)
# =============================================================================

try:
    from vllm.v1.core.sched.scheduler import Scheduler, RequestStatus

    # ---- add_request ----
    _orig_sched_add_request = Scheduler.add_request

    def _patched_sched_add_request(
        self: Scheduler, request: Any
    ) -> None:
        _ttft_log(request.request_id, "scheduler_add_request")
        # We need to also log when enqueueing.  Since the original method
        # calls _enqueue_waiting_request internally for new requests,
        # we patch that separately.
        _orig_sched_add_request(self, request)

    Scheduler.add_request = _patched_sched_add_request

    # ---- _enqueue_waiting_request ----
    _orig_enqueue_waiting = Scheduler._enqueue_waiting_request

    def _patched_enqueue_waiting(
        self: Scheduler, request: Any
    ) -> None:
        _ttft_log(request.request_id, "scheduler_enqueue_waiting")
        _orig_enqueue_waiting(self, request)

    Scheduler._enqueue_waiting_request = _patched_enqueue_waiting

    # ---- schedule (wrapper) ----
    # schedule() is ~500 lines.  We wrap it instead of replacing it so that
    # future vLLM changes don't silently break the timing.  After the
    # original returns we inspect the output to detect newly-scheduled
    # requests and encoder inputs.
    _orig_schedule = Scheduler.schedule

    def _patched_schedule(self: Scheduler) -> Any:
        output = _orig_schedule(self)
        try:
            new_reqs = getattr(output, "scheduled_new_reqs", None)
        except Exception as exc:
            logger.warning("TTFT: schedule wrapper failed to read scheduled_new_reqs: %s", exc)
            new_reqs = None
        if new_reqs:
            for req_data in new_reqs:
                rid = getattr(req_data, "req_id", None) or getattr(req_data, "request_id", None)
                if rid:
                    _ttft_log(str(rid), "scheduler_scheduled")
        else:
            # Debug: check whether schedule ever returns empty new_reqs
            has_any = bool(getattr(output, "scheduled_new_reqs", None) or
                           getattr(output, "scheduled_running_reqs", None))
            if has_any is False:
                pass  # idle step, no requests at all
        enc_inputs = getattr(output, "scheduled_encoder_inputs", None)
        if enc_inputs and new_reqs:
            new_ids = {getattr(nr, "req_id", None) or getattr(nr, "request_id", None) for nr in new_reqs}
            for req_id in enc_inputs:
                if req_id in new_ids:
                    _ttft_log(req_id, "scheduler_encoder_scheduled")
        return output

    Scheduler.schedule = _patched_schedule

    # ---- peek_request (pickup from waiting queue) ----
    # RequestQueue is an ABC; peek_request is overridden by concrete
    # subclasses (FCFSRequestQueue, PriorityRequestQueue).  We must
    # patch the concrete classes, not the ABC.
    from vllm.v1.core.sched.request_queue import (
        RequestQueue, FCFSRequestQueue, PriorityRequestQueue,
    )

    _orig_fcfs_peek = FCFSRequestQueue.peek_request

    def _patched_fcfs_peek(self: FCFSRequestQueue) -> Any:
        req = _orig_fcfs_peek(self)
        if getattr(req, "status", None) == RequestStatus.WAITING:
            _ttft_log(req.request_id, "scheduler_pickup_from_waiting")
        return req

    FCFSRequestQueue.peek_request = _patched_fcfs_peek

    _orig_prio_peek = PriorityRequestQueue.peek_request

    def _patched_prio_peek(self: PriorityRequestQueue) -> Any:
        req = _orig_prio_peek(self)
        if getattr(req, "status", None) == RequestStatus.WAITING:
            _ttft_log(req.request_id, "scheduler_pickup_from_waiting")
        return req

    PriorityRequestQueue.peek_request = _patched_prio_peek
    logger.warning("TTFT: patched Scheduler (add_request / enqueue / schedule / peek_request)")
except ImportError as e:
    logger.warning("TTFT: Scheduler patches NOT available, skipping")


# =============================================================================
#  Patch: RecomputeScheduler (Ascend scheduler subclass with own schedule())
# =============================================================================
#  vllm_ascend replaces the default Scheduler with RecomputeScheduler, which
#  overrides schedule().  We must patch the subclass separately.

try:
    from vllm_ascend.core.recompute_scheduler import RecomputeScheduler

    _orig_recomp_schedule = RecomputeScheduler.schedule

    def _patched_recomp_schedule(self: Any) -> Any:
        output = _orig_recomp_schedule(self)
        try:
            new_reqs = getattr(output, "scheduled_new_reqs", None)
        except Exception:
            new_reqs = None
        if new_reqs:
            for req_data in new_reqs:
                rid = getattr(req_data, "req_id", None) or getattr(req_data, "request_id", None)
                if rid:
                    _ttft_log(str(rid), "scheduler_scheduled")
        enc_inputs = getattr(output, "scheduled_encoder_inputs", None)
        if enc_inputs and new_reqs:
            new_ids = {getattr(nr, "req_id", None) or getattr(nr, "request_id", None) for nr in new_reqs}
            for req_id in enc_inputs:
                if req_id in new_ids:
                    _ttft_log(req_id, "scheduler_encoder_scheduled")
        return output

    RecomputeScheduler.schedule = _patched_recomp_schedule
    logger.warning("TTFT: patched RecomputeScheduler.schedule")
except ImportError:
    pass


# =============================================================================
#  Patch: NPUModelRunner  (Model execution, worker process)
# =============================================================================

try:
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    # ---- execute_model ----
    _orig_execute_model = NPUModelRunner.execute_model

    def _patched_execute_model(
        self: NPUModelRunner, scheduler_output: Any, intermediate_tensors: Any = None
    ) -> Any:
        # Per-request: log model start / end for every scheduled request
        _all_req_ids: set[str] = set()
        for nr in getattr(scheduler_output, "scheduled_new_reqs", []):
            _all_req_ids.add(nr.req_id)
        for rr in getattr(scheduler_output, "scheduled_running_reqs", []):
            _all_req_ids.add(rr.req_id)
        for req_id in _all_req_ids:
            _ttft_log(req_id, "worker_model_exec_start")
        # Stash req_ids so _patched_model_forward can emit per-request logs
        self._ttft_forward_req_ids = _all_req_ids
        result = _orig_execute_model(self, scheduler_output, intermediate_tensors)
        _sync_device()
        for req_id in _all_req_ids:
            _ttft_log(req_id, "worker_model_exec_end")
        return result

    NPUModelRunner.execute_model = _patched_execute_model

    # ---- _model_forward ----
    # NPUModelRunner._model_forward has an extra positional arg
    # ``num_tokens_padded`` that GPUModelRunner does not.
    _orig_model_forward = NPUModelRunner._model_forward

    def _patched_model_forward(
        self: NPUModelRunner,
        num_tokens_padded: int,
        input_ids: Any = None,
        positions: Any = None,
        intermediate_tensors: Any = None,
        inputs_embeds: Any = None,
        **model_kwargs: Any,
    ) -> Any:
        # Per-request forward pass timing (IDs stashed by execute_model)
        _req_ids = getattr(self, "_ttft_forward_req_ids", set())
        for req_id in _req_ids:
            _ttft_log(req_id, "worker_forward_pass_start")
        output = _orig_model_forward(
            self,
            num_tokens_padded,
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )
        _sync_device()
        for req_id in _req_ids:
            _ttft_log(req_id, "worker_forward_pass_end")
        return output

    NPUModelRunner._model_forward = _patched_model_forward
    logger.warning("TTFT: patched NPUModelRunner.execute_model / _model_forward")
except ImportError as e:
    logger.warning("TTFT: NPUModelRunner NOT available, skipping")


# =============================================================================
#  Patch: EncoderRunner  (ViT forward)
# =============================================================================

try:
    from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner

    _orig_enc_execute = EncoderRunner.execute_mm_encoder

    def _patched_enc_execute(
        self: EncoderRunner, mm_kwargs: list[Any]
    ) -> Any:
        _ttft_log("batch", "encoder_runner_vit_start")
        result = _orig_enc_execute(self, mm_kwargs)
        _sync_device()
        _ttft_log("batch", "encoder_runner_vit_end")
        return result

    EncoderRunner.execute_mm_encoder = _patched_enc_execute
    logger.warning("TTFT: patched EncoderRunner.execute_mm_encoder")
except ImportError as e:
    logger.warning("TTFT: EncoderRunner NOT available, skipping")


# =============================================================================
#  Patch: OutputProcessor  (First token detection)
# =============================================================================

try:
    from vllm.v1.engine.output_processor import OutputProcessor

    # ---- process_outputs (was _process_engine_outputs in older vllm) ----
    _orig_process_outputs = OutputProcessor.process_outputs

    def _patched_process_outputs(
        self: OutputProcessor,
        engine_core_outputs: Any,
        engine_core_timestamp: float | None = None,
        iteration_stats: Any = None,
    ) -> Any:
        # Intercept: for each output, check if this is the first token
        for engine_core_output in engine_core_outputs:
            req_id = engine_core_output.request_id
            req_state = self.request_states.get(req_id)
            if req_state is not None:
                new_token_ids = getattr(engine_core_output, "new_token_ids", [])
                if (
                    getattr(req_state, "is_prefilling", False)
                    and new_token_ids
                ):
                    _ttft_log(req_id, "first_token_output")

        return _orig_process_outputs(
            self, engine_core_outputs, engine_core_timestamp, iteration_stats
        )

    OutputProcessor.process_outputs = _patched_process_outputs
    logger.warning("TTFT: patched OutputProcessor.process_outputs")
except ImportError as e:
    logger.warning("TTFT: OutputProcessor NOT available, skipping")


# =============================================================================
#  Patch: MooncakeLayerwiseConnectorWorker  (Distributed KV transfer, Ascend)
# =============================================================================

try:
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
        MooncakeLayerwiseConnectorWorker,
    )

    # ---- start_load_kv ----
    _orig_start_load_kv = MooncakeLayerwiseConnectorWorker.start_load_kv

    def _patched_start_load_kv(self: MooncakeLayerwiseConnectorWorker, metadata: Any) -> None:
        if getattr(self.vllm_config.kv_transfer_config, "is_kv_consumer", False):
            for req_id in getattr(metadata, "requests", {}):
                _ttft_log(req_id, "mooncake_kv_load_start")
        _orig_start_load_kv(self, metadata)

    MooncakeLayerwiseConnectorWorker.start_load_kv = _patched_start_load_kv

    # ---- get_finished ----
    _orig_get_finished = MooncakeLayerwiseConnectorWorker.get_finished

    def _patched_get_finished(
        self: MooncakeLayerwiseConnectorWorker,
    ) -> tuple[set[str], set[str]]:
        done_sending, done_recving = _orig_get_finished(self)
        for req_id in done_recving:
            _ttft_log(req_id, "mooncake_kv_load_end")
        return done_sending, done_recving

    MooncakeLayerwiseConnectorWorker.get_finished = _patched_get_finished
    # NOTE: start_load_kv already calls ttft_log from vllm.v1.utils.ttft_timing
    # in the original mooncake connector (line ~1452).  Our wrapper re-logs
    # via _ttft_log to use the same logger as other trace events.
    logger.warning("TTFT: patched MooncakeLayerwiseConnectorWorker")
except ImportError as e:
    logger.warning("TTFT: MooncakeLayerwiseConnectorWorker NOT available, skipping: %s", e)

# =============================================================================
#  Final log
# =============================================================================

logger.warning("TTFT timing monkey patches activated. Filter logs with: grep 'TTFT_TRACE' <logfile>")

# =============================================================================
#  Patch: GPUModelRunner._execute_mm_encoder  (ViT encoder, worker)
#
#  NOTE: Phase2 (phase2.py) also patches this method.  If phase2 runs AFTER
#  this import, it will overwrite our wrapper.  Place this import LAST in
#  __init__.py so phase2 patches are already in place when we capture them.
# =============================================================================

try:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    _orig_gpu_execute_mm_encoder = GPUModelRunner._execute_mm_encoder

    def _patched_gpu_execute_mm_encoder(
        self: GPUModelRunner, scheduler_output: Any
    ) -> Any:
        _ttft_log("batch", "model_runner_mm_encoder_start")
        result = _orig_gpu_execute_mm_encoder(self, scheduler_output)
        _sync_device()
        _ttft_log("batch", "model_runner_mm_encoder_end")
        return result

    GPUModelRunner._execute_mm_encoder = _patched_gpu_execute_mm_encoder
    logger.warning("TTFT: patched GPUModelRunner._execute_mm_encoder (post-phase2)")
except ImportError:
    pass
