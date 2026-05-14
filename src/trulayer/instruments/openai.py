"""Auto-instrumentation for the OpenAI Python SDK."""

from __future__ import annotations

import functools
import time
import warnings
from collections.abc import AsyncGenerator, Generator
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trulayer.client import TruLayerClient

_original_create: Any = None
_original_acreate: Any = None
_patched = False


def instrument_openai(client: TruLayerClient) -> None:
    """
    Monkey-patch openai to wrap chat.completions.create in a TruLayer span.
    Handles both standard and streaming responses. Idempotent.
    """
    global _original_create, _original_acreate, _patched
    if _patched:
        return
    try:
        import openai  # noqa: PLC0415
    except ImportError:
        warnings.warn("trulayer: openai not installed, skipping instrumentation", stacklevel=2)
        return

    _original_create = openai.resources.chat.completions.Completions.create
    _original_acreate = openai.resources.chat.completions.AsyncCompletions.create

    @functools.wraps(_original_create)
    def _patched_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_wall = datetime.now(tz=UTC)
        start = time.monotonic()
        result = _original_create(self, *args, **kwargs)
        if kwargs.get("stream"):
            return _wrap_sync_stream(client, kwargs, result, start, start_wall)
        _record_span(client, kwargs, result, time.monotonic() - start, start_wall)
        return result

    @functools.wraps(_original_acreate)
    async def _patched_acreate(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_wall = datetime.now(tz=UTC)
        start = time.monotonic()
        result = await _original_acreate(self, *args, **kwargs)
        if kwargs.get("stream"):
            return _wrap_async_stream(client, kwargs, result, start, start_wall)
        _record_span(client, kwargs, result, time.monotonic() - start, start_wall)
        return result

    openai.resources.chat.completions.Completions.create = _patched_create
    openai.resources.chat.completions.AsyncCompletions.create = _patched_acreate
    _patched = True


def uninstrument_openai() -> None:
    """Restore original OpenAI methods. Idempotent."""
    global _original_create, _original_acreate, _patched
    if not _patched:
        return
    try:
        import openai  # noqa: PLC0415

        if _original_create:
            openai.resources.chat.completions.Completions.create = _original_create
        if _original_acreate:
            openai.resources.chat.completions.AsyncCompletions.create = _original_acreate
    except ImportError:
        pass
    _patched = False


def _record_span(
    client: TruLayerClient,
    kwargs: dict[str, Any],
    result: Any,
    elapsed: float,
    start_wall: datetime | None = None,
) -> None:
    if start_wall is None:
        # Best-effort fallback: reconstruct start from now - elapsed. Used only
        # when callers (e.g. tests) skip passing the wall-clock anchor; the
        # patched create() always supplies it.
        start_wall = datetime.now(tz=UTC) - timedelta(seconds=elapsed)
    try:
        from trulayer.trace import current_trace  # noqa: PLC0415

        trace = current_trace()
        if trace is None:
            return

        output = ""
        prompt_tokens: int | None = None
        completion_tokens: int | None = None

        try:
            output = result.choices[0].message.content or ""
            if result.usage:
                prompt_tokens = result.usage.prompt_tokens
                completion_tokens = result.usage.completion_tokens
        except Exception:
            pass

        messages = kwargs.get("messages", [])
        input_text = messages[-1].get("content", "") if messages else ""

        latency_ms = int(elapsed * 1000)
        with trace.span("openai.chat", span_type="llm") as span:
            span.set_input(str(input_text))
            span.set_output(output)
            span.set_model(kwargs.get("model", ""))
            span.set_tokens(prompt=prompt_tokens, completion=completion_tokens)
        # Overwrite the auto-captured fields. The context manager's
        # __enter__/__exit__ both ran *after* the upstream call returned, so
        # started_at, ended_at, and latency_ms all reflect the SDK-internal
        # microsecond gap instead of the real call. Replace them with the
        # wall-clock window we anchored before invoking the upstream.
        span._data.latency_ms = latency_ms
        span._data.started_at = start_wall
        span._data.ended_at = start_wall + timedelta(milliseconds=latency_ms)
    except Exception as exc:
        warnings.warn(f"trulayer: failed to record OpenAI span: {exc}", stacklevel=2)


def _wrap_sync_stream(
    client: TruLayerClient,
    kwargs: dict[str, Any],
    stream: Any,
    start: float,
    start_wall: datetime | None = None,
) -> Generator[Any, None, None]:
    """Yield chunks from a streaming OpenAI response while recording a span."""
    if start_wall is None:
        start_wall = datetime.now(tz=UTC)
    try:
        from trulayer.trace import SpanContext, current_trace  # noqa: PLC0415

        trace = current_trace()
        if trace is None:
            yield from stream
            return

        messages = kwargs.get("messages", [])
        input_text = str(messages[-1].get("content", "") if messages else "")

        span = SpanContext(trace, "openai.chat", "llm")
        span.set_input(input_text)
        span.set_model(kwargs.get("model", ""))
        span.__enter__()
        # Override the __enter__ timestamp with the wall-clock moment the
        # upstream stream call actually began.
        span._data.started_at = start_wall

        accumulated: list[str] = []
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        exc_info: tuple[Any, Any, Any] = (None, None, None)

        try:
            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        accumulated.append(delta)
                    # Some SDKs include usage on the final chunk
                    if hasattr(chunk, "usage") and chunk.usage:
                        prompt_tokens = getattr(chunk.usage, "prompt_tokens", None)
                        completion_tokens = getattr(chunk.usage, "completion_tokens", None)
                except Exception:
                    pass
                yield chunk
        except BaseException as exc:
            exc_info = (type(exc), exc, exc.__traceback__)
            raise
        finally:
            span.set_output("".join(accumulated))
            span.set_tokens(prompt=prompt_tokens, completion=completion_tokens)
            latency_ms = int((time.monotonic() - start) * 1000)
            span.__exit__(*exc_info)
            # __exit__ overwrites latency/ended_at with its own monotonic
            # delta and wall clock. Re-anchor both to the upstream window
            # captured when the patched create() was invoked.
            span._data.latency_ms = latency_ms
            span._data.started_at = start_wall
            span._data.ended_at = start_wall + timedelta(milliseconds=latency_ms)

    except Exception as exc:
        warnings.warn(f"trulayer: failed to record OpenAI streaming span: {exc}", stacklevel=2)


async def _wrap_async_stream(
    client: TruLayerClient,
    kwargs: dict[str, Any],
    stream: Any,
    start: float,
    start_wall: datetime | None = None,
) -> AsyncGenerator[Any, None]:
    """Async-yield chunks from a streaming OpenAI response while recording a span."""
    if start_wall is None:
        start_wall = datetime.now(tz=UTC)
    try:
        from trulayer.trace import SpanContext, current_trace  # noqa: PLC0415

        trace = current_trace()
        if trace is None:
            async for chunk in stream:
                yield chunk
            return

        messages = kwargs.get("messages", [])
        input_text = str(messages[-1].get("content", "") if messages else "")

        span = SpanContext(trace, "openai.chat", "llm")
        span.set_input(input_text)
        span.set_model(kwargs.get("model", ""))
        span.__enter__()
        # Override the __enter__ timestamp with the wall-clock moment the
        # upstream stream call actually began.
        span._data.started_at = start_wall

        accumulated: list[str] = []
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        exc_info: tuple[Any, Any, Any] = (None, None, None)

        try:
            async for chunk in stream:
                try:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        accumulated.append(delta)
                    if hasattr(chunk, "usage") and chunk.usage:
                        prompt_tokens = getattr(chunk.usage, "prompt_tokens", None)
                        completion_tokens = getattr(chunk.usage, "completion_tokens", None)
                except Exception:
                    pass
                yield chunk
        except BaseException as exc:
            exc_info = (type(exc), exc, exc.__traceback__)
            raise
        finally:
            span.set_output("".join(accumulated))
            span.set_tokens(prompt=prompt_tokens, completion=completion_tokens)
            latency_ms = int((time.monotonic() - start) * 1000)
            span.__exit__(*exc_info)
            # __exit__ overwrites latency/ended_at with its own monotonic
            # delta and wall clock. Re-anchor both to the upstream window
            # captured when the patched create() was invoked.
            span._data.latency_ms = latency_ms
            span._data.started_at = start_wall
            span._data.ended_at = start_wall + timedelta(milliseconds=latency_ms)

    except Exception as exc:
        warnings.warn(
            f"trulayer: failed to record OpenAI async streaming span: {exc}",
            stacklevel=2,
        )
