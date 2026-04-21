"""Auto-instrumentation for the Anthropic Python SDK."""

from __future__ import annotations

import functools
import time
import warnings
from collections.abc import AsyncGenerator, Generator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trulayer.client import TruLayerClient

_original_create: Any = None
_original_acreate: Any = None
_patched = False


def instrument_anthropic(client: TruLayerClient) -> None:
    """
    Monkey-patch anthropic to wrap messages.create in a TruLayer span.
    Handles both standard and streaming responses. Idempotent.
    """
    global _original_create, _original_acreate, _patched
    if _patched:
        return
    try:
        import anthropic  # noqa: PLC0415
    except ImportError:
        warnings.warn("trulayer: anthropic not installed, skipping instrumentation", stacklevel=2)
        return

    _original_create = anthropic.resources.messages.Messages.create
    _original_acreate = anthropic.resources.messages.AsyncMessages.create

    @functools.wraps(_original_create)
    def _patched_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.monotonic()
        result = _original_create(self, *args, **kwargs)
        if kwargs.get("stream"):
            return _wrap_sync_stream(client, kwargs, result, start)
        _record_span(client, kwargs, result, time.monotonic() - start)
        return result

    @functools.wraps(_original_acreate)
    async def _patched_acreate(self: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.monotonic()
        result = await _original_acreate(self, *args, **kwargs)
        if kwargs.get("stream"):
            return _wrap_async_stream(client, kwargs, result, start)
        _record_span(client, kwargs, result, time.monotonic() - start)
        return result

    anthropic.resources.messages.Messages.create = _patched_create
    anthropic.resources.messages.AsyncMessages.create = _patched_acreate
    _patched = True


def uninstrument_anthropic() -> None:
    """Restore original Anthropic methods. Idempotent."""
    global _original_create, _original_acreate, _patched
    if not _patched:
        return
    try:
        import anthropic  # noqa: PLC0415

        if _original_create:
            anthropic.resources.messages.Messages.create = _original_create
        if _original_acreate:
            anthropic.resources.messages.AsyncMessages.create = _original_acreate
    except ImportError:
        pass
    _patched = False


def _record_span(
    client: TruLayerClient,
    kwargs: dict[str, Any],
    result: Any,
    elapsed: float,
) -> None:
    try:
        from trulayer.trace import current_trace  # noqa: PLC0415

        trace = current_trace()
        if trace is None:
            return

        output = ""
        prompt_tokens: int | None = None
        completion_tokens: int | None = None

        try:
            for block in result.content:
                if block.type == "text":
                    output = block.text
                    break
            if result.usage:
                prompt_tokens = result.usage.input_tokens
                completion_tokens = result.usage.output_tokens
        except Exception:
            pass

        messages = kwargs.get("messages", [])
        input_text = messages[-1].get("content", "") if messages else ""

        with trace.span("anthropic.messages", span_type="llm") as span:
            span.set_input(str(input_text))
            span.set_output(output)
            span.set_model(kwargs.get("model", ""))
            span.set_tokens(prompt=prompt_tokens, completion=completion_tokens)
            span._data.latency_ms = int(elapsed * 1000)
    except Exception as exc:
        warnings.warn(f"trulayer: failed to record Anthropic span: {exc}", stacklevel=2)


def _wrap_sync_stream(
    client: TruLayerClient,
    kwargs: dict[str, Any],
    stream: Any,
    start: float,
) -> Generator[Any, None, None]:
    """Yield events from a streaming Anthropic response while recording a span."""
    try:
        from trulayer.trace import SpanContext, current_trace  # noqa: PLC0415

        trace = current_trace()
        if trace is None:
            yield from stream
            return

        messages = kwargs.get("messages", [])
        input_text = str(messages[-1].get("content", "") if messages else "")

        span = SpanContext(trace, "anthropic.messages", "llm")
        span.set_input(input_text)
        span.set_model(kwargs.get("model", ""))
        span.__enter__()

        accumulated: list[str] = []
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        exc_info: tuple[Any, Any, Any] = (None, None, None)

        try:
            for event in stream:
                try:
                    event_type = getattr(event, "type", None)
                    if event_type == "content_block_delta":
                        delta = getattr(event.delta, "text", None)
                        if delta:
                            accumulated.append(delta)
                    elif event_type == "message_delta":
                        usage = getattr(event, "usage", None)
                        if usage:
                            prompt_tokens = getattr(usage, "input_tokens", None)
                            completion_tokens = getattr(usage, "output_tokens", None)
                except Exception:
                    pass
                yield event
        except BaseException as exc:
            exc_info = (type(exc), exc, exc.__traceback__)
            raise
        finally:
            span.set_output("".join(accumulated))
            span.set_tokens(prompt=prompt_tokens, completion=completion_tokens)
            span._data.latency_ms = int((time.monotonic() - start) * 1000)
            span.__exit__(*exc_info)

    except Exception as exc:
        warnings.warn(f"trulayer: failed to record Anthropic streaming span: {exc}", stacklevel=2)


async def _wrap_async_stream(
    client: TruLayerClient,
    kwargs: dict[str, Any],
    stream: Any,
    start: float,
) -> AsyncGenerator[Any, None]:
    """Async-yield events from a streaming Anthropic response while recording a span."""
    try:
        from trulayer.trace import SpanContext, current_trace  # noqa: PLC0415

        trace = current_trace()
        if trace is None:
            async for event in stream:
                yield event
            return

        messages = kwargs.get("messages", [])
        input_text = str(messages[-1].get("content", "") if messages else "")

        span = SpanContext(trace, "anthropic.messages", "llm")
        span.set_input(input_text)
        span.set_model(kwargs.get("model", ""))
        span.__enter__()

        accumulated: list[str] = []
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        exc_info: tuple[Any, Any, Any] = (None, None, None)

        try:
            async for event in stream:
                try:
                    event_type = getattr(event, "type", None)
                    if event_type == "content_block_delta":
                        delta = getattr(event.delta, "text", None)
                        if delta:
                            accumulated.append(delta)
                    elif event_type == "message_delta":
                        usage = getattr(event, "usage", None)
                        if usage:
                            prompt_tokens = getattr(usage, "input_tokens", None)
                            completion_tokens = getattr(usage, "output_tokens", None)
                except Exception:
                    pass
                yield event
        except BaseException as exc:
            exc_info = (type(exc), exc, exc.__traceback__)
            raise
        finally:
            span.set_output("".join(accumulated))
            span.set_tokens(prompt=prompt_tokens, completion=completion_tokens)
            span._data.latency_ms = int((time.monotonic() - start) * 1000)
            span.__exit__(*exc_info)

    except Exception as exc:
        warnings.warn(
            f"trulayer: failed to record Anthropic async streaming span: {exc}",
            stacklevel=2,
        )
