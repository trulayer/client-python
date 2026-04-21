"""PydanticAI Agent instrumentation for TruLayer auto-instrumentation."""

from __future__ import annotations

import functools
import warnings
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trulayer.trace import TraceContext


def instrument_pydanticai(
    agent: Any,
    trace_ctx: TraceContext,
    *,
    capture_input: bool = True,
    capture_output: bool = True,
    run_name: str | None = None,
) -> Any:
    """
    Wrap a PydanticAI Agent so every ``.run()`` / ``.run_sync()`` / ``.run_stream()``
    call is recorded as a TruLayer trace with child spans for each tool invocation.

    Returns the same agent instance (mutated in place via monkey-patching).
    Never raises into user code.
    """
    try:
        _patch_run(agent, trace_ctx, capture_input, capture_output, run_name)
        _patch_run_sync(agent, trace_ctx, capture_input, capture_output, run_name)
        _patch_run_stream(agent, trace_ctx, capture_input, capture_output, run_name)
        _patch_tools(agent, trace_ctx)
    except Exception as exc:
        warnings.warn(
            f"trulayer: failed to instrument PydanticAI agent: {exc}",
            stacklevel=2,
        )
    return agent


def _span_name(agent: Any, run_name: str | None) -> str:
    if run_name:
        return run_name
    agent_name = getattr(agent, "name", None) or "agent"
    return f"pydanticai:{agent_name}"


def _patch_run(
    agent: Any,
    trace_ctx: TraceContext,
    capture_input: bool,
    capture_output: bool,
    run_name: str | None,
) -> None:
    original_run = agent.run

    @functools.wraps(original_run)
    async def instrumented_run(user_prompt: Any, *, deps: Any = None, **kwargs: Any) -> Any:
        name = _span_name(agent, run_name)
        with trace_ctx.span(name, span_type="agent") as span:
            if capture_input:
                span.set_input(str(user_prompt))
            try:
                result = await original_run(user_prompt, deps=deps, **kwargs)
                if capture_output:
                    span.set_output(str(getattr(result, "data", result)))
                try:
                    usage = result.usage()
                    span.set_metadata(
                        usage={
                            "request_tokens": getattr(usage, "request_tokens", None),
                            "response_tokens": getattr(usage, "response_tokens", None),
                        }
                    )
                except Exception:
                    pass
                return result
            except Exception as exc:
                span.set_metadata(error=str(exc))
                raise

    agent.run = instrumented_run


def _patch_run_sync(
    agent: Any,
    trace_ctx: TraceContext,
    capture_input: bool,
    capture_output: bool,
    run_name: str | None,
) -> None:
    original_run_sync = agent.run_sync

    @functools.wraps(original_run_sync)
    def instrumented_run_sync(user_prompt: Any, *, deps: Any = None, **kwargs: Any) -> Any:
        name = _span_name(agent, run_name)
        with trace_ctx.span(name, span_type="agent") as span:
            if capture_input:
                span.set_input(str(user_prompt))
            try:
                result = original_run_sync(user_prompt, deps=deps, **kwargs)
                if capture_output:
                    span.set_output(str(getattr(result, "data", result)))
                try:
                    usage = result.usage()
                    span.set_metadata(
                        usage={
                            "request_tokens": getattr(usage, "request_tokens", None),
                            "response_tokens": getattr(usage, "response_tokens", None),
                        }
                    )
                except Exception:
                    pass
                return result
            except Exception as exc:
                span.set_metadata(error=str(exc))
                raise

    agent.run_sync = instrumented_run_sync


def _patch_run_stream(
    agent: Any,
    trace_ctx: TraceContext,
    capture_input: bool,
    capture_output: bool,
    run_name: str | None,
) -> None:
    original_run_stream = agent.run_stream

    @functools.wraps(original_run_stream)
    async def instrumented_run_stream(user_prompt: Any, **kwargs: Any) -> Any:
        name = _span_name(agent, run_name)
        span = trace_ctx.span(name, span_type="agent")
        span.__enter__()
        if capture_input:
            span.set_input(str(user_prompt))
        try:
            stream_result = await original_run_stream(user_prompt, **kwargs)
            # The span stays open; wrap the stream's iteration method
            # to close the span when the stream is fully consumed.
            _wrap_stream_iter(stream_result, span, capture_output)
            return stream_result
        except Exception as exc:
            span.set_metadata(error=str(exc))
            span.__exit__(type(exc), exc, exc.__traceback__)
            raise

    agent.run_stream = instrumented_run_stream


def _wrap_stream_iter(
    stream_result: Any, span: Any, capture_output: bool
) -> None:
    """Wrap the stream result's async iteration to close the span when done."""
    original_stream_response = getattr(stream_result, "stream_response", None)

    if original_stream_response is not None:
        accumulated: list[str] = []
        _closed = False

        async def patched_stream_response(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            nonlocal _closed
            try:
                async for chunk in original_stream_response(*args, **kwargs):
                    if isinstance(chunk, str):
                        accumulated.append(chunk)
                    yield chunk
            except Exception as exc:
                span.set_metadata(error=str(exc))
                if not _closed:
                    _closed = True
                    span.__exit__(type(exc), exc, exc.__traceback__)
                raise
            else:
                if capture_output and accumulated:
                    span.set_output("".join(accumulated))
                if not _closed:
                    _closed = True
                    span.__exit__(None, None, None)

        stream_result.stream_response = patched_stream_response
        return

    # No known iteration method; close span immediately
    span.__exit__(None, None, None)


def _patch_tools(agent: Any, trace_ctx: TraceContext) -> None:
    """Wrap each tool in agent._function_tools with a child span."""
    tools = getattr(agent, "_function_tools", None)
    if not isinstance(tools, dict):
        return

    for tool_name, tool_obj in tools.items():
        original_fn = getattr(tool_obj, "function", None)
        if original_fn is None:
            continue

        @functools.wraps(original_fn)
        async def _wrapped(
            *args: Any,
            _orig: Any = original_fn,
            _name: str = tool_name,
            **kwargs: Any,
        ) -> Any:
            with trace_ctx.span(_name, span_type="tool") as span:
                try:
                    if kwargs:
                        span.set_input(str(kwargs))
                    result = await _orig(*args, **kwargs)
                    span.set_output(str(result))
                    return result
                except Exception as exc:
                    span.set_metadata(error=str(exc))
                    raise

        tool_obj.function = _wrapped
