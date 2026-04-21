"""AutoGen ConversableAgent instrumentation for TruLayer auto-instrumentation."""

from __future__ import annotations

import functools
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trulayer.trace import TraceContext


def instrument_autogen(
    agent: Any,
    trace_ctx: TraceContext,
    *,
    capture_messages: bool = True,
) -> Any:
    """
    Wrap a ConversableAgent so ``initiate_chat()`` is recorded as a trace.
    Each reply in the conversation creates a child span.

    Returns the same agent instance (mutated in place via monkey-patching).
    Never raises into user code — catches instrumentation setup errors and warns.
    """
    try:
        _patch_initiate_chat(agent, trace_ctx, capture_messages)
        _patch_generate_reply(agent, trace_ctx, capture_messages)
    except Exception as exc:
        warnings.warn(
            f"trulayer: failed to instrument AutoGen agent: {exc}",
            stacklevel=2,
        )
    return agent


def _patch_initiate_chat(
    agent: Any,
    trace_ctx: TraceContext,
    capture_messages: bool,
) -> None:
    original_initiate_chat = agent.initiate_chat

    @functools.wraps(original_initiate_chat)
    def instrumented_initiate_chat(
        recipient: Any = None, message: Any = None, *args: Any, **kwargs: Any
    ) -> Any:
        with trace_ctx.span("autogen:chat", span_type="agent") as span:
            if capture_messages and message is not None:
                span.set_input(str(message))
            try:
                result = original_initiate_chat(recipient, message, *args, **kwargs)
                if capture_messages:
                    summary = getattr(result, "summary", None)
                    if summary is not None:
                        span.set_output(str(summary))
                    elif isinstance(result, dict) and "summary" in result:
                        span.set_output(str(result["summary"]))
                    else:
                        span.set_output(str(result))
                return result
            except Exception as exc:
                span.set_metadata(error=str(exc))
                warnings.warn(
                    f"trulayer: error during AutoGen initiate_chat instrumentation: {exc}",
                    stacklevel=2,
                )
                raise

    agent.initiate_chat = instrumented_initiate_chat


def _patch_generate_reply(
    agent: Any,
    trace_ctx: TraceContext,
    capture_messages: bool,
) -> None:
    original_generate_reply = agent.generate_reply

    agent_name = getattr(agent, "name", "agent")

    @functools.wraps(original_generate_reply)
    def instrumented_generate_reply(*args: Any, **kwargs: Any) -> Any:
        span_name = f"autogen:{agent_name}:reply"
        with trace_ctx.span(span_name, span_type="llm") as span:
            if capture_messages:
                messages = args[0] if args else kwargs.get("messages")
                if messages is not None:
                    span.set_input(str(messages))
            try:
                result = original_generate_reply(*args, **kwargs)
                if capture_messages:
                    span.set_output(str(result))
                return result
            except Exception as exc:
                span.set_metadata(error=str(exc))
                raise

    agent.generate_reply = instrumented_generate_reply
