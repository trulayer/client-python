"""LangChain callback handler for TruLayer auto-instrumentation."""

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING, Any
from uuid import UUID

if TYPE_CHECKING:
    from trulayer.client import TruLayerClient


def instrument_langchain(client: TruLayerClient) -> TruLayerCallbackHandler:
    """
    Return a LangChain callback handler that records LLM calls as TruLayer spans.

    Pass the returned handler to any LangChain LLM, chat model, or chain:

        handler = instrument_langchain(client)
        llm = ChatOpenAI(callbacks=[handler])
        chain = my_chain | {"callbacks": [handler]}
    """
    try:
        from langchain_core.callbacks import BaseCallbackHandler  # noqa: PLC0415
    except ImportError:
        try:
            from langchain.callbacks import BaseCallbackHandler  # noqa: PLC0415, I001
        except ImportError:
            warnings.warn(
                "trulayer: langchain_core not installed. "
                "Install it with: pip install langchain-core",
                stacklevel=2,
            )
            raise

    return TruLayerCallbackHandler(client, BaseCallbackHandler)


class TruLayerCallbackHandler:
    """
    LangChain BaseCallbackHandler subclass that captures LLM calls as TruLayer spans.

    Works with LangChain v0.1+ (langchain_core) and v0.2+.
    Handles both raw LLMs (on_llm_start) and chat models (on_chat_model_start).
    """

    def __new__(
        cls,
        client: TruLayerClient,
        base_class: type,
    ) -> TruLayerCallbackHandler:
        # Dynamically subclass BaseCallbackHandler so the import happens at call time,
        # not at module load time (keeps langchain optional).
        handler_cls = type(
            "TruLayerCallbackHandlerImpl",
            (base_class,),
            {
                "_tl_client": client,
                "_tl_starts": {},  # run_id -> (start_time, model_name, input_text)
                "on_llm_start": _on_llm_start,
                "on_chat_model_start": _on_chat_model_start,
                "on_llm_end": _on_llm_end,
                "on_llm_error": _on_llm_error,
            },
        )
        return handler_cls()  # type: ignore[no-any-return]


def _on_llm_start(
    self: Any,
    serialized: dict[str, Any],
    prompts: list[str],
    *,
    run_id: UUID,
    **kwargs: Any,
) -> None:
    model_name = _extract_model(serialized, kwargs)
    input_text = prompts[0] if prompts else ""
    self._tl_starts[run_id] = (time.monotonic(), model_name, input_text)


def _on_chat_model_start(
    self: Any,
    serialized: dict[str, Any],
    messages: list[list[Any]],
    *,
    run_id: UUID,
    **kwargs: Any,
) -> None:
    model_name = _extract_model(serialized, kwargs)
    input_text = _extract_chat_input(messages)
    self._tl_starts[run_id] = (time.monotonic(), model_name, input_text)


def _on_llm_end(self: Any, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
    entry = self._tl_starts.pop(run_id, None)
    if entry is None:
        return

    start_time, model_name, input_text = entry
    elapsed = time.monotonic() - start_time

    try:
        from trulayer.trace import current_trace  # noqa: PLC0415

        trace = current_trace()
        if trace is None:
            return

        output = ""
        prompt_tokens: int | None = None
        completion_tokens: int | None = None

        try:
            gen = response.generations
            if gen and gen[0]:
                first = gen[0][0]
                # ChatGeneration has .text or .message.content
                output = getattr(first, "text", None) or ""
                if not output:
                    msg = getattr(first, "message", None)
                    output = getattr(msg, "content", "") or ""
        except Exception:
            pass

        try:
            usage = getattr(response, "llm_output", {}) or {}
            token_usage = usage.get("token_usage") or usage.get("usage", {})
            prompt_tokens = (
                token_usage.get("prompt_tokens")
                or token_usage.get("input_tokens")
            )
            completion_tokens = (
                token_usage.get("completion_tokens")
                or token_usage.get("output_tokens")
            )
        except Exception:
            pass

        with trace.span("langchain.llm", span_type="llm") as span:
            span.set_input(input_text)
            span.set_output(output)
            if model_name:
                span.set_model(model_name)
            span.set_tokens(prompt=prompt_tokens, completion=completion_tokens)
            span._data.latency_ms = int(elapsed * 1000)

    except Exception as exc:
        warnings.warn(f"trulayer: failed to record LangChain span: {exc}", stacklevel=2)


def _on_llm_error(
    self: Any,
    error: BaseException,
    *,
    run_id: UUID,
    **kwargs: Any,
) -> None:
    self._tl_starts.pop(run_id, None)


def _extract_model(serialized: dict[str, Any], kwargs: dict[str, Any]) -> str:
    # LangChain stores model name in serialized["kwargs"]["model_name"] or
    # serialized["kwargs"]["model"] or invocation_params
    try:
        sk = serialized.get("kwargs", {})
        return (
            sk.get("model_name")
            or sk.get("model")
            or kwargs.get("invocation_params", {}).get("model_name")
            or kwargs.get("invocation_params", {}).get("model")
            or ""
        )
    except Exception:
        return ""


def _extract_chat_input(messages: list[list[Any]]) -> str:
    try:
        flat = messages[-1] if messages else []
        last = flat[-1] if flat else None
        if last is None:
            return ""
        # HumanMessage / AIMessage have .content attribute
        content = getattr(last, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # multimodal content blocks
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    return str(block.get("text", ""))
        return ""
    except Exception:
        return ""
