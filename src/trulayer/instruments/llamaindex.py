"""LlamaIndex callback handler for TruLayer auto-instrumentation."""

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trulayer.client import TruLayerClient

try:
    from llama_index.core.callbacks import CallbackManager as _  # noqa: F401
    from llama_index.core.callbacks import CBEventType
    from llama_index.core.callbacks.base_handler import BaseCallbackHandler
except ImportError as _imp_err:
    raise ImportError(
        "llama_index is required for LlamaIndex instrumentation. "
        "Install it with: pip install llama-index-core"
    ) from _imp_err


_EVENT_TYPE_MAP: dict[CBEventType, str] = {
    CBEventType.LLM: "llm",
    CBEventType.QUERY: "retrieval",
    CBEventType.RETRIEVE: "retrieval",
}


class TruLayerCallbackHandler(BaseCallbackHandler):  # type: ignore[misc]
    """
    LlamaIndex BaseCallbackHandler that captures query and LLM events as TruLayer spans.

    Works with llama-index-core 0.10+.
    """

    def __init__(
        self,
        client: TruLayerClient,
        event_starts_to_ignore: list[CBEventType] | None = None,
        event_ends_to_ignore: list[CBEventType] | None = None,
    ) -> None:
        super().__init__(
            event_starts_to_ignore=event_starts_to_ignore or [],
            event_ends_to_ignore=event_ends_to_ignore or [],
        )
        self._tl_client = client
        self._open_spans: dict[str, tuple[float, str, str]] = {}
        # event_id -> (start_monotonic, span_type, input_text)

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        span_type = _EVENT_TYPE_MAP.get(event_type, "other")
        input_text = ""

        if payload is not None:
            if event_type == CBEventType.LLM:
                # LlamaIndex stores messages under EventPayload.MESSAGES
                messages = payload.get("messages") or payload.get("template_args")
                if messages:
                    input_text = str(messages)
            elif event_type in (CBEventType.QUERY, CBEventType.RETRIEVE):
                query = payload.get("query_str") or payload.get("query_bundle")
                if query:
                    input_text = str(query)

        self._open_spans[event_id] = (time.monotonic(), span_type, input_text)
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        entry = self._open_spans.pop(event_id, None)
        if entry is None:
            return

        start_time, span_type, input_text = entry
        elapsed = time.monotonic() - start_time

        output_text = ""
        if payload is not None:
            if event_type == CBEventType.LLM:
                resp = payload.get("response") or payload.get("completion")
                if resp:
                    output_text = str(resp)
            elif event_type in (CBEventType.QUERY, CBEventType.RETRIEVE):
                resp = payload.get("response") or payload.get("nodes")
                if resp:
                    output_text = str(resp)
            else:
                resp = payload.get("response")
                if resp:
                    output_text = str(resp)

        try:
            from trulayer.trace import current_trace  # noqa: PLC0415

            trace = current_trace()
            if trace is None:
                return

            span_name = f"llamaindex.{span_type}"
            with trace.span(span_name, span_type=span_type) as span:
                span.set_input(input_text)
                span.set_output(output_text)
                span._data.latency_ms = int(elapsed * 1000)

        except Exception as exc:
            warnings.warn(
                f"trulayer: failed to record LlamaIndex span: {exc}",
                stacklevel=2,
            )

    def start_trace(self, trace_id: str | None = None) -> None:
        """Called when a LlamaIndex trace starts. No-op for TruLayer."""

    def end_trace(
        self,
        trace_id: str | None = None,
        trace_map: dict[str, list[str]] | None = None,
    ) -> None:
        """Called when a LlamaIndex trace ends. No-op for TruLayer."""
