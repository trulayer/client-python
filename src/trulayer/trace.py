from __future__ import annotations

import contextvars
import random
import time
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from types import TracebackType
from typing import TYPE_CHECKING, Any

from trulayer.model import SpanData, TraceData

if TYPE_CHECKING:
    from trulayer.client import TruLayerClient

_current_trace: contextvars.ContextVar[TraceContext | None] = contextvars.ContextVar(
    "trulayer_current_trace", default=None
)


def _now() -> datetime:
    return datetime.now(tz=UTC)


_SCRUB_FIELDS = ("input", "output", "error")


def _validate_metadata(
    payload: dict[str, Any],
    validator: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    """Apply validator to trace metadata and each span's metadata.
    On failure, replace that metadata dict with {} and emit a warning.
    """
    import warnings as _warnings  # noqa: PLC0415

    for item in [payload, *payload.get("spans", [])]:
        meta = item.get("metadata")
        if not isinstance(meta, dict):
            continue
        try:
            validator(meta)
        except Exception as exc:
            _warnings.warn(
                f"trulayer: metadata validation failed, clearing metadata: {exc}",
                stacklevel=2,
            )
            item["metadata"] = {}
    return payload


def _scrub_payload(payload: dict[str, Any], fn: Callable[[str], str]) -> dict[str, Any]:
    for field in _SCRUB_FIELDS:
        if isinstance(payload.get(field), str):
            payload[field] = fn(payload[field])
    for span in payload.get("spans", []):
        for field in _SCRUB_FIELDS:
            if isinstance(span.get(field), str):
                span[field] = fn(span[field])
    return payload


class SpanContext:
    """Sync context manager for a single span within a trace."""

    def __init__(self, trace_ctx: TraceContext, name: str, span_type: str = "other") -> None:
        self._trace = trace_ctx
        self._data = SpanData(trace_id=trace_ctx._data.id, name=name, span_type=span_type)
        self._start_ns = 0

    def set_input(self, value: str) -> None:
        self._data.input = value

    def set_output(self, value: str) -> None:
        self._data.output = value

    def set_model(self, model: str) -> None:
        self._data.model = model

    def set_tokens(self, prompt: int | None = None, completion: int | None = None) -> None:
        self._data.prompt_tokens = prompt
        self._data.completion_tokens = completion

    def set_metadata(self, **kwargs: Any) -> None:
        self._data.metadata.update(kwargs)

    def __enter__(self) -> SpanContext:
        self._start_ns = time.monotonic_ns()
        self._data.started_at = _now()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        elapsed = time.monotonic_ns() - self._start_ns
        self._data.latency_ms = elapsed // 1_000_000
        self._data.ended_at = _now()
        if exc_type is not None:
            self._data.error = "".join(
                traceback.format_exception(exc_type, exc_val, exc_tb)
            )
        self._trace._data.spans.append(self._data)

    async def __aenter__(self) -> SpanContext:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)


class TraceContext:
    """Sync context manager for a full trace. Flushes to batch sender on exit."""

    def __init__(
        self,
        client: TruLayerClient,
        name: str | None = None,
        session_id: str | None = None,
        external_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        tag_map: dict[str, str] | None = None,
    ) -> None:
        self._client = client
        self._data = TraceData(
            project_id=client._project_id,
            name=name,
            session_id=session_id,
            external_id=external_id,
            tags=tags or [],
            tag_map=dict(tag_map) if tag_map is not None else None,
            metadata=metadata or {},
        )
        self._start_ns = 0
        self._sampled = True
        self._token: contextvars.Token[TraceContext | None] | None = None

    def span(self, name: str, span_type: str = "other") -> SpanContext:
        return SpanContext(self, name, span_type)

    def set_input(self, value: str) -> None:
        self._data.input = value

    def set_output(self, value: str) -> None:
        self._data.output = value

    def set_model(self, model: str) -> None:
        self._data.model = model

    def set_cost(self, cost: float) -> None:
        self._data.cost = cost

    def set_metadata(self, **kwargs: Any) -> None:
        self._data.metadata.update(kwargs)

    def add_tag(self, tag: str) -> None:
        self._data.tags.append(tag)

    def set_tag(self, key: str, value: str) -> None:
        """Attach a structured key -> value tag to the trace.

        Unlike :meth:`add_tag`, these are indexed server-side and filterable
        via the ``tag_key`` / ``tag_value`` parameters on list endpoints.

        Limits: max 20 keys per trace, 64 characters per key and value.
        Values exceeding these limits are accepted client-side and rejected
        by the server with a 400 response.
        """
        if self._data.tag_map is None:
            self._data.tag_map = {}
        self._data.tag_map[key] = value

    def __enter__(self) -> TraceContext:
        try:
            sample_rate = float(self._client._sample_rate)
        except (AttributeError, TypeError, ValueError):
            sample_rate = 1.0
        self._sampled = random.random() < sample_rate
        self._start_ns = time.monotonic_ns()
        self._data.started_at = _now()
        self._token = _current_trace.set(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        elapsed = time.monotonic_ns() - self._start_ns
        self._data.ended_at = _now()
        if self._data.latency_ms is None:
            self._data.latency_ms = elapsed // 1_000_000
        if exc_type is not None:
            self._data.error = "".join(
                traceback.format_exception(exc_type, exc_val, exc_tb)
            )
        if self._token is not None:
            _current_trace.reset(self._token)
        if not self._sampled:
            return
        # Fire-and-forget: enqueue to batch sender, never raise
        try:
            payload = self._data.to_wire()
            scrub_fn = getattr(self._client, "_scrub_fn", None)
            if scrub_fn is not None:
                payload = _scrub_payload(payload, scrub_fn)
            metadata_validator = getattr(self._client, "_metadata_validator", None)
            if metadata_validator is not None:
                payload = _validate_metadata(payload, metadata_validator)
            self._client._batch.enqueue(payload)
        except Exception:
            pass

    async def __aenter__(self) -> TraceContext:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)


def current_trace() -> TraceContext | None:
    """Return the active TraceContext in the current context, or None."""
    return _current_trace.get()
