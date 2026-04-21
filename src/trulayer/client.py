from __future__ import annotations

import atexit
import os
import warnings
from collections.abc import Callable
from typing import Any

import httpx

from trulayer.batch import BatchSender
from trulayer.model import FeedbackData
from trulayer.trace import TraceContext

_DEFAULT_ENDPOINT = "https://api.trulayer.ai"


class TruLayerClient:
    """
    Main SDK client. Initialize once per process.

    Usage:
        tl = TruLayerClient(api_key="tl_...", project_name="proj_...")
        with tl.trace("my-operation") as t:
            t.set_input("hello")
            with t.span("llm-call") as span:
                result = openai.chat.completions.create(...)
                span.set_output(result.choices[0].message.content)
            t.set_output("world")

    Set ``TRULAYER_MODE=local`` to capture traces in-memory without sending
    them to the API (no API key required).  Useful for CI and offline testing.
    """

    def __init__(
        self,
        api_key: str = "",
        project_name: str | None = None,
        endpoint: str = _DEFAULT_ENDPOINT,
        batch_size: int = 50,
        flush_interval: float = 2.0,
        sample_rate: float = 1.0,
        scrub_fn: Callable[[str], str] | None = None,
        metadata_validator: Callable[[dict[str, Any]], None] | None = None,
        project_id: str | None = None,  # deprecated alias
        _sender: Any = None,
    ) -> None:
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError(f"sample_rate must be between 0.0 and 1.0, got {sample_rate}")

        local_mode = _sender is not None or os.environ.get("TRULAYER_MODE") == "local"

        name = project_name or project_id
        if not name:
            if local_mode:
                name = "local"
            else:
                raise TypeError("trulayer: project_name is required")
        if project_id and not project_name:
            warnings.warn(
                "trulayer: `project_id` is deprecated; rename to `project_name`. "
                "Removed in 0.3.x.",
                DeprecationWarning,
                stacklevel=2,
            )
        self._api_key = api_key
        # Stored under _project_id for wire-format compatibility — the backend
        # currently accepts a free-form string in the project_id slot and
        # resolves it from the API key.
        self._project_id = name
        self._project_name = name
        self._endpoint = endpoint
        self._sample_rate = sample_rate
        self._scrub_fn = scrub_fn
        self._metadata_validator = metadata_validator

        if _sender is not None:
            self._batch = _sender
        elif os.environ.get("TRULAYER_MODE") == "local":
            from trulayer.local_batch import LocalBatchSender  # noqa: PLC0415

            self._batch = LocalBatchSender()
            warnings.warn(
                "[trulayer] running in LOCAL mode — no data will be sent to the API",
                stacklevel=2,
            )
        else:
            self._batch = BatchSender(
                api_key=api_key,
                endpoint=endpoint,
                batch_size=batch_size,
                flush_interval=flush_interval,
            )
        self._batch.start()
        atexit.register(self.shutdown)

    def trace(
        self,
        name: str | None = None,
        session_id: str | None = None,
        external_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TraceContext:
        return TraceContext(
            self,
            name=name,
            session_id=session_id,
            external_id=external_id,
            tags=tags,
            metadata=metadata,
        )

    def feedback(
        self,
        trace_id: str,
        label: str,
        score: float | None = None,
        comment: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Submit feedback for a trace. Fire-and-forget — never raises."""
        fb = FeedbackData(
            trace_id=trace_id,
            label=label,
            score=score,
            comment=comment,
            metadata=metadata or {},
        )
        try:
            resp = httpx.post(
                f"{self._endpoint}/v1/feedback",
                json=fb.model_dump(mode="json"),
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=5.0,
            )
            resp.raise_for_status()
        except Exception as exc:
            warnings.warn(f"trulayer: feedback submission failed: {exc}", stacklevel=2)

    def flush(self, timeout: float = 5.0) -> None:
        """Flush buffered events. Blocks until complete or timeout."""
        self._batch.shutdown(timeout=timeout)
        self._batch.start()

    def shutdown(self, timeout: float = 5.0) -> None:
        """Flush and shut down the batch sender. Called automatically on process exit."""
        self._batch.shutdown(timeout=timeout)
