from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading
import warnings
from typing import Any

import httpx

from trulayer.errors import (
    InvalidAPIKeyError,
    TruLayerFlushError,
    parse_invalid_api_key_payload,
)

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 0.5  # seconds


def _fail_mode_is_block() -> bool:
    """Return True iff ``TRULAYER_FAIL_MODE=block`` is set in the environment."""
    return os.environ.get("TRULAYER_FAIL_MODE", "").strip().lower() == "block"


class BatchSender:
    """
    Thread-safe batch sender. Producers call enqueue() from any thread.
    A dedicated daemon thread runs the asyncio flush loop.
    """

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        batch_size: int = 50,
        flush_interval: float = 2.0,
    ) -> None:
        self._api_key = api_key
        self._endpoint = endpoint.rstrip("/")
        self._batch_size = batch_size
        self._flush_interval = flush_interval

        self._queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Latched when the API reports a permanent credential failure. Once
        # set, the sender drops all queued and future events — retrying would
        # waste the backend's time and cannot succeed.
        self._fatal_error: InvalidAPIKeyError | None = None
        # When True, flush failures are raised as TruLayerFlushError via the
        # awaited flush future instead of dropped with a warning. Operators
        # opt in by setting TRULAYER_FAIL_MODE=block. Read once at
        # construction time so runtime mutation of the env var can't flip
        # behavior mid-process.
        self._fail_mode_block = _fail_mode_is_block()
        # Tracks whether we've already warned about a drop in the current
        # failure window. Cleared on a successful send so the next outage
        # produces exactly one warning, not one per batch.
        self._drop_warned = False

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="trulayer-batch")
        self._thread.start()

    def enqueue(self, item: dict[str, Any]) -> None:
        if self._fatal_error is not None:
            return
        self._queue.put_nowait(item)
        if self._queue.qsize() >= self._batch_size and self._loop:
            asyncio.run_coroutine_threadsafe(self._flush(), self._loop)

    def shutdown(self, timeout: float = 5.0) -> None:
        pending_error: TruLayerFlushError | None = None
        if self._loop and self._thread and self._thread.is_alive():
            if self._fatal_error is None:
                future = asyncio.run_coroutine_threadsafe(self._flush(), self._loop)
                try:
                    future.result(timeout=timeout)
                except TruLayerFlushError as exc:
                    # In fail-mode=block, surface the error to the caller
                    # after we've cleanly stopped the background loop.
                    pending_error = exc
                except Exception as exc:
                    warnings.warn(f"trulayer: flush on shutdown failed: {exc}", stacklevel=2)
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=timeout)
        if pending_error is not None:
            raise pending_error

    @property
    def fatal_error(self) -> InvalidAPIKeyError | None:
        """The latched non-retryable error, if any.

        Exposed for tests and for callers that want to surface configuration
        failures proactively.
        """
        return self._fatal_error

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._flush_loop())
        except RuntimeError:
            pass  # raised by loop.stop() during shutdown — expected
        finally:
            self._loop.close()

    async def _flush_loop(self) -> None:
        while True:
            await asyncio.sleep(self._flush_interval)
            if self._fatal_error is not None:
                # Nothing more to do — keep the loop alive so shutdown() can stop it cleanly.
                continue
            await self._flush()

    async def _flush(self) -> None:
        if self._fatal_error is not None:
            self._drain_queue()
            return
        items: list[dict[str, Any]] = []
        while True:
            try:
                items.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if not items:
            return
        await self._send_with_retry(items)

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    async def _send_with_retry(self, items: list[dict[str, Any]]) -> None:
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        f"{self._endpoint}/v1/ingest/batch",
                        json={"traces": items},
                        headers={"Authorization": f"Bearer {self._api_key}"},
                    )
                    if resp.status_code == 401:
                        payload: Any = None
                        try:
                            payload = resp.json()
                        except Exception:
                            payload = None
                        code = parse_invalid_api_key_payload(payload)
                        if code is not None:
                            self._fatal_error = InvalidAPIKeyError(code)
                            self._drain_queue()
                            warnings.warn(
                                f"trulayer: {self._fatal_error} (code: {code}) "
                                f"— halting trace submission for this client.",
                                stacklevel=2,
                            )
                            return
                    resp.raise_for_status()
                    # Success — reset the one-warning-per-window latch so the
                    # next outage produces exactly one warning.
                    self._drop_warned = False
                    return
            except Exception as exc:
                last_exc = exc
                if attempt == _MAX_RETRIES - 1:
                    if self._fail_mode_block:
                        raise TruLayerFlushError(
                            f"trulayer: failed to send batch of {len(items)} items after "
                            f"{_MAX_RETRIES} retries"
                        ) from exc
                    if not self._drop_warned:
                        warnings.warn(
                            f"trulayer: failed to send batch of {len(items)} items after "
                            f"{_MAX_RETRIES} retries: {exc} "
                            f"(further drops in this failure window will be silent "
                            f"until the next successful send)",
                            stacklevel=2,
                        )
                        self._drop_warned = True
                    return
                delay = _RETRY_BASE_DELAY * (2**attempt)
                logger.debug("trulayer: retry %d after %.1fs: %s", attempt + 1, delay, exc)
                await asyncio.sleep(delay)
        # Defensive: retry loop should always return or raise above.
        if last_exc is not None and self._fail_mode_block:
            raise TruLayerFlushError("trulayer: flush failed") from last_exc
