"""Test helpers for TruLayer SDK consumers.

Provides :func:`create_test_client` for quick in-memory setup and
:func:`assert_sender` for fluent assertions on captured traces and spans.

Example::

    from trulayer.testing import create_test_client, assert_sender

    client, sender = create_test_client()
    with client.trace("my-test") as trace:
        with trace.span("step", "llm") as span:
            span.set_input("hello")
            span.set_output("world")
    client.flush()

    assert len(sender.spans) == 1
    assert_sender(sender).has_trace().span_count(1).has_span_named("step")
"""

from __future__ import annotations

from typing import Any

from trulayer.client import TruLayerClient
from trulayer.local_batch import LocalBatchSender


def create_test_client(**kwargs: Any) -> tuple[TruLayerClient, LocalBatchSender]:
    """Return a ``(client, sender)`` pair for unit tests.

    No API key needed.  No network I/O.
    """
    sender = LocalBatchSender()
    client = TruLayerClient(api_key="test-key", _sender=sender, **kwargs)
    return client, sender


class SenderAssertions:
    """Fluent assertions on a :class:`LocalBatchSender`."""

    def __init__(self, sender: LocalBatchSender) -> None:
        self._sender = sender

    def has_trace(self, trace_id: str | None = None) -> SenderAssertions:
        if trace_id:
            ids = [t.get("trace_id", t.get("id")) for t in self._sender.traces]
            if trace_id not in ids:
                raise AssertionError(f"Expected trace {trace_id!r}, found: {ids}")
        elif not self._sender.traces:
            raise AssertionError("Expected at least one trace, found none")
        return self

    def span_count(self, n: int) -> SenderAssertions:
        actual = len(self._sender.spans)
        if actual != n:
            raise AssertionError(f"Expected {n} spans, got {actual}")
        return self

    def has_span_named(self, name: str) -> SenderAssertions:
        names = [s.get("name") for s in self._sender.spans]
        if name not in names:
            raise AssertionError(f"Expected span named {name!r}, found: {names}")
        return self


def assert_sender(sender: LocalBatchSender) -> SenderAssertions:
    """Return a :class:`SenderAssertions` wrapper for fluent checking."""
    return SenderAssertions(sender)
