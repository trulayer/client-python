"""Span timing tests across instruments.

These tests guard the invariant the latency-waterfall depends on: every
recorded span must carry a wall-clock ``start_time`` captured *before* the
upstream call begins and an ``end_time`` consistent with it. When spans are
flushed in a batch after the trace completes the dashboard relies on these
absolute timestamps to position waterfall bars; without the fix, both
timestamps would collapse to the moment the SDK happened to construct the
span model — typically *after* the call returned.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

from trulayer.instruments.anthropic import _record_span as anthropic_record
from trulayer.instruments.openai import _record_span as openai_record
from trulayer.trace import TraceContext


def _make_client(project_id: str = "proj-1") -> MagicMock:
    client = MagicMock()
    client._project_id = project_id
    client._batch = MagicMock()
    client._scrub_fn = None
    client._sample_rate = 1.0
    client._metadata_validator = None
    return client


def _openai_response() -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )


def _anthropic_response() -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hi")],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )


def _parse(iso: str) -> datetime:
    # `datetime.fromisoformat` accepts trailing "+00:00" but not "Z"; Pydantic
    # serializes UTC as "...+00:00" so this round-trip is safe.
    return datetime.fromisoformat(iso)


def test_openai_record_span_uses_caller_supplied_start_wall() -> None:
    """``_record_span`` honours the wall-clock anchor captured before the call."""
    client = _make_client()
    elapsed_s = 0.250
    # Pretend the upstream call started 5 seconds ago and took 250ms.
    start_wall = datetime(2026, 5, 14, 12, 0, 0, tzinfo=UTC)

    with TraceContext(client, name="test"):
        openai_record(
            client,
            {"model": "gpt-4o", "messages": [{"content": "hi"}]},
            _openai_response(),
            elapsed_s,
            start_wall=start_wall,
        )

    payload = client._batch.enqueue.call_args[0][0]
    span = payload["spans"][0]

    assert _parse(span["start_time"]) == start_wall
    # end_time == start_time + latency_ms — and is strictly after start_time.
    end = _parse(span["end_time"])
    assert end > _parse(span["start_time"])
    assert span["latency_ms"] == int(elapsed_s * 1000)
    delta_ms = (end - start_wall).total_seconds() * 1000
    assert abs(delta_ms - span["latency_ms"]) < 1.0


def test_anthropic_record_span_uses_caller_supplied_start_wall() -> None:
    client = _make_client()
    elapsed_s = 0.4
    start_wall = datetime(2026, 5, 14, 12, 0, 0, tzinfo=UTC)

    with TraceContext(client, name="test"):
        anthropic_record(
            client,
            {"model": "claude-3", "messages": [{"content": "hi"}]},
            _anthropic_response(),
            elapsed_s,
            start_wall=start_wall,
        )

    payload = client._batch.enqueue.call_args[0][0]
    span = payload["spans"][0]
    assert _parse(span["start_time"]) == start_wall
    assert _parse(span["end_time"]) > _parse(span["start_time"])
    assert span["latency_ms"] == int(elapsed_s * 1000)


def test_openai_record_span_start_precedes_record_call() -> None:
    """Realistic flow: capture start_wall, simulate a slow call, then record.

    The recorded ``start_time`` must equal the captured wall-clock — *not*
    drift to the moment we constructed the span — proving the fix.
    """
    client = _make_client()
    captured = datetime.now(tz=UTC)
    time.sleep(0.05)  # simulate upstream latency
    elapsed_s = (datetime.now(tz=UTC) - captured).total_seconds()

    with TraceContext(client, name="test"):
        openai_record(
            client,
            {"model": "gpt-4o", "messages": [{"content": "hi"}]},
            _openai_response(),
            elapsed_s,
            start_wall=captured,
        )

    payload = client._batch.enqueue.call_args[0][0]
    span = payload["spans"][0]
    recorded_start = _parse(span["start_time"])
    # Recorded start must equal what we captured before the call, not "now".
    assert recorded_start == captured
    # And the gap to "now" must be at least the sleep duration we forced.
    assert (datetime.now(tz=UTC) - recorded_start).total_seconds() >= 0.05


def test_openai_record_span_default_start_wall_is_close_to_now() -> None:
    """When callers omit ``start_wall`` the fallback reconstructs from elapsed."""
    client = _make_client()
    elapsed_s = 0.1
    before = datetime.now(tz=UTC)

    with TraceContext(client, name="test"):
        openai_record(
            client,
            {"model": "gpt-4o", "messages": [{"content": "hi"}]},
            _openai_response(),
            elapsed_s,
        )

    payload = client._batch.enqueue.call_args[0][0]
    span = payload["spans"][0]
    started = _parse(span["start_time"])
    ended = _parse(span["end_time"])
    # Fallback: start ~= now - elapsed. Allow generous slack for slow CI.
    assert (before - started).total_seconds() <= elapsed_s + 1.0
    assert ended >= started
