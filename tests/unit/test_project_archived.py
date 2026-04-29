"""Tests for HTTP 403 permanent-disable behavior.

When the TruLayer API returns 403 — either the project-archived error code
or any generic 403 — the SDK must:

* not retry the request;
* drop any queued items;
* refuse to enqueue future items;
* surface a clear, actionable warning at ERROR level.
"""

from __future__ import annotations

import logging
import warnings
from unittest.mock import patch

import httpx
import pytest
import respx

from trulayer.batch import BatchSender
from trulayer.errors import ForbiddenError, ProjectArchivedError


def _make_sender(batch_size: int = 50) -> BatchSender:
    return BatchSender(
        api_key="tl_test",
        endpoint="https://api.trulayer.ai",
        batch_size=batch_size,
        flush_interval=60.0,
    )


# ---------------------------------------------------------------------------
# Project-archived 403
# ---------------------------------------------------------------------------


@respx.mock
async def test_403_project_archived_disables_sender_without_retry(
    caplog: pytest.LogCaptureFixture,
) -> None:
    route = respx.post("https://api.trulayer.ai/v1/ingest/batch").mock(
        return_value=httpx.Response(
            403,
            json={
                "code": "error.project.archived",
                "message": (
                    "The project associated with this API key has been "
                    "archived. Unarchive the project to resume ingestion."
                ),
            },
        )
    )
    sender = _make_sender()

    with (
        caplog.at_level(logging.ERROR, logger="trulayer.batch"),
        warnings.catch_warnings(record=True) as w,
    ):
        warnings.simplefilter("always")
        await sender._send_with_retry([{"id": "trace-1"}])

    assert route.call_count == 1  # no retry
    assert sender.disabled is True
    assert isinstance(sender.fatal_error, ProjectArchivedError)
    assert any("Project is archived" in str(msg.message) for msg in w)
    assert any(
        record.levelno == logging.ERROR and "Project is archived" in record.message
        for record in caplog.records
    )


@respx.mock
async def test_403_project_archived_accepts_error_field() -> None:
    """``error`` field should also match for forward compatibility."""
    respx.post("https://api.trulayer.ai/v1/ingest/batch").mock(
        return_value=httpx.Response(403, json={"error": "error.project.archived"})
    )
    sender = _make_sender()
    await sender._send_with_retry([{"id": "trace-1"}])
    assert isinstance(sender.fatal_error, ProjectArchivedError)


@respx.mock
async def test_403_project_archived_drops_queued_items() -> None:
    respx.post("https://api.trulayer.ai/v1/ingest/batch").mock(
        return_value=httpx.Response(403, json={"code": "error.project.archived"})
    )
    sender = _make_sender()
    sender._queue.put_nowait({"id": "queued-1"})
    sender._queue.put_nowait({"id": "queued-2"})
    await sender._send_with_retry([{"id": "trace-1"}])
    assert sender._queue.qsize() == 0


async def test_enqueue_drops_after_403() -> None:
    sender = _make_sender(batch_size=1)
    sender._fatal_error = ProjectArchivedError()
    sender._disabled = True

    sender.enqueue({"id": "trace-1"})
    sender.enqueue({"id": "trace-2"})
    assert sender._queue.qsize() == 0


@respx.mock
async def test_flush_is_noop_after_403() -> None:
    route = respx.post("https://api.trulayer.ai/v1/ingest/batch").mock(
        return_value=httpx.Response(200)
    )
    sender = _make_sender()
    sender._fatal_error = ProjectArchivedError()
    sender._disabled = True
    sender._queue.put_nowait({"id": "trace-x"})
    await sender._flush()
    assert not route.called
    assert sender._queue.qsize() == 0


# ---------------------------------------------------------------------------
# Generic 403
# ---------------------------------------------------------------------------


@respx.mock
async def test_403_generic_disables_sender(
    caplog: pytest.LogCaptureFixture,
) -> None:
    route = respx.post("https://api.trulayer.ai/v1/ingest/batch").mock(
        return_value=httpx.Response(403, json={"code": "forbidden"})
    )
    sender = _make_sender()

    with (
        caplog.at_level(logging.ERROR, logger="trulayer.batch"),
        warnings.catch_warnings(record=True) as w,
    ):
        warnings.simplefilter("always")
        await sender._send_with_retry([{"id": "trace-1"}])

    assert route.call_count == 1
    assert sender.disabled is True
    assert isinstance(sender.fatal_error, ForbiddenError)
    assert any("HTTP 403" in str(msg.message) for msg in w)
    assert any(
        record.levelno == logging.ERROR and "HTTP 403" in record.message
        for record in caplog.records
    )


@respx.mock
async def test_403_with_unparseable_body_disables_sender() -> None:
    """A 403 with a non-JSON body still latches into ForbiddenError."""
    respx.post("https://api.trulayer.ai/v1/ingest/batch").mock(
        return_value=httpx.Response(403, text="not-json")
    )
    sender = _make_sender()
    await sender._send_with_retry([{"id": "trace-1"}])
    assert sender.disabled is True
    assert isinstance(sender.fatal_error, ForbiddenError)


# ---------------------------------------------------------------------------
# Non-403 errors do NOT disable
# ---------------------------------------------------------------------------


@respx.mock
async def test_500_does_not_disable_sender() -> None:
    respx.post("https://api.trulayer.ai/v1/ingest/batch").mock(return_value=httpx.Response(500))
    sender = _make_sender()
    with patch("trulayer.batch._RETRY_BASE_DELAY", 0.0):
        await sender._send_with_retry([{"id": "trace-1"}])
    assert sender.disabled is False
    assert sender.fatal_error is None


@respx.mock
async def test_400_does_not_disable_sender() -> None:
    respx.post("https://api.trulayer.ai/v1/ingest/batch").mock(
        return_value=httpx.Response(400, json={"error": "bad_request"})
    )
    sender = _make_sender()
    with patch("trulayer.batch._RETRY_BASE_DELAY", 0.0):
        await sender._send_with_retry([{"id": "trace-1"}])
    assert sender.disabled is False
    assert sender.fatal_error is None


@respx.mock
async def test_422_does_not_disable_sender() -> None:
    respx.post("https://api.trulayer.ai/v1/ingest/batch").mock(
        return_value=httpx.Response(422, json={"error": "unprocessable"})
    )
    sender = _make_sender()
    with patch("trulayer.batch._RETRY_BASE_DELAY", 0.0):
        await sender._send_with_retry([{"id": "trace-1"}])
    assert sender.disabled is False
    assert sender.fatal_error is None


# ---------------------------------------------------------------------------
# Instance scope
# ---------------------------------------------------------------------------


def test_disabled_flag_is_instance_scoped() -> None:
    """A fresh sender instance always starts with ``disabled=False``."""
    s1 = _make_sender()
    s1._disabled = True
    s1._fatal_error = ProjectArchivedError()

    s2 = _make_sender()
    assert s2.disabled is False
    assert s2.fatal_error is None


# ---------------------------------------------------------------------------
# Error shape
# ---------------------------------------------------------------------------


def test_project_archived_error_message() -> None:
    err = ProjectArchivedError()
    assert "archived" in str(err).lower()
    assert "app.trulayer.ai" in str(err)


def test_forbidden_error_message() -> None:
    err = ForbiddenError()
    assert "403" in str(err)
    assert "app.trulayer.ai" in str(err)
