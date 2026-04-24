"""Tests for TRULAYER_FAIL_MODE drop/block behavior (TRU-233)."""

from __future__ import annotations

import os
import warnings
from unittest import mock
from unittest.mock import patch

import httpx
import pytest
import respx

from trulayer.batch import BatchSender
from trulayer.errors import TruLayerFlushError


def _make_sender() -> BatchSender:
    return BatchSender(
        api_key="tl_test",
        endpoint="https://api.trulayer.ai",
        batch_size=50,
        flush_interval=60.0,
    )


# ---------------------------------------------------------------------------
# Default (drop) mode
# ---------------------------------------------------------------------------


@respx.mock
async def test_drop_mode_warns_once_per_failure_window() -> None:
    """Default fail-mode: repeated failures produce exactly one warning until
    a successful send resets the latch.
    """
    respx.post("https://api.trulayer.ai/v1/ingest/batch").mock(return_value=httpx.Response(500))
    sender = _make_sender()
    assert sender._fail_mode_block is False

    with patch("trulayer.batch._RETRY_BASE_DELAY", 0.0), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        await sender._send_with_retry([{"id": "t1"}])
        await sender._send_with_retry([{"id": "t2"}])
        await sender._send_with_retry([{"id": "t3"}])
        retry_warnings = [x for x in w if "retries" in str(x.message)]
        assert len(retry_warnings) == 1


@respx.mock
async def test_drop_mode_re_warns_after_recovery() -> None:
    """After a successful send, the next failure should warn again."""
    route = respx.post("https://api.trulayer.ai/v1/ingest/batch").mock(
        side_effect=[
            httpx.Response(500),  # first batch fails
            httpx.Response(200),  # second batch succeeds
            httpx.Response(500),  # third batch fails — new window, must warn
        ]
    )
    sender = _make_sender()

    with (
        patch("trulayer.batch._RETRY_BASE_DELAY", 0.0),
        patch("trulayer.batch._MAX_RETRIES", 1),
        warnings.catch_warnings(record=True) as w,
    ):
        warnings.simplefilter("always")
        await sender._send_with_retry([{"id": "t1"}])
        await sender._send_with_retry([{"id": "t2"}])
        await sender._send_with_retry([{"id": "t3"}])
        retry_warnings = [x for x in w if "retries" in str(x.message)]
        assert len(retry_warnings) == 2
    assert route.call_count == 3


# ---------------------------------------------------------------------------
# Block mode
# ---------------------------------------------------------------------------


@respx.mock
async def test_block_mode_raises_flush_error() -> None:
    """TRULAYER_FAIL_MODE=block: flush failure raises TruLayerFlushError."""
    respx.post("https://api.trulayer.ai/v1/ingest/batch").mock(return_value=httpx.Response(500))
    with mock.patch.dict(os.environ, {"TRULAYER_FAIL_MODE": "block"}):
        sender = _make_sender()
        assert sender._fail_mode_block is True

        with patch("trulayer.batch._RETRY_BASE_DELAY", 0.0), pytest.raises(TruLayerFlushError):
            await sender._send_with_retry([{"id": "t1"}])


@respx.mock
async def test_block_mode_success_does_not_raise() -> None:
    """Block mode is only about failures — happy path is unchanged."""
    respx.post("https://api.trulayer.ai/v1/ingest/batch").mock(return_value=httpx.Response(200))
    with mock.patch.dict(os.environ, {"TRULAYER_FAIL_MODE": "block"}):
        sender = _make_sender()
        await sender._send_with_retry([{"id": "t1"}])  # no raise


@respx.mock
async def test_block_mode_preserves_underlying_cause() -> None:
    respx.post("https://api.trulayer.ai/v1/ingest/batch").mock(return_value=httpx.Response(500))
    with mock.patch.dict(os.environ, {"TRULAYER_FAIL_MODE": "block"}):
        sender = _make_sender()
        with (
            patch("trulayer.batch._RETRY_BASE_DELAY", 0.0),
            pytest.raises(TruLayerFlushError) as excinfo,
        ):
            await sender._send_with_retry([{"id": "t1"}])
        assert excinfo.value.__cause__ is not None


def test_fail_mode_env_is_case_insensitive() -> None:
    with mock.patch.dict(os.environ, {"TRULAYER_FAIL_MODE": "BLOCK"}):
        sender = _make_sender()
        assert sender._fail_mode_block is True
    with mock.patch.dict(os.environ, {"TRULAYER_FAIL_MODE": " block "}):
        sender = _make_sender()
        assert sender._fail_mode_block is True


def test_fail_mode_off_by_default() -> None:
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("TRULAYER_FAIL_MODE", None)
        sender = _make_sender()
        assert sender._fail_mode_block is False
