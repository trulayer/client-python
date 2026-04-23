from unittest.mock import patch

import httpx
import pytest
import respx

from trulayer.batch import BatchSender
from trulayer.errors import InvalidAPIKeyError


def _make_sender(**kwargs: object) -> BatchSender:
    return BatchSender(
        api_key="tl_test",
        endpoint="https://api.trulayer.ai",
        batch_size=kwargs.get("batch_size", 50),  # type: ignore[arg-type]
        flush_interval=kwargs.get("flush_interval", 60.0),  # type: ignore[arg-type]
    )


@respx.mock
async def test_flush_sends_batch() -> None:
    route = respx.post("https://api.trulayer.ai/v1/ingest/batch").mock(
        return_value=httpx.Response(200, json={"ingested": 1})
    )
    sender = _make_sender()
    sender.enqueue({"id": "trace-1", "project_id": "proj-1"})
    await sender._flush()
    assert route.called


@respx.mock
async def test_flush_empty_queue_is_noop() -> None:
    route = respx.post("https://api.trulayer.ai/v1/ingest/batch").mock(
        return_value=httpx.Response(200)
    )
    sender = _make_sender()
    await sender._flush()
    assert not route.called


@respx.mock
async def test_retry_on_server_error() -> None:
    route = respx.post("https://api.trulayer.ai/v1/ingest/batch").mock(
        side_effect=[
            httpx.Response(500),
            httpx.Response(500),
            httpx.Response(200, json={"ingested": 1}),
        ]
    )
    sender = _make_sender()
    sender.enqueue({"id": "trace-1"})

    with patch("trulayer.batch._RETRY_BASE_DELAY", 0.0):
        await sender._send_with_retry([{"id": "trace-1"}])

    assert route.call_count == 3


async def test_max_retries_drops_and_warns() -> None:
    sender = _make_sender()
    with respx.mock:
        respx.post("https://api.trulayer.ai/v1/ingest/batch").mock(return_value=httpx.Response(500))
        with patch("trulayer.batch._RETRY_BASE_DELAY", 0.0):
            import warnings

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                await sender._send_with_retry([{"id": "trace-1"}])
                assert len(w) == 1
                assert "retries" in str(w[0].message)


@pytest.mark.parametrize("code", ["invalid_api_key", "api_key_expired"])
@respx.mock
async def test_401_invalid_or_expired_halts_without_retry(code: str) -> None:
    route = respx.post("https://api.trulayer.ai/v1/ingest/batch").mock(
        return_value=httpx.Response(401, json={"error": code})
    )
    sender = _make_sender()
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        await sender._send_with_retry([{"id": "trace-1"}])

    assert route.call_count == 1  # no retry
    assert sender.fatal_error is not None
    assert isinstance(sender.fatal_error, InvalidAPIKeyError)
    assert sender.fatal_error.code == code
    assert "invalid or has expired" in str(sender.fatal_error)
    assert any("halting trace submission" in str(msg.message) for msg in w)


@respx.mock
async def test_401_accepts_code_field_as_well_as_error() -> None:
    respx.post("https://api.trulayer.ai/v1/ingest/batch").mock(
        return_value=httpx.Response(401, json={"code": "invalid_api_key"})
    )
    sender = _make_sender()
    await sender._send_with_retry([{"id": "trace-1"}])
    assert isinstance(sender.fatal_error, InvalidAPIKeyError)
    assert sender.fatal_error.code == "invalid_api_key"


@respx.mock
async def test_401_unrelated_error_still_retries() -> None:
    route = respx.post("https://api.trulayer.ai/v1/ingest/batch").mock(
        return_value=httpx.Response(401, json={"error": "unauthorized"})
    )
    sender = _make_sender()
    with patch("trulayer.batch._RETRY_BASE_DELAY", 0.0):
        await sender._send_with_retry([{"id": "trace-1"}])
    assert route.call_count == _MAX_RETRIES_EXPECTED
    assert sender.fatal_error is None


_MAX_RETRIES_EXPECTED = 3


async def test_enqueue_drops_after_fatal_error() -> None:
    sender = _make_sender(batch_size=1)
    # Simulate latched fatal state without going through the network.
    sender._fatal_error = InvalidAPIKeyError("invalid_api_key")

    sender.enqueue({"id": "trace-1"})
    sender.enqueue({"id": "trace-2"})
    assert sender._queue.qsize() == 0


@respx.mock
async def test_flush_is_noop_after_fatal_error() -> None:
    route = respx.post("https://api.trulayer.ai/v1/ingest/batch").mock(
        return_value=httpx.Response(200)
    )
    sender = _make_sender()
    sender._fatal_error = InvalidAPIKeyError("api_key_expired")
    # Stuff an item in directly — enqueue() would now refuse.
    sender._queue.put_nowait({"id": "trace-x"})
    await sender._flush()
    assert not route.called
    assert sender._queue.qsize() == 0


def test_invalid_api_key_error_shape() -> None:
    err = InvalidAPIKeyError("invalid_api_key")
    assert err.code == "invalid_api_key"
    assert str(err) == "API key is invalid or has expired — check your configuration."
    assert isinstance(err, Exception)
