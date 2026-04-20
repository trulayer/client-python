from unittest.mock import patch

import httpx
import respx

from trulayer.batch import BatchSender


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
        respx.post("https://api.trulayer.ai/v1/ingest/batch").mock(
            return_value=httpx.Response(500)
        )
        with patch("trulayer.batch._RETRY_BASE_DELAY", 0.0):
            import warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                await sender._send_with_retry([{"id": "trace-1"}])
                assert len(w) == 1
                assert "retries" in str(w[0].message)
