import time

import httpx
import respx

from trulayer.batch import BatchSender


@respx.mock
def test_batch_sender_start_and_shutdown() -> None:
    respx.post("https://api.trulayer.ai/v1/ingest/batch").mock(
        return_value=httpx.Response(200, json={"ingested": 1})
    )
    sender = BatchSender(
        api_key="tl_test",
        endpoint="https://api.trulayer.ai",
        flush_interval=60.0,  # long interval — we flush manually
    )
    sender.start()
    assert sender._thread is not None
    assert sender._thread.is_alive()

    sender.enqueue({"id": "trace-1"})
    sender.shutdown(timeout=3.0)

    # After shutdown loop is stopped; thread should exit
    sender._thread.join(timeout=2.0)


@respx.mock
def test_enqueue_flushes_at_batch_size() -> None:
    route = respx.post("https://api.trulayer.ai/v1/ingest/batch").mock(
        return_value=httpx.Response(200)
    )
    sender = BatchSender(
        api_key="tl_test",
        endpoint="https://api.trulayer.ai",
        batch_size=2,
        flush_interval=60.0,
    )
    sender.start()

    sender.enqueue({"id": "a"})
    sender.enqueue({"id": "b"})  # triggers flush

    # Give the async flush a moment to fire
    time.sleep(0.2)
    sender.shutdown(timeout=2.0)

    assert route.called
