import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import trulayer.instruments.anthropic as ant_module
from trulayer.instruments.anthropic import (
    _record_span,
    _wrap_async_stream,
    _wrap_sync_stream,
    instrument_anthropic,
    uninstrument_anthropic,
)
from trulayer.trace import TraceContext


def _make_client() -> MagicMock:
    client = MagicMock()
    client._project_id = "proj-1"
    client._batch = MagicMock()
    client._scrub_fn = None
    client._sample_rate = 1.0
    client._metadata_validator = None
    return client


def _make_anthropic_response(text: str = "hello") -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=8, output_tokens=4),
    )


def test_record_span_inside_trace() -> None:
    client = _make_client()
    with TraceContext(client) as t:
        result = _make_anthropic_response("response text")
        _record_span(
            client,
            {"model": "claude-haiku-4-5-20251001", "messages": [{"content": "prompt"}]},
            result,
            0.2,
        )

    payload = client._batch.enqueue.call_args[0][0]
    span = payload["spans"][0]
    assert span["name"] == "anthropic.messages"
    assert span["span_type"] == "llm"
    assert span["input"] == "prompt"
    assert span["output"] == "response text"
    assert span["model"] == "claude-haiku-4-5-20251001"
    assert span["prompt_tokens"] == 8
    assert span["completion_tokens"] == 4


def test_record_span_outside_trace_is_noop() -> None:
    client = _make_client()
    result = _make_anthropic_response()
    _record_span(client, {"model": "claude-3", "messages": []}, result, 0.1)
    client._batch.enqueue.assert_not_called()


def test_uninstrument_is_idempotent() -> None:
    uninstrument_anthropic()
    uninstrument_anthropic()


def test_instrument_anthropic_no_package_installed() -> None:
    client = _make_client()
    with patch.dict(sys.modules, {"anthropic": None}):  # type: ignore[dict-item]
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ant_module._patched = False
            instrument_anthropic(client)
            assert any("anthropic" in str(warning.message).lower() for warning in w)


def test_instrument_anthropic_patches_and_unpatches() -> None:
    client = _make_client()

    mock_messages = MagicMock()
    mock_messages.create = MagicMock()
    mock_async_messages = MagicMock()
    mock_async_messages.create = MagicMock()

    mock_anthropic = MagicMock()
    mock_anthropic.resources.messages.Messages = mock_messages
    mock_anthropic.resources.messages.AsyncMessages = mock_async_messages

    with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
        ant_module._patched = False
        ant_module._original_create = None
        ant_module._original_acreate = None

        instrument_anthropic(client)
        assert ant_module._patched
        instrument_anthropic(client)  # idempotent
        assert ant_module._patched

        uninstrument_anthropic()
        assert not ant_module._patched


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

def _make_content_delta_event(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(text=text))


def _make_message_delta_event(input_tokens: int = 5, output_tokens: int = 3) -> SimpleNamespace:
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(type="message_delta", usage=usage)


def test_wrap_sync_stream_records_span() -> None:
    client = _make_client()
    events = [
        _make_content_delta_event("Hello"),
        _make_content_delta_event(" world"),
        _make_message_delta_event(input_tokens=8, output_tokens=4),
    ]

    with TraceContext(client, name="ant-stream") as _:
        collected = list(_wrap_sync_stream(
            client,
            {"model": "claude-3-haiku", "messages": [{"content": "say hi"}]},
            iter(events),
            0.0,
        ))

    assert len(collected) == 3
    payload = client._batch.enqueue.call_args[0][0]
    span = payload["spans"][0]
    assert span["name"] == "anthropic.messages"
    assert span["output"] == "Hello world"
    assert span["model"] == "claude-3-haiku"
    assert span["input"] == "say hi"
    assert span["prompt_tokens"] == 8
    assert span["completion_tokens"] == 4


def test_wrap_sync_stream_outside_trace_passthrough() -> None:
    client = _make_client()
    events = [_make_content_delta_event("hi")]
    collected = list(_wrap_sync_stream(client, {}, iter(events), 0.0))
    assert len(collected) == 1
    client._batch.enqueue.assert_not_called()


@pytest.mark.asyncio
async def test_wrap_async_stream_records_span() -> None:
    client = _make_client()

    async def _async_events() -> object:
        for e in [_make_content_delta_event("async"), _make_content_delta_event(" msg")]:
            yield e

    with TraceContext(client, name="ant-async") as _:
        collected = [e async for e in _wrap_async_stream(
            client,
            {"model": "claude-3", "messages": [{"content": "q"}]},
            _async_events(),
            0.0,
        )]

    assert len(collected) == 2
    payload = client._batch.enqueue.call_args[0][0]
    assert payload["spans"][0]["output"] == "async msg"


@pytest.mark.asyncio
async def test_wrap_async_stream_outside_trace_passthrough() -> None:
    client = _make_client()

    async def _async_events() -> object:
        yield _make_content_delta_event("x")

    collected = [e async for e in _wrap_async_stream(client, {}, _async_events(), 0.0)]
    assert len(collected) == 1
    client._batch.enqueue.assert_not_called()
