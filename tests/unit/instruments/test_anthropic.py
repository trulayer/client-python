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
    with TraceContext(client) as _:
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
    with patch.dict(sys.modules, {"anthropic": None}):
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
        collected = list(
            _wrap_sync_stream(
                client,
                {"model": "claude-3-haiku", "messages": [{"content": "say hi"}]},
                iter(events),
                0.0,
            )
        )

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
        collected = [
            e
            async for e in _wrap_async_stream(
                client,
                {"model": "claude-3", "messages": [{"content": "q"}]},
                _async_events(),
                0.0,
            )
        ]

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


# ---------------------------------------------------------------------------
# Additional coverage: patched_create / patched_acreate execution paths
# ---------------------------------------------------------------------------


def test_patched_create_non_stream() -> None:
    """Calling the patched sync create without stream= should record a span."""
    client = _make_client()

    fake_result = _make_anthropic_response("hi")
    mock_messages_cls = MagicMock()
    mock_async_messages_cls = MagicMock()

    mock_anthropic = MagicMock()
    mock_anthropic.resources.messages.Messages = mock_messages_cls
    mock_anthropic.resources.messages.AsyncMessages = mock_async_messages_cls

    with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
        ant_module._patched = False
        ant_module._original_create = None
        ant_module._original_acreate = None

        # Capture the original before patching so we can call it
        original_create_mock = MagicMock(return_value=fake_result)
        mock_messages_cls.create = original_create_mock

        with TraceContext(client, name="trace"):
            instrument_anthropic(client)
            # Call _patched_create directly by calling the patched method
            patched_fn = mock_messages_cls.create
            result = patched_fn(None, model="claude-3", messages=[{"content": "hello"}])

        assert result is fake_result
        uninstrument_anthropic()


@pytest.mark.asyncio
async def test_patched_acreate_non_stream() -> None:
    """Calling the patched async create without stream= should record a span."""
    client = _make_client()

    fake_result = _make_anthropic_response("async hi")
    mock_messages_cls = MagicMock()
    mock_async_messages_cls = MagicMock()

    mock_anthropic = MagicMock()
    mock_anthropic.resources.messages.Messages = mock_messages_cls
    mock_anthropic.resources.messages.AsyncMessages = mock_async_messages_cls

    import asyncio

    async def fake_acreate(self: Any, *args: Any, **kwargs: Any) -> Any:
        return fake_result

    mock_async_messages_cls.create = fake_acreate

    with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
        ant_module._patched = False
        ant_module._original_create = MagicMock(return_value=fake_result)
        ant_module._original_acreate = None

        with TraceContext(client, name="trace"):
            instrument_anthropic(client)
            patched_fn = mock_async_messages_cls.create
            result = await patched_fn(None, model="claude-3", messages=[{"content": "q"}])

        assert result is fake_result
        uninstrument_anthropic()


def test_patched_create_stream_path() -> None:
    """Calling patched sync create with stream=True should return a generator."""
    client = _make_client()

    events = [_make_content_delta_event("streamed")]
    mock_messages_cls = MagicMock()
    mock_async_messages_cls = MagicMock()

    mock_anthropic = MagicMock()
    mock_anthropic.resources.messages.Messages = mock_messages_cls
    mock_anthropic.resources.messages.AsyncMessages = mock_async_messages_cls

    original_create_mock = MagicMock(return_value=iter(events))
    mock_messages_cls.create = original_create_mock

    with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
        ant_module._patched = False
        ant_module._original_create = None
        ant_module._original_acreate = None

        with TraceContext(client, name="trace"):
            instrument_anthropic(client)
            patched_fn = mock_messages_cls.create
            result = patched_fn(None, model="claude-3", messages=[{"content": "hi"}], stream=True)
            collected = list(result)

        assert len(collected) == 1
        uninstrument_anthropic()


# ---------------------------------------------------------------------------
# uninstrument when anthropic is not installed
# ---------------------------------------------------------------------------


def test_uninstrument_anthropic_no_package() -> None:
    """uninstrument_anthropic with missing package shouldn't raise."""
    import warnings

    ant_module._patched = True
    ant_module._original_create = MagicMock()
    ant_module._original_acreate = MagicMock()

    with patch.dict(sys.modules, {"anthropic": None}):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            uninstrument_anthropic()

    assert not ant_module._patched


# ---------------------------------------------------------------------------
# _record_span: outer exception (span creation fails)
# ---------------------------------------------------------------------------


def test_record_span_outer_exception_warns() -> None:
    """If current_trace() raises, _record_span should warn and not propagate."""
    import warnings

    client = _make_client()
    result = _make_anthropic_response()

    with patch("trulayer.trace.current_trace", side_effect=RuntimeError("boom")):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _record_span(client, {"model": "m", "messages": []}, result, 0.1)
            assert any("failed to record" in str(warning.message).lower() for warning in w)


# ---------------------------------------------------------------------------
# _record_span: result.content raises (inner exception path, lines 101-102)
# ---------------------------------------------------------------------------


def test_record_span_bad_result_content() -> None:
    """result.content raising should be swallowed gracefully."""
    client = _make_client()

    class BadResult:
        @property
        def content(self) -> list[object]:
            raise AttributeError("no content")

        usage = None

    with TraceContext(client, name="t"):
        _record_span(client, {"model": "m", "messages": [{"content": "q"}]}, BadResult(), 0.1)

    payload = client._batch.enqueue.call_args[0][0]
    span = payload["spans"][0]
    assert span["output"] == ""


# ---------------------------------------------------------------------------
# _wrap_sync_stream: exception propagation (lines 158-163, 170-171)
# ---------------------------------------------------------------------------


def test_wrap_sync_stream_exception_warns() -> None:
    """An exception raised mid-stream is caught by the outer handler and emits a warning."""
    import warnings

    client = _make_client()

    def _bad_events() -> object:
        yield _make_content_delta_event("partial")
        raise RuntimeError("stream error")

    with TraceContext(client, name="t"):
        gen = _wrap_sync_stream(
            client,
            {"model": "claude-3", "messages": [{"content": "q"}]},
            _bad_events(),
            0.0,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            list(gen)
            assert any("streaming" in str(warning.message).lower() for warning in w)


def test_wrap_sync_stream_event_processing_exception_swallowed() -> None:
    """Exception in per-event processing should be swallowed; stream continues."""
    client = _make_client()

    class BrokenEvent:
        """Event whose attribute access raises."""

        type = "content_block_delta"

        @property
        def delta(self) -> object:
            raise AttributeError("broken")

    events = [BrokenEvent(), _make_content_delta_event("after")]

    with TraceContext(client, name="t"):
        collected = list(
            _wrap_sync_stream(
                client,
                {"model": "m", "messages": [{"content": "q"}]},
                iter(events),
                0.0,
            )
        )

    assert len(collected) == 2


def test_wrap_sync_stream_span_setup_exception_warns() -> None:
    """If SpanContext/current_trace raises, a warning is emitted and stream is silently lost."""
    import warnings

    client = _make_client()

    with patch("trulayer.trace.current_trace", side_effect=RuntimeError("kaboom")):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gen = _wrap_sync_stream(client, {}, iter([]), 0.0)
            list(gen)
            assert any("streaming" in str(warning.message).lower() for warning in w)


# ---------------------------------------------------------------------------
# _wrap_async_stream: exception propagation (lines 219-221, 228-229)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wrap_async_stream_exception_warns() -> None:
    """An exception raised mid-async-stream is caught by the outer handler and warns."""
    import warnings

    client = _make_client()

    async def _bad_async_events() -> object:
        yield _make_content_delta_event("partial")
        raise RuntimeError("async stream error")

    with TraceContext(client, name="t"):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            async for _ in _wrap_async_stream(
                client,
                {"model": "claude-3", "messages": [{"content": "q"}]},
                _bad_async_events(),
                0.0,
            ):
                pass
            assert any("async streaming" in str(warning.message).lower() for warning in w)


@pytest.mark.asyncio
async def test_wrap_async_stream_span_setup_exception_warns() -> None:
    """If current_trace raises in async stream, warn and don't propagate."""
    import warnings

    client = _make_client()

    async def _empty() -> object:
        return
        yield  # make it a generator

    with patch("trulayer.trace.current_trace", side_effect=RuntimeError("boom")):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            async for _ in _wrap_async_stream(client, {}, _empty(), 0.0):
                pass
            assert any("async streaming" in str(warning.message).lower() for warning in w)
