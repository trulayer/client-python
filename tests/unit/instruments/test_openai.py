import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import trulayer.instruments.openai as oai_module
from trulayer.instruments.openai import (
    _record_span,
    _wrap_async_stream,
    _wrap_sync_stream,
    instrument_openai,
    uninstrument_openai,
)
from trulayer.trace import TraceContext


def _make_client(project_id: str = "proj-1") -> MagicMock:
    client = MagicMock()
    client._project_id = project_id
    client._batch = MagicMock()
    client._scrub_fn = None
    client._sample_rate = 1.0
    client._metadata_validator = None
    return client


def _make_openai_response(content: str = "hello", model: str = "gpt-4o") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        model=model,
    )


def test_record_span_inside_trace() -> None:
    client = _make_client()
    with TraceContext(client, name="test") as _:
        result = _make_openai_response("world")
        _record_span(client, {"model": "gpt-4o", "messages": [{"content": "hi"}]}, result, 0.1)

    payload = client._batch.enqueue.call_args[0][0]
    span = payload["spans"][0]
    assert span["name"] == "openai.chat"
    assert span["type"] == "llm"
    assert span["input"] == "hi"
    assert span["output"] == "world"
    assert span["model"] == "gpt-4o"
    assert span["prompt_tokens"] == 10
    assert span["completion_tokens"] == 5


def test_record_span_malformed_result_still_records() -> None:
    """If result parsing raises, _record_span still records with empty output."""
    client = _make_client()

    class _BadResult:
        @property
        def choices(self) -> object:
            raise AttributeError("no choices")

    with TraceContext(client, name="test") as _:
        _record_span(
            client, {"model": "gpt-4o", "messages": [{"content": "hi"}]}, _BadResult(), 0.1
        )

    payload = client._batch.enqueue.call_args[0][0]
    assert payload["spans"][0]["output"] == ""


def test_record_span_outside_trace_is_noop() -> None:
    client = _make_client()
    result = _make_openai_response()
    # Should not raise, should not crash
    _record_span(client, {"model": "gpt-4o", "messages": []}, result, 0.1)
    client._batch.enqueue.assert_not_called()


def test_uninstrument_is_idempotent() -> None:
    uninstrument_openai()
    uninstrument_openai()  # second call should not raise


def test_instrument_openai_no_openai_installed() -> None:
    """When openai is not installed, instrument_openai warns and returns gracefully."""
    client = _make_client()
    with patch.dict(sys.modules, {"openai": None}):
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            oai_module._patched = False
            instrument_openai(client)
            assert any("openai" in str(warning.message).lower() for warning in w)


def test_instrument_openai_patches_and_unpatches() -> None:
    """instrument_openai wraps create; uninstrument_openai restores it."""
    client = _make_client()

    fake_response = _make_openai_response("patched response")
    original_sync = MagicMock(return_value=fake_response)
    original_async = MagicMock(return_value=fake_response)
    mock_completions = MagicMock()
    mock_completions.create = original_sync
    mock_async_completions = MagicMock()
    mock_async_completions.create = original_async

    mock_openai = MagicMock()
    mock_openai.resources.chat.completions.Completions = mock_completions
    mock_openai.resources.chat.completions.AsyncCompletions = mock_async_completions

    with patch.dict(sys.modules, {"openai": mock_openai}):
        oai_module._patched = False
        oai_module._original_create = None
        oai_module._original_acreate = None

        instrument_openai(client)
        assert oai_module._patched

        # Call the patched sync function to cover its body
        patched_fn = mock_completions.create
        patched_fn(None, messages=[{"content": "hi"}], model="gpt-4o")
        original_sync.assert_called_once()

        instrument_openai(client)  # idempotent
        uninstrument_openai()
        assert not oai_module._patched


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def _make_chunk(text: str | None = "hello", usage: object | None = None) -> SimpleNamespace:
    delta = SimpleNamespace(content=text)
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice], usage=usage)


def test_wrap_sync_stream_records_span() -> None:
    client = _make_client()
    chunks = [_make_chunk("hel"), _make_chunk("lo"), _make_chunk(None)]

    with TraceContext(client, name="stream-test") as _:
        gen = _wrap_sync_stream(
            client,
            {"model": "gpt-4o", "messages": [{"content": "ping"}]},
            iter(chunks),
            0.0,
        )
        collected = list(gen)

    assert len(collected) == 3
    payload = client._batch.enqueue.call_args[0][0]
    span = payload["spans"][0]
    assert span["name"] == "openai.chat"
    assert span["output"] == "hello"
    assert span["model"] == "gpt-4o"
    assert span["input"] == "ping"


def test_wrap_sync_stream_captures_usage_from_chunk() -> None:
    client = _make_client()
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    chunks = [_make_chunk("hi", usage=usage)]

    with TraceContext(client, name="t") as _:
        list(_wrap_sync_stream(client, {"model": "gpt-4o"}, iter(chunks), 0.0))

    payload = client._batch.enqueue.call_args[0][0]
    span = payload["spans"][0]
    assert span["prompt_tokens"] == 10
    assert span["completion_tokens"] == 5


def test_wrap_sync_stream_outside_trace_passthrough() -> None:
    client = _make_client()
    chunks = [_make_chunk("a"), _make_chunk("b")]
    collected = list(_wrap_sync_stream(client, {}, iter(chunks), 0.0))
    assert len(collected) == 2
    client._batch.enqueue.assert_not_called()


@pytest.mark.asyncio
async def test_wrap_async_stream_records_span() -> None:
    client = _make_client()

    async def _async_chunks() -> object:
        for c in [_make_chunk("async"), _make_chunk(" response")]:
            yield c

    with TraceContext(client, name="async-stream") as _:
        collected = [
            chunk
            async for chunk in _wrap_async_stream(
                client,
                {"model": "gpt-4o", "messages": [{"content": "question"}]},
                _async_chunks(),
                0.0,
            )
        ]

    assert len(collected) == 2
    payload = client._batch.enqueue.call_args[0][0]
    assert payload["spans"][0]["output"] == "async response"


@pytest.mark.asyncio
def test_wrap_sync_stream_exception_warns_and_records_partial() -> None:
    """Stream errors are caught, warned about, and partial output is recorded."""
    import warnings as _warnings

    client = _make_client()

    def _bad_stream() -> object:
        yield _make_chunk("partial")
        raise ValueError("stream cut")

    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        with TraceContext(client, name="err-stream") as _:
            list(_wrap_sync_stream(client, {"model": "gpt-4o"}, _bad_stream(), 0.0))

    assert any("stream cut" in str(warning.message) for warning in w)
    payload = client._batch.enqueue.call_args[0][0]
    assert payload["spans"][0]["output"] == "partial"


async def test_wrap_async_stream_outside_trace_passthrough() -> None:
    client = _make_client()

    async def _async_chunks() -> object:
        for c in [_make_chunk("x")]:
            yield c

    collected = [c async for c in _wrap_async_stream(client, {}, _async_chunks(), 0.0)]
    assert len(collected) == 1
    client._batch.enqueue.assert_not_called()


async def test_instrument_openai_async_patched_function() -> None:
    """The patched async create function delegates to the original."""
    client = _make_client()

    fake_response = _make_openai_response("async response")

    async def async_original(self: object, *args: object, **kwargs: object) -> object:
        return fake_response

    mock_completions = MagicMock()
    mock_completions.create = MagicMock()
    mock_async_completions = MagicMock()
    mock_async_completions.create = async_original

    mock_openai = MagicMock()
    mock_openai.resources.chat.completions.Completions = mock_completions
    mock_openai.resources.chat.completions.AsyncCompletions = mock_async_completions

    with patch.dict(sys.modules, {"openai": mock_openai}):
        oai_module._patched = False
        instrument_openai(client)

        patched_async = mock_async_completions.create
        result = await patched_async(None, messages=[{"content": "hi"}], model="gpt-4o")
        assert result is fake_response

        uninstrument_openai()
