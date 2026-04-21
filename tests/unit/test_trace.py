from unittest.mock import MagicMock, patch

import pytest

from trulayer.trace import TraceContext, _scrub_payload, _validate_metadata, current_trace


def _make_client(project_id: str = "proj-1", sample_rate: float = 1.0) -> MagicMock:
    client = MagicMock()
    client._project_id = project_id
    client._batch = MagicMock()
    client._sample_rate = sample_rate
    client._scrub_fn = None
    client._metadata_validator = None
    return client


def test_trace_context_sets_current() -> None:
    client = _make_client()
    with TraceContext(client, name="test") as t:
        assert current_trace() is t
    assert current_trace() is None


def test_trace_enqueues_on_exit() -> None:
    client = _make_client()
    with TraceContext(client, name="my-trace") as t:
        t.set_input("hello")
        t.set_output("world")
    client._batch.enqueue.assert_called_once()
    payload = client._batch.enqueue.call_args[0][0]
    assert payload["name"] == "my-trace"
    assert payload["input"] == "hello"
    assert payload["output"] == "world"


def test_trace_records_exception() -> None:
    client = _make_client()
    with pytest.raises(ValueError), TraceContext(client):
        raise ValueError("boom")
    payload = client._batch.enqueue.call_args[0][0]
    assert payload["error"] is True


def test_span_captures_latency_and_output() -> None:
    client = _make_client()
    with TraceContext(client) as t, t.span("llm-call", span_type="llm") as span:
        span.set_input("prompt")
        span.set_output("response")
        span.set_model("gpt-4o")

    payload = client._batch.enqueue.call_args[0][0]
    assert len(payload["spans"]) == 1
    s = payload["spans"][0]
    assert s["name"] == "llm-call"
    assert s["input"] == "prompt"
    assert s["output"] == "response"
    assert s["model"] == "gpt-4o"
    assert s["latency_ms"] >= 0


def test_span_records_exception() -> None:
    client = _make_client()
    with pytest.raises(RuntimeError), TraceContext(client) as t, t.span("bad-span"):
        raise RuntimeError("span error")

    payload = client._batch.enqueue.call_args[0][0]
    assert payload["spans"][0]["error"] is True
    assert "RuntimeError" in payload["spans"][0]["error_message"]


def test_nested_spans() -> None:
    client = _make_client()
    with TraceContext(client) as t:
        with t.span("span-1"):
            pass
        with t.span("span-2"):
            pass

    payload = client._batch.enqueue.call_args[0][0]
    assert len(payload["spans"]) == 2
    assert payload["spans"][0]["name"] == "span-1"
    assert payload["spans"][1]["name"] == "span-2"


async def test_async_trace_context() -> None:
    client = _make_client()
    async with TraceContext(client, name="async-trace") as t:
        t.set_output("async output")
    client._batch.enqueue.assert_called_once()
    payload = client._batch.enqueue.call_args[0][0]
    assert payload["name"] == "async-trace"


async def test_async_span_context() -> None:
    client = _make_client()
    async with (
        TraceContext(client, name="async-span-trace") as t,
        t.span("async-span", span_type="llm") as span,
    ):
        span.set_input("in")
        span.set_output("out")
        span.set_metadata(key="val")
    payload = client._batch.enqueue.call_args[0][0]
    assert payload["spans"][0]["name"] == "async-span"


def test_trace_set_metadata_and_add_tag() -> None:
    client = _make_client()
    with TraceContext(client, name="t") as t:
        t.set_metadata(env="test", version=2)
        t.add_tag("smoke")
    payload = client._batch.enqueue.call_args[0][0]
    assert payload["metadata"]["env"] == "test"
    assert payload["metadata"]["version"] == 2
    assert "smoke" in payload["tags"]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def test_sample_rate_1_always_enqueues() -> None:
    client = _make_client(sample_rate=1.0)
    with TraceContext(client, name="t"):
        pass
    client._batch.enqueue.assert_called_once()


def test_sample_rate_0_never_enqueues() -> None:
    client = _make_client(sample_rate=0.0)
    with TraceContext(client, name="t"):
        pass
    client._batch.enqueue.assert_not_called()


def test_sample_rate_respected_via_random() -> None:
    client = _make_client(sample_rate=0.5)

    with patch("trulayer.trace.random.random", return_value=0.3), TraceContext(client, name="t"):
        pass
    client._batch.enqueue.assert_called_once()

    client._batch.enqueue.reset_mock()

    with patch("trulayer.trace.random.random", return_value=0.7), TraceContext(client, name="t"):
        pass
    client._batch.enqueue.assert_not_called()


def test_enqueue_error_is_swallowed() -> None:
    """If batch.enqueue raises, the trace context does not propagate the error."""
    client = _make_client()
    client._batch.enqueue.side_effect = RuntimeError("queue full")
    with TraceContext(client, name="t"):
        pass


# ---------------------------------------------------------------------------
# PII scrubbing
# ---------------------------------------------------------------------------

def test_scrub_payload_scrubs_trace_fields() -> None:
    payload: dict[str, object] = {
        "input": "my email is foo@bar.com",
        "output": "secret",
        "error_message": None,
        "spans": [],
    }
    result = _scrub_payload(payload, lambda s: s.replace("foo@bar.com", "[REDACTED]"))
    assert result["input"] == "my email is [REDACTED]"
    assert result["output"] == "secret"
    assert result["error_message"] is None


def test_scrub_payload_scrubs_span_fields() -> None:
    payload = {
        "input": "clean",
        "spans": [
            {"input": "user: foo@bar.com", "output": "assistant reply", "error_message": None},
        ],
    }
    result = _scrub_payload(payload, lambda s: "[SCRUBBED]" if "foo@bar.com" in s else s)
    assert result["spans"][0]["input"] == "[SCRUBBED]"
    assert result["spans"][0]["output"] == "assistant reply"


def test_scrub_fn_applied_on_enqueue() -> None:
    client = _make_client()
    client._scrub_fn = lambda s: s.replace("secret", "***")
    with TraceContext(client, name="t") as t:
        t.set_input("contains secret info")
        t.set_output("also secret")
        with t.span("s") as span:
            span.set_input("span secret")
            span.set_output("ok")
    payload = client._batch.enqueue.call_args[0][0]
    assert payload["input"] == "contains *** info"
    assert payload["output"] == "also ***"
    assert payload["spans"][0]["input"] == "span ***"
    assert payload["spans"][0]["output"] == "ok"


def test_scrub_fn_none_is_noop() -> None:
    client = _make_client()
    client._scrub_fn = None
    with TraceContext(client, name="t") as t:
        t.set_input("raw input")
    payload = client._batch.enqueue.call_args[0][0]
    assert payload["input"] == "raw input"


def test_scrub_fn_exception_does_not_propagate() -> None:
    def _bad_scrub(s: str) -> str:
        raise RuntimeError("scrub failed")

    client = _make_client()
    client._scrub_fn = _bad_scrub
    with TraceContext(client, name="t") as t:
        t.set_input("data")


# ---------------------------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------------------------

def test_validate_metadata_passes_valid() -> None:
    def _require_source(m: dict[str, object]) -> None:
        if "source" not in m:
            raise ValueError("missing 'source' key")

    payload = {"metadata": {"source": "api"}, "spans": []}
    result = _validate_metadata(payload, _require_source)
    assert result["metadata"] == {"source": "api"}


def test_validate_metadata_strips_invalid_and_warns() -> None:
    import warnings as _w

    def _strict(m: dict[str, object]) -> None:
        raise ValueError("invalid")

    payload = {"metadata": {"secret": "value"}, "spans": []}
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        result = _validate_metadata(payload, _strict)

    assert result["metadata"] == {}
    assert any("metadata validation" in str(w.message).lower() for w in caught)


def test_validate_metadata_strips_invalid_span_metadata() -> None:
    import warnings as _w

    def _no_pii(m: dict[str, object]) -> None:
        if "email" in m:
            raise ValueError("PII detected")

    payload = {
        "metadata": {},
        "spans": [{"metadata": {"email": "x@y.com"}}, {"metadata": {"ok": True}}],
    }
    with _w.catch_warnings(record=True):
        result = _validate_metadata(payload, _no_pii)

    assert result["spans"][0]["metadata"] == {}
    assert result["spans"][1]["metadata"] == {"ok": True}


def test_metadata_validator_applied_on_enqueue() -> None:
    def _require_env(m: dict[str, object]) -> None:
        if "env" not in m:
            raise ValueError("missing env")

    client = _make_client()
    client._metadata_validator = _require_env

    with MagicMock():
        pass

    import warnings as _w

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        with TraceContext(client, name="t", metadata={"bad": "data"}) as t, t.span("s") as span:
            span.set_metadata(env="prod")

    payload = client._batch.enqueue.call_args[0][0]
    assert payload["metadata"] == {}
    assert payload["spans"][0]["metadata"] == {"env": "prod"}
    assert any("metadata validation" in str(w.message).lower() for w in caught)


def test_metadata_validator_none_is_noop() -> None:
    client = _make_client()
    client._metadata_validator = None
    with TraceContext(client, name="t", metadata={"key": "val"}) as _:
        pass
    payload = client._batch.enqueue.call_args[0][0]
    assert payload["metadata"] == {"key": "val"}


def test_sample_rate_missing_on_client_defaults_to_always() -> None:
    """Client without _sample_rate attribute defaults to always sampling."""
    class _MinimalClient:
        _project_id = "proj-1"
        _batch: MagicMock = MagicMock()
        _scrub_fn = None
        _metadata_validator = None

    client = _MinimalClient()
    with TraceContext(client, name="t"):  # type: ignore[arg-type]
        pass
    client._batch.enqueue.assert_called_once()


def test_external_id_propagates_to_payload() -> None:
    client = _make_client()
    with TraceContext(client, name="t", external_id="ext-99"):
        pass
    payload = client._batch.enqueue.call_args[0][0]
    assert payload["external_id"] == "ext-99"


def test_set_model_and_set_cost_populate_payload() -> None:
    client = _make_client()
    with TraceContext(client, name="t") as ctx:
        ctx.set_model("gpt-4o-mini")
        ctx.set_cost(0.0123)
    payload = client._batch.enqueue.call_args[0][0]
    assert payload["model"] == "gpt-4o-mini"
    assert payload["cost"] == pytest.approx(0.0123)


def test_latency_ms_is_auto_derived_on_exit() -> None:
    client = _make_client()
    with TraceContext(client, name="t"):
        pass
    payload = client._batch.enqueue.call_args[0][0]
    assert payload["latency_ms"] is not None
    assert payload["latency_ms"] >= 0
