"""Wire-shape contract tests.

These tests verify the exact JSON payload the SDK produces before it reaches
the network layer. They use LocalBatchSender (via ``create_test_client``) or
mock senders so no real HTTP request is made.

The assertions here are the contract between the SDK and the TruLayer
ingestion API. Any change that alters the wire format must update these
tests in the same change.
"""

from __future__ import annotations

import json
import warnings
from unittest.mock import patch

import httpx
import pytest
import respx

from trulayer.client import TruLayerClient
from trulayer.model import SpanData, TraceData
from trulayer.testing import create_test_client

# ---------------------------------------------------------------------------
# Trace payload shape
# ---------------------------------------------------------------------------


class TestTraceWireShape:
    def test_trace_enqueues_with_required_fields(self) -> None:
        client, sender = create_test_client(project_name="proj-wire")
        with client.trace("my-op") as t:
            t.set_input("user prompt")
            t.set_output("model answer")
            t.set_model("gpt-4o-mini")
            t.set_cost(0.001)

        traces = sender.traces
        assert len(traces) == 1
        p = traces[0]
        assert isinstance(p["id"], str)
        assert p["project_id"] == "proj-wire"
        assert p["name"] == "my-op"
        assert p["input"] == "user prompt"
        assert p["output"] == "model answer"
        assert p["model"] == "gpt-4o-mini"
        assert p["cost"] == 0.001
        assert isinstance(p["tags"], list)
        assert isinstance(p["metadata"], dict)
        assert isinstance(p["started_at"], str)

    def test_trace_error_field_is_null_on_success(self) -> None:
        """Spec: ``TraceRequest.error`` is ``string | null``; null when no error."""
        client, sender = create_test_client(project_name="proj-wire")
        with client.trace("happy"):
            pass

        p = sender.traces[0]
        assert p["error"] is None

    def test_trace_error_field_is_string_on_exception(self) -> None:
        """Spec: ``TraceRequest.error`` carries the error message string on failure."""
        client, sender = create_test_client(project_name="proj-wire")
        with pytest.raises(RuntimeError), client.trace("err-trace"):
            raise RuntimeError("something broke")

        p = sender.traces[0]
        assert isinstance(p["error"], str)
        assert "something broke" in p["error"]
        # The legacy separate boolean+message pair must not appear on the wire.
        assert "error_message" not in p

    def test_session_id_and_external_id_populated(self) -> None:
        client, sender = create_test_client(project_name="proj-wire")
        with client.trace("scoped", session_id="sess-abc", external_id="ext-xyz"):
            pass

        p = sender.traces[0]
        assert p["session_id"] == "sess-abc"
        assert p["external_id"] == "ext-xyz"

    def test_session_id_and_external_id_null_when_absent(self) -> None:
        client, sender = create_test_client(project_name="proj-wire")
        with client.trace("bare"):
            pass

        p = sender.traces[0]
        assert p["session_id"] is None
        assert p["external_id"] is None

    def test_tags_propagated_to_payload(self) -> None:
        client, sender = create_test_client(project_name="proj-wire")
        with client.trace("tagged", tags=["prod", "v2"]):
            pass

        assert sender.traces[0]["tags"] == ["prod", "v2"]

    def test_metadata_propagated_to_payload(self) -> None:
        client, sender = create_test_client(project_name="proj-wire")
        with client.trace("meta", metadata={"env": "staging"}):
            pass

        assert sender.traces[0]["metadata"]["env"] == "staging"

    def test_latency_ms_is_non_negative_integer(self) -> None:
        client, sender = create_test_client(project_name="proj-wire")
        with client.trace("latency"):
            pass

        lms = sender.traces[0]["latency_ms"]
        assert lms is not None
        assert isinstance(lms, int)
        assert lms >= 0


# ---------------------------------------------------------------------------
# Span payload shape
# ---------------------------------------------------------------------------


class TestSpanWireShape:
    def test_span_included_in_trace_spans_array(self) -> None:
        client, sender = create_test_client(project_name="proj-wire")
        with client.trace("with-span") as t, t.span("llm-call", span_type="llm") as s:
            s.set_input("prompt")
            s.set_output("answer")
            s.set_model("gpt-4o")
            s.set_tokens(prompt=10, completion=5)

        trace_payload = sender.traces[0]
        spans = trace_payload["spans"]
        assert len(spans) == 1
        sp = spans[0]
        assert sp["name"] == "llm-call"
        assert sp["input"] == "prompt"
        assert sp["output"] == "answer"
        assert sp["model"] == "gpt-4o"
        assert sp["prompt_tokens"] == 10
        assert sp["completion_tokens"] == 5
        assert sp["trace_id"] == trace_payload["id"]
        assert isinstance(sp["latency_ms"], int)
        assert sp["latency_ms"] >= 0

    def test_span_error_is_string_on_exception(self) -> None:
        """Spec: span ``error`` carries the formatted traceback / message."""
        client, sender = create_test_client(project_name="proj-wire")
        with (
            pytest.raises(ValueError),
            client.trace("span-err") as t,
            t.span("bad-span"),
        ):
            raise ValueError("span failed")

        spans = sender.traces[0]["spans"]
        assert len(spans) == 1
        sp = spans[0]
        assert isinstance(sp["error"], str)
        assert "span failed" in sp["error"]
        assert "error_message" not in sp

    def test_span_error_is_null_on_success(self) -> None:
        """Spec: span ``error`` is ``null`` when the span succeeds."""
        client, sender = create_test_client(project_name="proj-wire")
        with client.trace("ok") as t, t.span("s"):
            pass

        sp = sender.traces[0]["spans"][0]
        assert sp["error"] is None

    def test_span_field_is_type_not_span_type(self) -> None:
        """Spec: ``SpanRequest.type`` — the SDK serializes the Python attribute
        ``span_type`` as ``type`` on the wire.
        """
        client, sender = create_test_client(project_name="proj-wire")
        with client.trace("field-name") as t, t.span("s", span_type="llm"):
            pass

        sp = sender.traces[0]["spans"][0]
        assert sp["type"] == "llm"
        assert "span_type" not in sp

    def test_span_timestamps_use_start_time_end_time(self) -> None:
        """Spec: ``SpanRequest.start_time`` / ``end_time`` — the SDK serializes
        ``started_at`` → ``start_time`` and ``ended_at`` → ``end_time``.
        """
        client, sender = create_test_client(project_name="proj-wire")
        with client.trace("ts-fields") as t, t.span("s"):
            pass

        sp = sender.traces[0]["spans"][0]
        assert "start_time" in sp
        assert "end_time" in sp
        assert isinstance(sp["start_time"], str)
        assert "started_at" not in sp
        assert "ended_at" not in sp


# ---------------------------------------------------------------------------
# Model-level serialization (unit coverage for to_wire)
# ---------------------------------------------------------------------------


class TestModelToWire:
    def test_span_to_wire_field_names(self) -> None:
        span = SpanData(
            trace_id="trace-1",
            name="op",
            span_type="llm",
            error="boom",
        )
        wire = span.to_wire()
        assert wire["type"] == "llm"
        assert wire["start_time"]
        assert "end_time" in wire
        assert wire["error"] == "boom"
        assert "span_type" not in wire
        assert "error_message" not in wire
        assert "started_at" not in wire
        assert "ended_at" not in wire

    def test_span_to_wire_error_null_when_no_error(self) -> None:
        span = SpanData(trace_id="trace-1", name="op", span_type="llm")
        wire = span.to_wire()
        assert wire["error"] is None

    def test_trace_to_wire_error_string(self) -> None:
        trace = TraceData(
            project_id="p",
            error="fatal",
            spans=[SpanData(name="s", span_type="tool")],
        )
        wire = trace.to_wire()
        assert wire["error"] == "fatal"
        assert "error_message" not in wire
        # Nested spans use the wire-shape too
        assert wire["spans"][0]["type"] == "tool"
        assert "span_type" not in wire["spans"][0]


# ---------------------------------------------------------------------------
# Batch sender wire format — verified via mock of _send_with_retry
# ---------------------------------------------------------------------------


class TestBatchSenderWireFormat:
    """Verify the exact JSON body the BatchSender would POST."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_sends_traces_array_as_json_body(self) -> None:
        route = respx.post("https://api.trulayer.ai/v1/ingest/batch").mock(
            return_value=httpx.Response(201, json={"ingested": 1, "ids": []})
        )

        from trulayer.batch import BatchSender

        sender = BatchSender(
            api_key="tl_test",
            endpoint="https://api.trulayer.ai",
        )
        trace = TraceData(project_id="proj-1")
        await sender._send_with_retry([trace.to_wire()])

        assert route.called
        body = json.loads(route.calls.last.request.content)
        assert "traces" in body
        assert isinstance(body["traces"], list)
        assert len(body["traces"]) == 1

    @pytest.mark.asyncio
    @respx.mock
    async def test_sends_authorization_header(self) -> None:
        route = respx.post("https://api.trulayer.ai/v1/ingest/batch").mock(
            return_value=httpx.Response(201, json={"ingested": 1, "ids": []})
        )

        from trulayer.batch import BatchSender

        sender = BatchSender(
            api_key="tl_secret",
            endpoint="https://api.trulayer.ai",
        )
        trace = TraceData(project_id="proj-1")
        await sender._send_with_retry([trace.to_wire()])

        request = route.calls.last.request
        assert request.headers["authorization"] == "Bearer tl_secret"

    @pytest.mark.asyncio
    @respx.mock
    async def test_sends_content_type_application_json(self) -> None:
        route = respx.post("https://api.trulayer.ai/v1/ingest/batch").mock(
            return_value=httpx.Response(201, json={"ingested": 1, "ids": []})
        )

        from trulayer.batch import BatchSender

        sender = BatchSender(
            api_key="tl_test",
            endpoint="https://api.trulayer.ai",
        )
        trace = TraceData(project_id="proj-1")
        await sender._send_with_retry([trace.to_wire()])

        request = route.calls.last.request
        assert "application/json" in request.headers["content-type"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_targets_v1_ingest_batch_endpoint(self) -> None:
        route = respx.post("https://api.trulayer.ai/v1/ingest/batch").mock(
            return_value=httpx.Response(201, json={"ingested": 1, "ids": []})
        )

        from trulayer.batch import BatchSender

        sender = BatchSender(
            api_key="tl_test",
            endpoint="https://api.trulayer.ai",
        )
        trace = TraceData(project_id="proj-1")
        await sender._send_with_retry([trace.to_wire()])

        assert route.called
        assert "/v1/ingest/batch" in str(route.calls.last.request.url)


# ---------------------------------------------------------------------------
# Feedback wire shape
# ---------------------------------------------------------------------------


class TestFeedbackWireShape:
    @respx.mock
    def test_posts_to_feedback_endpoint(self) -> None:
        route = respx.post("https://api.trulayer.ai/v1/feedback").mock(
            return_value=httpx.Response(201)
        )
        with patch("trulayer.batch.BatchSender.start"):
            client = TruLayerClient(api_key="tl_test", project_name="p")
        client.feedback(trace_id="trace-001", label="good")
        assert route.called

    @respx.mock
    def test_feedback_body_has_trace_id_and_label(self) -> None:
        route = respx.post("https://api.trulayer.ai/v1/feedback").mock(
            return_value=httpx.Response(201)
        )
        with patch("trulayer.batch.BatchSender.start"):
            client = TruLayerClient(api_key="tl_test", project_name="p")
        client.feedback(trace_id="trace-002", label="bad")

        body = json.loads(route.calls.last.request.content)
        assert body["trace_id"] == "trace-002"
        assert body["label"] == "bad"

    @respx.mock
    def test_feedback_body_includes_optional_score_and_comment(self) -> None:
        route = respx.post("https://api.trulayer.ai/v1/feedback").mock(
            return_value=httpx.Response(201)
        )
        with patch("trulayer.batch.BatchSender.start"):
            client = TruLayerClient(api_key="tl_test", project_name="p")
        client.feedback(trace_id="trace-003", label="good", score=0.9, comment="nice")

        body = json.loads(route.calls.last.request.content)
        assert body["score"] == pytest.approx(0.9)
        assert body["comment"] == "nice"

    @respx.mock
    def test_feedback_sends_authorization_header(self) -> None:
        route = respx.post("https://api.trulayer.ai/v1/feedback").mock(
            return_value=httpx.Response(201)
        )
        with patch("trulayer.batch.BatchSender.start"):
            client = TruLayerClient(api_key="tl_mykey", project_name="p")
        client.feedback(trace_id="trace-004", label="neutral")

        assert route.calls.last.request.headers["authorization"] == "Bearer tl_mykey"

    @respx.mock
    def test_feedback_does_not_raise_on_4xx(self) -> None:
        respx.post("https://api.trulayer.ai/v1/feedback").mock(return_value=httpx.Response(422))
        with patch("trulayer.batch.BatchSender.start"):
            client = TruLayerClient(api_key="tl_test", project_name="p")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # Must not raise
            client.feedback(trace_id="trace-005", label="good")
        # A warning must be emitted instead of raising
        assert any("feedback" in str(w.message).lower() for w in caught)

    @respx.mock
    def test_feedback_does_not_raise_on_401(self) -> None:
        respx.post("https://api.trulayer.ai/v1/feedback").mock(return_value=httpx.Response(401))
        with patch("trulayer.batch.BatchSender.start"):
            client = TruLayerClient(api_key="tl_test", project_name="p")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            client.feedback(trace_id="trace-006", label="bad")
        # No exception — SDK swallows HTTP errors from the feedback path


# ---------------------------------------------------------------------------
# Eval endpoint
# ---------------------------------------------------------------------------


class TestEvalWireShape:
    @respx.mock
    def test_eval_method_exists(self) -> None:
        with patch("trulayer.batch.BatchSender.start"):
            client = TruLayerClient(api_key="tl_test", project_name="p")
        assert hasattr(client, "eval")
        assert callable(client.eval)

    @respx.mock
    def test_eval_posts_to_v1_eval_endpoint(self) -> None:
        route = respx.post("https://api.trulayer.ai/v1/eval").mock(
            return_value=httpx.Response(
                202,
                json={"eval_id": "eval-abc", "status": "pending"},
            )
        )
        with patch("trulayer.batch.BatchSender.start"):
            client = TruLayerClient(api_key="tl_test", project_name="p")
        eval_id = client.eval(
            trace_id="trace-xyz",
            evaluator_type="llm",
            metric_name="correctness",
        )
        assert route.called
        assert eval_id == "eval-abc"

    @respx.mock
    def test_eval_body_shape(self) -> None:
        route = respx.post("https://api.trulayer.ai/v1/eval").mock(
            return_value=httpx.Response(
                202,
                json={"eval_id": "eval-123", "status": "pending"},
            )
        )
        with patch("trulayer.batch.BatchSender.start"):
            client = TruLayerClient(api_key="tl_test", project_name="p")
        client.eval(
            trace_id="trace-1",
            evaluator_type="rule",
            metric_name="length",
        )
        body = json.loads(route.calls.last.request.content)
        assert body == {
            "trace_id": "trace-1",
            "evaluator_type": "rule",
            "metric_name": "length",
        }

    @respx.mock
    def test_eval_sends_authorization_header(self) -> None:
        route = respx.post("https://api.trulayer.ai/v1/eval").mock(
            return_value=httpx.Response(202, json={"eval_id": "e1", "status": "pending"})
        )
        with patch("trulayer.batch.BatchSender.start"):
            client = TruLayerClient(api_key="tl_mykey", project_name="p")
        client.eval(trace_id="t", evaluator_type="llm", metric_name="m")
        assert route.calls.last.request.headers["authorization"] == "Bearer tl_mykey"

    @respx.mock
    def test_eval_returns_none_on_error_and_does_not_raise(self) -> None:
        respx.post("https://api.trulayer.ai/v1/eval").mock(return_value=httpx.Response(404))
        with patch("trulayer.batch.BatchSender.start"):
            client = TruLayerClient(api_key="tl_test", project_name="p")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = client.eval(trace_id="t", evaluator_type="llm", metric_name="m")
        assert result is None
        assert any("eval" in str(w.message).lower() for w in caught)
