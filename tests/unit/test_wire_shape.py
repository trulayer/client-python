"""Wire-shape contract tests.

These tests verify the exact JSON payload the SDK produces before it reaches
the network layer.  They use LocalBatchSender or mock senders so no real HTTP
request is made.

FINDINGS DOCUMENTED AS ASSERTIONS — tests marked with 'KNOWN:' reflect known
divergences between the SDK and the backend OpenAPI spec (SpanRequest).
Engineers must fix the underlying production code; these tests serve as
regression anchors so the fix is verifiable.
"""
from __future__ import annotations

import json
import warnings
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import respx

from trulayer.client import TruLayerClient
from trulayer.local_batch import LocalBatchSender
from trulayer.model import TraceData
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

    def test_trace_marks_error_on_exception(self) -> None:
        client, sender = create_test_client(project_name="proj-wire")
        with pytest.raises(ValueError):
            with client.trace("err-trace"):
                raise ValueError("boom")

        traces = sender.traces
        assert len(traces) == 1
        assert traces[0]["error"] is True

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

    # KNOWN: TraceRequest.error is string | null in the spec (error message).
    # The SDK sends error: bool.  This test documents current SDK behaviour.
    def test_known_error_field_is_boolean_not_string(self) -> None:
        client, sender = create_test_client(project_name="proj-wire")
        with pytest.raises(RuntimeError):
            with client.trace("err-bool"):
                raise RuntimeError("spec says string, sdk sends bool")

        p = sender.traces[0]
        # Current SDK sends boolean
        assert isinstance(p["error"], bool)
        assert p["error"] is True
        # The error message is NOT captured at the trace level in the current SDK


# ---------------------------------------------------------------------------
# Span payload shape
# ---------------------------------------------------------------------------


class TestSpanWireShape:
    def test_span_included_in_trace_spans_array(self) -> None:
        client, sender = create_test_client(project_name="proj-wire")
        with client.trace("with-span") as t:
            with t.span("llm-call", span_type="llm") as s:
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

    def test_span_error_set_on_exception(self) -> None:
        client, sender = create_test_client(project_name="proj-wire")
        with pytest.raises(ValueError):
            with client.trace("span-err") as t:
                with t.span("bad-span"):
                    raise ValueError("span failed")

        spans = sender.traces[0]["spans"]
        assert len(spans) == 1
        sp = spans[0]
        assert sp["error"] is True
        assert "span failed" in (sp["error_message"] or "")

    # KNOWN: SpanRequest.type field name in spec vs span_type in SDK.
    # The backend OpenAPI SpanRequest schema uses `type`; the SDK sends
    # `span_type`.  This test documents current SDK behaviour.
    def test_known_span_field_is_span_type_not_type(self) -> None:
        client, sender = create_test_client(project_name="proj-wire")
        with client.trace("field-name") as t:
            with t.span("s", span_type="llm"):
                pass

        sp = sender.traces[0]["spans"][0]
        # SDK serialises to span_type
        assert "span_type" in sp
        assert "type" not in sp

    # KNOWN: SpanRequest.start_time / end_time (spec) vs started_at / ended_at (SDK).
    def test_known_span_timestamps_use_started_at_ended_at(self) -> None:
        client, sender = create_test_client(project_name="proj-wire")
        with client.trace("ts-fields") as t:
            with t.span("s"):
                pass

        sp = sender.traces[0]["spans"][0]
        assert "started_at" in sp
        assert "ended_at" in sp
        assert "start_time" not in sp
        assert "end_time" not in sp

    # KNOWN: SpanType enum mismatch.
    # Spec enum: [llm, tool, retrieval, other]
    # SDK accepts: any string — common values are llm, tool, retrieval, chain, default.
    def test_known_span_type_chain_and_default_are_producible(self) -> None:
        client, sender = create_test_client(project_name="proj-wire")
        with client.trace("enum-test") as t:
            with t.span("chain-span", span_type="chain"):
                pass
            with t.span("default-span", span_type="default"):
                pass

        types = [sp["span_type"] for sp in sender.traces[0]["spans"]]
        assert "chain" in types
        assert "default" in types
        # 'other' (spec-only) is not enforced by the SDK

    # KNOWN: SpanRequest.cost field exists in spec but SpanData has no cost.
    def test_known_span_has_no_cost_field(self) -> None:
        client, sender = create_test_client(project_name="proj-wire")
        with client.trace("cost-check") as t:
            with t.span("s"):
                pass

        sp = sender.traces[0]["spans"][0]
        assert "cost" not in sp


# ---------------------------------------------------------------------------
# Batch sender wire format — verified via mock of _send_with_retry
# ---------------------------------------------------------------------------


class TestBatchSenderWireFormat:
    """Verify the exact JSON body the BatchSender would POST.

    The BatchSender runs in a background thread, making real-network assertions
    timing-sensitive.  Instead we call _send_with_retry directly with a
    respx-mocked httpx client.
    """

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
        await sender._send_with_retry([trace.model_dump(mode="json")])

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
        await sender._send_with_retry([trace.model_dump(mode="json")])

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
        await sender._send_with_retry([trace.model_dump(mode="json")])

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
        await sender._send_with_retry([trace.model_dump(mode="json")])

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
        respx.post("https://api.trulayer.ai/v1/feedback").mock(
            return_value=httpx.Response(422)
        )
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
        respx.post("https://api.trulayer.ai/v1/feedback").mock(
            return_value=httpx.Response(401)
        )
        with patch("trulayer.batch.BatchSender.start"):
            client = TruLayerClient(api_key="tl_test", project_name="p")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            client.feedback(trace_id="trace-006", label="bad")
        # No exception — SDK swallows HTTP errors from the feedback path


# ---------------------------------------------------------------------------
# Eval endpoint — SDK surface gap
# ---------------------------------------------------------------------------


class TestEvalSurfaceGap:
    # SPEC GAP: POST /v1/eval (EvalTriggerRequest) is not exposed by the SDK.
    # Neither TruLayerClient nor any public helper sends to /v1/eval.
    # This test documents the absence so the gap is explicit and tracked.
    def test_eval_method_absent_from_client(self) -> None:
        with patch("trulayer.batch.BatchSender.start"):
            client = TruLayerClient(api_key="tl_test", project_name="p")
        assert not hasattr(client, "eval"), (
            "TruLayerClient.eval() method does not exist — POST /v1/eval is not "
            "accessible via the Python SDK. Engineers must add this method."
        )
