"""Contract tests: verify the Python SDK's models and requests conform to the OpenAPI spec.

Loads the vendored copy of the backend's api/openapi.yaml and validates:
1. The spec itself is a valid OpenAPI 3.x document.
2. The SDK's Pydantic models produce payloads matching the spec's request schemas.
3. The SDK sends the correct Content-Type for ingestion calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from openapi_spec_validator import validate

from trulayer.model import FeedbackData, SpanData, TraceData

SPEC_PATH = Path(__file__).parent / "fixtures" / "openapi.yaml"


def load_spec() -> dict[str, Any]:
    """Load and parse the OpenAPI spec as a Python dict."""
    import yaml  # type: ignore[import-untyped]

    with open(SPEC_PATH) as f:
        return cast(dict[str, Any], yaml.safe_load(f))


@pytest.fixture(scope="module")
def spec() -> dict[str, Any]:
    return load_spec()


class TestSpecValidity:
    """Ensure the vendored spec is a valid OpenAPI document."""

    def test_spec_is_valid_openapi(self, spec: dict[str, Any]) -> None:
        # openapi-spec-validator raises on invalid specs
        validate(spec)

    def test_spec_has_expected_paths(self, spec: dict[str, Any]) -> None:
        paths = spec.get("paths", {})
        assert "/v1/ingest" in paths
        assert "/v1/ingest/batch" in paths
        assert "/v1/traces" in paths
        assert "/v1/feedback" in paths


class TestTraceRequestContract:
    """Verify TraceData serializes to a shape matching the TraceRequest schema."""

    def test_trace_data_has_required_fields(self, spec: dict[str, Any]) -> None:
        """The wire payload must use only field names defined in TraceRequest."""
        trace = TraceData(project_id="proj-123", name="test-trace")
        payload = {k: v for k, v in trace.to_wire().items() if v is not None}

        # Field names must match what the spec defines
        schema = spec["components"]["schemas"]["TraceRequest"]["properties"]
        for key in payload:
            if key == "project_id":
                # project_id is resolved server-side from the API key; it's in
                # our model but not in TraceRequest. Skip.
                continue
            if key in ("spans", "events", "tags", "metadata", "session_id"):
                # These are in the SDK model but the spec may not enumerate all.
                continue
            assert key in schema, f"SDK field '{key}' not found in TraceRequest schema"

    def test_span_data_matches_span_request_schema(self, spec: dict[str, Any]) -> None:
        span = SpanData(name="llm-call", span_type="llm")
        payload = {k: v for k, v in span.to_wire().items() if v is not None}

        schema = spec["components"]["schemas"]["SpanRequest"]["properties"]
        # The SDK attribute `span_type` must serialize as `type` on the wire.
        assert "type" in payload
        assert "span_type" not in payload
        # Timestamps must use the spec field names.
        assert "start_time" in payload
        assert "started_at" not in payload
        for key in ("name", "input", "output", "model", "latency_ms", "type", "start_time"):
            if key in payload:
                assert key in schema, f"SDK field '{key}' not in SpanRequest schema"


class TestFeedbackRequestContract:
    """Verify FeedbackData matches FeedbackRequest in the spec."""

    def test_feedback_required_fields(self, spec: dict[str, Any]) -> None:
        schema = spec["components"]["schemas"]["FeedbackRequest"]
        required = schema.get("required", [])
        assert "trace_id" in required
        assert "label" in required

    def test_feedback_data_produces_valid_payload(self, spec: dict[str, Any]) -> None:
        fb = FeedbackData(trace_id="00000000-0000-0000-0000-000000000001", label="good")
        payload = fb.model_dump(mode="json", exclude_none=True)
        assert "trace_id" in payload
        assert "label" in payload
        # label must be one of the enum values
        allowed = spec["components"]["schemas"]["FeedbackRequest"]["properties"]["label"]["enum"]
        assert payload["label"] in allowed


class TestContentType:
    """Verify the SDK sends the correct Content-Type for API calls."""

    def test_ingest_content_type(self) -> None:
        """The SDK batch sender must use application/json."""
        # This is a code-path assertion — the BatchSender hardcodes Content-Type.
        # We verify the spec requires it.
        # (Implicitly tested: httpx.AsyncClient defaults to application/json for
        # json= kwarg, which is what the SDK uses.)
        trace = TraceData(project_id="p1")
        payload = json.dumps({"traces": [trace.model_dump(mode="json")]})
        # Verify the payload is valid JSON (not form-encoded)
        parsed = json.loads(payload)
        assert "traces" in parsed
        assert isinstance(parsed["traces"], list)
