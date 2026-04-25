import json

from trulayer.model import FeedbackData, SpanData, TraceData


def test_trace_data_defaults() -> None:
    t = TraceData(project_id="proj-1")
    assert t.id  # UUIDv7 auto-generated
    assert t.project_id == "proj-1"
    assert t.spans == []
    assert t.error is None


def test_span_data_defaults() -> None:
    s = SpanData(name="llm-call")
    assert s.id
    assert s.span_type == "other"
    assert s.error is None


def test_trace_data_serialises_to_json() -> None:
    t = TraceData(project_id="proj-1", name="test", input="hi", output="hello")
    data = t.model_dump(mode="json")
    assert data["project_id"] == "proj-1"
    # Verify round-trip
    t2 = TraceData(**data)
    assert t2.id == t.id


def test_feedback_data() -> None:
    fb = FeedbackData(trace_id="abc", label="good", score=1.0)
    assert fb.label == "good"
    dumped = json.loads(fb.model_dump_json())
    assert dumped["label"] == "good"
    assert dumped["score"] == 1.0


def test_trace_with_span_serialises() -> None:
    t = TraceData(project_id="proj-1")
    s = SpanData(name="llm-call", trace_id=t.id, span_type="llm", input="hi", output="hello")
    t.spans.append(s)
    data = t.model_dump(mode="json")
    assert len(data["spans"]) == 1
    assert data["spans"][0]["name"] == "llm-call"
