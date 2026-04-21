"""Tests for the Haystack pipeline instrumentation."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from trulayer.instruments.haystack import instrument_haystack
from trulayer.trace import TraceContext


# ---------------------------------------------------------------------------
# Mock Haystack types
# ---------------------------------------------------------------------------


class MockComponent:
    def __init__(self, name: str = "retriever") -> None:
        self._name = name

    def run(self, **kwargs: Any) -> dict[str, Any]:
        return {"documents": ["doc1", "doc2"]}


class MockGraph:
    """Minimal networkx DiGraph stand-in."""

    def __init__(self, components: dict[str, MockComponent] | None = None) -> None:
        self._components = components or {}
        self.nodes = _NodeView(self._components)


class _NodeView:
    """Minimal stand-in for networkx graph.nodes."""

    def __init__(self, components: dict[str, MockComponent]) -> None:
        self._components = components

    def __iter__(self) -> Any:
        return iter(self._components)

    def __getitem__(self, key: str) -> dict[str, Any]:
        return {"instance": self._components[key]}


class MockPipeline:
    """Minimal stand-in for haystack.core.pipeline.Pipeline."""

    def __init__(self, components: dict[str, MockComponent] | None = None) -> None:
        self.graph = MockGraph(components or {})

    def run(self, data: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        return {"results": "pipeline output"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client() -> MagicMock:
    client = MagicMock()
    client._project_id = "proj-1"
    client._batch = MagicMock()
    client._scrub_fn = None
    client._sample_rate = 1.0
    client._metadata_validator = None
    return client


def _get_spans(client: MagicMock) -> list[dict[str, Any]]:
    payload = client._batch.enqueue.call_args[0][0]
    return cast(list[dict[str, Any]], payload["spans"])


# ---------------------------------------------------------------------------
# 1. pipeline.run creates trace span
# ---------------------------------------------------------------------------


def test_pipeline_run_creates_span() -> None:
    client = _make_client()
    pipeline = MockPipeline()
    with TraceContext(client, name="trace") as ctx:
        instrument_haystack(pipeline, ctx)
        result = pipeline.run(data={"query": "hello"})

    assert result == {"results": "pipeline output"}
    spans = _get_spans(client)
    pipeline_spans = [s for s in spans if s["name"] == "haystack:pipeline"]
    assert len(pipeline_spans) == 1


# ---------------------------------------------------------------------------
# 2. inputs recorded when capture_inputs=True
# ---------------------------------------------------------------------------


def test_inputs_recorded() -> None:
    client = _make_client()
    pipeline = MockPipeline()
    with TraceContext(client, name="trace") as ctx:
        instrument_haystack(pipeline, ctx, capture_inputs=True)
        pipeline.run(data={"query": "search term"})

    spans = _get_spans(client)
    pipeline_spans = [s for s in spans if s["name"] == "haystack:pipeline"]
    assert len(pipeline_spans) == 1
    assert "search term" in pipeline_spans[0]["input"]


# ---------------------------------------------------------------------------
# 3. outputs recorded
# ---------------------------------------------------------------------------


def test_outputs_recorded() -> None:
    client = _make_client()
    pipeline = MockPipeline()
    with TraceContext(client, name="trace") as ctx:
        instrument_haystack(pipeline, ctx)
        pipeline.run(data={"query": "test"})

    spans = _get_spans(client)
    pipeline_spans = [s for s in spans if s["name"] == "haystack:pipeline"]
    assert len(pipeline_spans) == 1
    assert "pipeline output" in pipeline_spans[0]["output"]


# ---------------------------------------------------------------------------
# 4. component run creates child span
# ---------------------------------------------------------------------------


def test_component_run_creates_child_span() -> None:
    client = _make_client()
    retriever = MockComponent(name="retriever")
    pipeline = MockPipeline(components={"retriever": retriever})

    with TraceContext(client, name="trace") as ctx:
        instrument_haystack(pipeline, ctx)
        # Call the component's run directly to trigger child span
        retriever.run(query="test")

    spans = _get_spans(client)
    component_spans = [s for s in spans if s["name"] == "haystack:retriever"]
    assert len(component_spans) == 1


# ---------------------------------------------------------------------------
# 5. exception propagates, warning emitted
# ---------------------------------------------------------------------------


def test_exception_propagates() -> None:
    client = _make_client()

    class FailingPipeline(MockPipeline):
        def run(self, data: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
            raise ValueError("pipeline failed")

    pipeline = FailingPipeline()
    with pytest.warns(match="error during Haystack pipeline run"):
        with pytest.raises(ValueError, match="pipeline failed"):
            with TraceContext(client, name="trace") as ctx:
                instrument_haystack(pipeline, ctx)
                pipeline.run(data={"query": "test"})
