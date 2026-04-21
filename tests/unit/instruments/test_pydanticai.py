"""Tests for the PydanticAI agent instrumentation."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from trulayer.instruments.pydanticai import instrument_pydanticai
from trulayer.trace import TraceContext

# ---------------------------------------------------------------------------
# Mock PydanticAI types
# ---------------------------------------------------------------------------


class MockUsage:
    def __init__(self, request_tokens: int = 10, response_tokens: int = 5) -> None:
        self.request_tokens = request_tokens
        self.response_tokens = response_tokens


class MockResult:
    def __init__(self, data: str = "Paris") -> None:
        self.data = data
        self._usage = MockUsage()

    def usage(self) -> MockUsage:
        return self._usage


class MockAgent:
    """Minimal stand-in for pydantic_ai.Agent."""

    def __init__(self, name: str = "test-agent") -> None:
        self.name = name
        self._function_tools: dict[str, Any] = {}

    async def run(self, prompt: str, *, deps: Any = None, **kwargs: Any) -> MockResult:
        return MockResult()

    def run_sync(self, prompt: str, *, deps: Any = None, **kwargs: Any) -> MockResult:
        return MockResult()

    async def run_stream(self, prompt: str, **kwargs: Any) -> Any:
        return MockStreamResult()


class MockStreamResult:
    """Minimal stand-in for a PydanticAI streaming result."""

    def __init__(self) -> None:
        self._chunks = ["Par", "is"]

    async def stream_response(self) -> Any:
        for chunk in self._chunks:
            yield chunk


class MockToolObj:
    """Minimal stand-in for a PydanticAI tool entry."""

    def __init__(self, fn: Any) -> None:
        self.function = fn


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
    """Extract spans from the enqueued payload."""
    payload = client._batch.enqueue.call_args[0][0]
    return cast(list[dict[str, Any]], payload["spans"])


# ---------------------------------------------------------------------------
# 1. test_instrument_pydanticai_async_run
# ---------------------------------------------------------------------------


async def test_instrument_pydanticai_async_run() -> None:
    client = _make_client()
    agent = MockAgent()
    with TraceContext(client, name="trace") as ctx:
        instrument_pydanticai(agent, ctx)
        result = await agent.run("What is the capital of France?")

    assert result.data == "Paris"
    spans = _get_spans(client)
    agent_spans = [s for s in spans if s["span_type"] == "agent"]
    assert len(agent_spans) == 1
    span = agent_spans[0]
    assert span["name"] == "pydanticai:test-agent"
    assert span["input"] == "What is the capital of France?"
    assert span["output"] == "Paris"
    assert span["metadata"]["usage"]["request_tokens"] == 10
    assert span["metadata"]["usage"]["response_tokens"] == 5


# ---------------------------------------------------------------------------
# 2. test_instrument_pydanticai_sync_run
# ---------------------------------------------------------------------------


def test_instrument_pydanticai_sync_run() -> None:
    client = _make_client()
    agent = MockAgent()
    with TraceContext(client, name="trace") as ctx:
        instrument_pydanticai(agent, ctx)
        result = agent.run_sync("What is the capital of France?")

    assert result.data == "Paris"
    spans = _get_spans(client)
    agent_spans = [s for s in spans if s["span_type"] == "agent"]
    assert len(agent_spans) == 1
    span = agent_spans[0]
    assert span["name"] == "pydanticai:test-agent"
    assert span["input"] == "What is the capital of France?"
    assert span["output"] == "Paris"


# ---------------------------------------------------------------------------
# 3. test_stream_run_span_captured
# ---------------------------------------------------------------------------


async def test_stream_run_span_captured() -> None:
    client = _make_client()
    agent = MockAgent()
    with TraceContext(client, name="trace") as ctx:
        instrument_pydanticai(agent, ctx)
        stream_result = await agent.run_stream("stream prompt")
        collected: list[str] = []
        async for chunk in stream_result.stream_response():
            collected.append(chunk)

    assert "".join(collected) == "Paris"
    spans = _get_spans(client)
    agent_spans = [s for s in spans if s["span_type"] == "agent"]
    assert len(agent_spans) == 1
    span = agent_spans[0]
    assert span["input"] == "stream prompt"
    assert span["output"] == "Paris"


# ---------------------------------------------------------------------------
# 4. test_capture_input_false
# ---------------------------------------------------------------------------


async def test_capture_input_false() -> None:
    client = _make_client()
    agent = MockAgent()
    with TraceContext(client, name="trace") as ctx:
        instrument_pydanticai(agent, ctx, capture_input=False)
        await agent.run("secret prompt")

    spans = _get_spans(client)
    agent_spans = [s for s in spans if s["span_type"] == "agent"]
    assert len(agent_spans) == 1
    assert agent_spans[0]["input"] is None


# ---------------------------------------------------------------------------
# 5. test_capture_output_false
# ---------------------------------------------------------------------------


async def test_capture_output_false() -> None:
    client = _make_client()
    agent = MockAgent()
    with TraceContext(client, name="trace") as ctx:
        instrument_pydanticai(agent, ctx, capture_output=False)
        await agent.run("prompt")

    spans = _get_spans(client)
    agent_spans = [s for s in spans if s["span_type"] == "agent"]
    assert len(agent_spans) == 1
    assert agent_spans[0]["output"] is None


# ---------------------------------------------------------------------------
# 6. test_run_name_override
# ---------------------------------------------------------------------------


async def test_run_name_override() -> None:
    client = _make_client()
    agent = MockAgent()
    with TraceContext(client, name="trace") as ctx:
        instrument_pydanticai(agent, ctx, run_name="my-custom-agent")
        await agent.run("prompt")

    spans = _get_spans(client)
    agent_spans = [s for s in spans if s["span_type"] == "agent"]
    assert len(agent_spans) == 1
    assert agent_spans[0]["name"] == "my-custom-agent"


# ---------------------------------------------------------------------------
# 7. test_tool_call_child_span
# ---------------------------------------------------------------------------


async def test_tool_call_child_span() -> None:
    client = _make_client()
    agent = MockAgent()

    async def get_weather(city: str = "London") -> str:
        return "sunny"

    agent._function_tools["get_weather"] = MockToolObj(get_weather)

    with TraceContext(client, name="trace") as ctx:
        instrument_pydanticai(agent, ctx)
        # Call the tool directly to verify span is created
        tool_fn = agent._function_tools["get_weather"].function
        result = await tool_fn(city="London")

    assert result == "sunny"
    spans = _get_spans(client)
    tool_spans = [s for s in spans if s["span_type"] == "tool"]
    assert len(tool_spans) == 1
    assert tool_spans[0]["name"] == "get_weather"
    assert "London" in tool_spans[0]["input"]
    assert tool_spans[0]["output"] == "sunny"


# ---------------------------------------------------------------------------
# 8. test_exception_propagates
# ---------------------------------------------------------------------------


async def test_exception_propagates() -> None:
    client = _make_client()

    class FailingAgent(MockAgent):
        async def run(self, prompt: str, *, deps: Any = None, **kwargs: Any) -> MockResult:
            raise ValueError("model exploded")

    agent = FailingAgent()
    with (
        pytest.raises(ValueError, match="model exploded"),
        TraceContext(client, name="trace") as ctx,
    ):
        instrument_pydanticai(agent, ctx)
        await agent.run("prompt")

    spans = _get_spans(client)
    agent_spans = [s for s in spans if s["span_type"] == "agent"]
    assert len(agent_spans) == 1
    assert agent_spans[0]["metadata"]["error"] == "model exploded"
    assert agent_spans[0]["error"] is True
