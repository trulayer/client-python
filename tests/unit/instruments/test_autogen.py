"""Tests for the AutoGen agent instrumentation."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from trulayer.instruments.autogen import instrument_autogen
from trulayer.trace import TraceContext

# ---------------------------------------------------------------------------
# Mock AutoGen types
# ---------------------------------------------------------------------------


class MockChatResult:
    def __init__(self, summary: str = "Chat completed successfully") -> None:
        self.summary = summary


class MockConversableAgent:
    """Minimal stand-in for autogen.ConversableAgent."""

    def __init__(self, name: str = "assistant") -> None:
        self.name = name

    def initiate_chat(
        self, recipient: Any = None, message: str = "", *args: Any, **kwargs: Any
    ) -> MockChatResult:
        return MockChatResult()

    def generate_reply(self, messages: list[Any] | None = None, **kwargs: Any) -> str:
        return "I can help with that."


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
# 1. initiate_chat creates span with message
# ---------------------------------------------------------------------------


def test_initiate_chat_creates_span() -> None:
    client = _make_client()
    agent = MockConversableAgent(name="assistant")
    with TraceContext(client, name="trace") as ctx:
        instrument_autogen(agent, ctx)
        result = agent.initiate_chat(recipient=None, message="Hello, help me")

    assert result.summary == "Chat completed successfully"
    spans = _get_spans(client)
    chat_spans = [s for s in spans if s["name"] == "autogen:chat"]
    assert len(chat_spans) == 1
    assert chat_spans[0]["input"] == "Hello, help me"
    assert "Chat completed successfully" in chat_spans[0]["output"]


# ---------------------------------------------------------------------------
# 2. generate_reply creates child span
# ---------------------------------------------------------------------------


def test_generate_reply_creates_child_span() -> None:
    client = _make_client()
    agent = MockConversableAgent(name="assistant")
    with TraceContext(client, name="trace") as ctx:
        instrument_autogen(agent, ctx)
        result = agent.generate_reply(messages=[{"role": "user", "content": "Hi"}])

    assert result == "I can help with that."
    spans = _get_spans(client)
    reply_spans = [s for s in spans if s["name"] == "autogen:assistant:reply"]
    assert len(reply_spans) == 1
    assert reply_spans[0]["span_type"] == "llm"
    assert "Hi" in reply_spans[0]["input"]
    assert reply_spans[0]["output"] == "I can help with that."


# ---------------------------------------------------------------------------
# 3. capture_messages=False skips message recording
# ---------------------------------------------------------------------------


def test_capture_messages_false() -> None:
    client = _make_client()
    agent = MockConversableAgent(name="assistant")
    with TraceContext(client, name="trace") as ctx:
        instrument_autogen(agent, ctx, capture_messages=False)
        agent.initiate_chat(recipient=None, message="secret")
        agent.generate_reply(messages=[{"role": "user", "content": "secret"}])

    spans = _get_spans(client)
    chat_spans = [s for s in spans if s["name"] == "autogen:chat"]
    assert len(chat_spans) == 1
    assert chat_spans[0]["input"] is None

    reply_spans = [s for s in spans if s["name"] == "autogen:assistant:reply"]
    assert len(reply_spans) == 1
    assert reply_spans[0]["input"] is None


# ---------------------------------------------------------------------------
# 4. exception from initiate_chat re-raises with warning
# ---------------------------------------------------------------------------


def test_exception_from_initiate_chat() -> None:
    client = _make_client()

    class FailingAgent(MockConversableAgent):
        def initiate_chat(
            self, recipient: Any = None, message: str = "", *args: Any, **kwargs: Any
        ) -> MockChatResult:
            raise RuntimeError("chat failed")

    agent = FailingAgent(name="assistant")
    with (
        pytest.warns(match="error during AutoGen initiate_chat"),
        pytest.raises(RuntimeError, match="chat failed"),
        TraceContext(client, name="trace") as ctx,
    ):
        instrument_autogen(agent, ctx)
        agent.initiate_chat(recipient=None, message="hello")


# ---------------------------------------------------------------------------
# 5. agent name used in span name
# ---------------------------------------------------------------------------


def test_agent_name_in_span() -> None:
    client = _make_client()
    agent = MockConversableAgent(name="my-custom-bot")
    with TraceContext(client, name="trace") as ctx:
        instrument_autogen(agent, ctx)
        agent.generate_reply(messages=[])

    spans = _get_spans(client)
    reply_spans = [s for s in spans if "reply" in s["name"]]
    assert len(reply_spans) == 1
    assert reply_spans[0]["name"] == "autogen:my-custom-bot:reply"
