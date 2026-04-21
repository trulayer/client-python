"""Tests for the CrewAI crew instrumentation."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from trulayer.instruments.crewai import instrument_crewai
from trulayer.trace import TraceContext

# ---------------------------------------------------------------------------
# Mock CrewAI types
# ---------------------------------------------------------------------------


class MockTask:
    def __init__(self, description: str = "Write a poem about AI") -> None:
        self.description = description


class MockCrewResult:
    def __init__(self, raw: str = "Here is the poem") -> None:
        self.raw = raw


class MockAgent:
    def __init__(self, name: str = "poet") -> None:
        self.name = name

    def execute_task(self, task: Any, **kwargs: Any) -> str:
        return "task done"


class MockCrew:
    """Minimal stand-in for crewai.Crew."""

    def __init__(
        self,
        agents: list[Any] | None = None,
        inputs: dict[str, Any] | None = None,
    ) -> None:
        self.agents = agents or []
        self.inputs = inputs

    def kickoff(self, *args: Any, **kwargs: Any) -> MockCrewResult:
        return MockCrewResult()


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
# 1. kickoff creates trace span with input
# ---------------------------------------------------------------------------


def test_kickoff_creates_span_with_input() -> None:
    client = _make_client()
    crew = MockCrew(inputs={"topic": "AI"})
    with TraceContext(client, name="trace") as ctx:
        instrument_crewai(crew, ctx)
        result = crew.kickoff()

    assert result.raw == "Here is the poem"
    spans = _get_spans(client)
    kickoff_spans = [s for s in spans if s["name"] == "crewai:kickoff"]
    assert len(kickoff_spans) == 1
    assert "AI" in kickoff_spans[0]["input"]


# ---------------------------------------------------------------------------
# 2. kickoff records output
# ---------------------------------------------------------------------------


def test_kickoff_records_output() -> None:
    client = _make_client()
    crew = MockCrew()
    with TraceContext(client, name="trace") as ctx:
        instrument_crewai(crew, ctx)
        crew.kickoff()

    spans = _get_spans(client)
    kickoff_spans = [s for s in spans if s["name"] == "crewai:kickoff"]
    assert len(kickoff_spans) == 1
    assert kickoff_spans[0]["output"] == "Here is the poem"


# ---------------------------------------------------------------------------
# 3. capture_inputs=False skips input recording
# ---------------------------------------------------------------------------


def test_capture_inputs_false() -> None:
    client = _make_client()
    crew = MockCrew(inputs={"topic": "secret"})
    with TraceContext(client, name="trace") as ctx:
        instrument_crewai(crew, ctx, capture_inputs=False)
        crew.kickoff()

    spans = _get_spans(client)
    kickoff_spans = [s for s in spans if s["name"] == "crewai:kickoff"]
    assert len(kickoff_spans) == 1
    assert kickoff_spans[0]["input"] is None


# ---------------------------------------------------------------------------
# 4. agent task creates child span
# ---------------------------------------------------------------------------


def test_agent_task_creates_child_span() -> None:
    client = _make_client()
    agent = MockAgent(name="poet")
    crew = MockCrew(agents=[agent])
    task = MockTask(description="Write a poem about AI")

    with TraceContext(client, name="trace") as ctx:
        instrument_crewai(crew, ctx)
        result = agent.execute_task(task)

    assert result == "task done"
    spans = _get_spans(client)
    task_spans = [s for s in spans if s["name"].startswith("crewai:task:")]
    assert len(task_spans) == 1
    assert "Write a poem" in task_spans[0]["name"]


# ---------------------------------------------------------------------------
# 5. exception from kickoff re-raised, warning emitted
# ---------------------------------------------------------------------------


def test_exception_from_kickoff_reraises() -> None:
    client = _make_client()

    class FailingCrew(MockCrew):
        def kickoff(self, *args: Any, **kwargs: Any) -> MockCrewResult:
            raise RuntimeError("crew failed")

    crew = FailingCrew()
    with (
        pytest.warns(match="error during CrewAI kickoff"),
        pytest.raises(RuntimeError, match="crew failed"),
        TraceContext(client, name="trace") as ctx,
    ):
        instrument_crewai(crew, ctx)
        crew.kickoff()


# ---------------------------------------------------------------------------
# 6. crew with no agents doesn't crash
# ---------------------------------------------------------------------------


def test_crew_no_agents() -> None:
    client = _make_client()
    crew = MockCrew(agents=[])
    with TraceContext(client, name="trace") as ctx:
        instrument_crewai(crew, ctx)
        result = crew.kickoff()

    assert result.raw == "Here is the poem"
    spans = _get_spans(client)
    assert len(spans) == 1
    assert spans[0]["name"] == "crewai:kickoff"
