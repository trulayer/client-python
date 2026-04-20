from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from trulayer._ids import new_id


def _now() -> datetime:
    return datetime.now(tz=UTC)


class SpanData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=new_id)
    trace_id: str = ""
    name: str
    span_type: str = "default"  # llm, tool, chain, retrieval, default
    input: str | None = None
    output: str | None = None
    error: bool = False
    error_message: str | None = None
    latency_ms: int | None = None
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=_now)
    ended_at: datetime | None = None


class TraceData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=new_id)
    project_id: str
    session_id: str | None = None
    external_id: str | None = None
    name: str | None = None
    input: str | None = None
    output: str | None = None
    model: str | None = None
    latency_ms: int | None = None
    cost: float | None = None
    error: bool = False
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    spans: list[SpanData] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=_now)
    ended_at: datetime | None = None


class EventData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=new_id)
    trace_id: str = ""
    span_id: str | None = None
    level: str = "info"  # debug, info, warning, error
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_now)


class FeedbackData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    trace_id: str
    label: str  # good, bad, neutral
    score: float | None = None
    comment: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
