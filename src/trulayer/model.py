from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from trulayer._ids import new_id


def _now() -> datetime:
    return datetime.now(tz=UTC)


class SpanData(BaseModel):
    """SDK-facing span model.

    Attributes are named ergonomically for Python callers (``span_type``,
    ``started_at``, ``ended_at``). When serialized to the wire via
    ``model_dump(mode="json", by_alias=True)`` the JSON keys match the
    TruLayer ingestion schema (``type``, ``start_time``, ``end_time``).

    The trace-level and span-level ``error`` field is a single
    ``string | null`` carrying the error message (or ``None`` when the
    span/trace succeeded). Both the SDK model and the wire format use the
    same shape.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str = Field(default_factory=new_id)
    trace_id: str = ""
    name: str
    span_type: str = Field(default="other", serialization_alias="type")
    input: str | None = None
    output: str | None = None
    error: str | None = None
    latency_ms: int | None = None
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=_now, serialization_alias="start_time")
    ended_at: datetime | None = Field(default=None, serialization_alias="end_time")

    def to_wire(self) -> dict[str, Any]:
        """Serialize this span to the JSON shape expected by ``POST /v1/ingest``.

        Renames ``span_type`` → ``type``, ``started_at`` → ``start_time``,
        ``ended_at`` → ``end_time``. The ``error`` field is forwarded as-is
        (``string | null``).
        """
        return self.model_dump(mode="json", by_alias=True)


class TraceData(BaseModel):
    """SDK-facing trace model.

    The ``error`` field is a single ``string | null`` carrying the error
    message (or ``None`` when the trace succeeded), matching the TruLayer
    ingestion schema.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

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
    error: str | None = None
    tags: list[str] = Field(default_factory=list)
    # Optional structured key -> value tags. When non-empty, emitted as the
    # ``tags`` wire field (object form) and takes precedence over the legacy
    # string-array ``tags``. Enables server-side filtering by tag key/value
    # pair on list endpoints. Limits: max 20 keys, 64 chars per key/value.
    tag_map: dict[str, str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    spans: list[SpanData] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=_now)
    ended_at: datetime | None = None

    def to_wire(self) -> dict[str, Any]:
        """Serialize this trace (and nested spans) to the JSON shape expected
        by ``POST /v1/ingest`` / ``POST /v1/ingest/batch``.
        """
        payload = self.model_dump(
            mode="json",
            by_alias=True,
            exclude={"spans", "tag_map"},
        )
        payload["spans"] = [s.to_wire() for s in self.spans]
        # A non-empty tag map wins over the legacy string-array on the wire.
        if self.tag_map:
            payload["tags"] = dict(self.tag_map)
        return payload


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
