---
description: Add a Pydantic model to src/trulayer/model.py. Usage: /model <name> — e.g. /model SpanData
---

Add a typed Pydantic v2 model to `src/trulayer/model.py`. The argument is: $ARGUMENTS

Parse the argument as: <name>
- name: PascalCase model class name (e.g. TraceData, SpanData, EventData)

Read `src/trulayer/model.py` first to understand existing models and field conventions.

Generate a Pydantic v2 model following these rules:
- Inherit from `pydantic.BaseModel`
- All IDs are `str` (UUIDv7 as string — generated via `uuid.uuid7()`)
- All timestamps are `datetime` with `default_factory=datetime.utcnow`
- Optional fields use `field: SomeType | None = None`
- Use `model_config = ConfigDict(extra="ignore")` so forward-compatible with new server fields
- Serializes to JSON via `.model_dump(mode="json")` — confirm field names match the TruLayer API JSON field names

Example shape:
```python
from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


class <name>(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: str(uuid.uuid7()))
    tenant_id: str
    # TODO: domain fields
    created_at: datetime = Field(default_factory=datetime.utcnow)
```

After generating:
1. Export the new model from `src/trulayer/__init__.py` if it's part of the public API.
2. Add a unit test in `tests/unit/test_model.py` verifying serialization round-trips correctly.
3. Cross-check field names against the TruLayer OpenAPI spec in `tests/fixtures/openapi.yaml`.
