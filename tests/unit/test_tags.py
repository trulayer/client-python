"""Key -> value tag tests.

Verifies the SDK accepts structured tags via ``trace(tag_map=...)`` and
``TraceContext.set_tag``, and that a non-empty tag map takes precedence over
the legacy string-array ``tags`` on the wire.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from trulayer.trace import TraceContext


def _make_client(project_id: str = "proj-tags", sample_rate: float = 1.0) -> MagicMock:
    client = MagicMock()
    client._project_id = project_id
    client._batch = MagicMock()
    client._sample_rate = sample_rate
    client._scrub_fn = None
    client._metadata_validator = None
    return client


def _captured_payload(client: MagicMock) -> dict[str, Any]:
    return client._batch.enqueue.call_args[0][0]


def test_tag_map_option_is_captured_and_sent_as_tags_field() -> None:
    client = _make_client()
    with TraceContext(client, name="t", tag_map={"env": "prod", "region": "us-east-1"}):
        pass

    payload = _captured_payload(client)
    assert payload["tags"] == {"env": "prod", "region": "us-east-1"}
    assert isinstance(payload["tags"], dict)


def test_set_tag_appends_individual_tags_to_the_map() -> None:
    client = _make_client()
    with TraceContext(client, name="t") as t:
        t.set_tag("env", "staging")
        t.set_tag("user_id", "abc-123")

    payload = _captured_payload(client)
    assert payload["tags"] == {"env": "staging", "user_id": "abc-123"}


def test_set_tag_overrides_an_earlier_value_for_the_same_key() -> None:
    client = _make_client()
    with TraceContext(client, name="t") as t:
        t.set_tag("env", "staging")
        t.set_tag("env", "prod")

    payload = _captured_payload(client)
    assert payload["tags"] == {"env": "prod"}


def test_non_empty_tag_map_takes_precedence_over_legacy_array_tags() -> None:
    client = _make_client()
    with TraceContext(client, name="t", tags=["from-options"]) as t:
        t.add_tag("legacy")
        t.set_tag("env", "prod")

    payload = _captured_payload(client)
    # Map form wins -- the legacy array is still preserved in memory but must
    # not appear on the wire when the map is non-empty.
    assert payload["tags"] == {"env": "prod"}
    assert isinstance(payload["tags"], dict)


def test_legacy_string_array_shape_preserved_when_no_tag_map_set() -> None:
    client = _make_client()
    with TraceContext(client, name="t") as t:
        t.add_tag("beta")
        t.add_tag("dogfood")

    payload = _captured_payload(client)
    assert payload["tags"] == ["beta", "dogfood"]
    assert isinstance(payload["tags"], list)


def test_empty_tag_map_does_not_shadow_legacy_array() -> None:
    client = _make_client()
    with TraceContext(client, name="t", tag_map={}) as t:
        t.add_tag("beta")

    payload = _captured_payload(client)
    assert payload["tags"] == ["beta"]
    assert isinstance(payload["tags"], list)
