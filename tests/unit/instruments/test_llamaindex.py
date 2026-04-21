"""Tests for the LlamaIndex callback handler."""

from __future__ import annotations

import importlib
import sys
from enum import Enum
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from trulayer.trace import TraceContext

# ---------------------------------------------------------------------------
# Fake LlamaIndex types — injected into sys.modules before importing the handler
# ---------------------------------------------------------------------------


class FakeCBEventType(Enum):
    LLM = "llm"
    QUERY = "query"
    RETRIEVE = "retrieve"
    EMBEDDING = "embedding"


class FakeBaseCallbackHandler:
    """Minimal stand-in for llama_index.core.callbacks.base_handler.BaseCallbackHandler."""

    def __init__(
        self,
        event_starts_to_ignore: list[Any] | None = None,
        event_ends_to_ignore: list[Any] | None = None,
    ) -> None:
        self.event_starts_to_ignore = event_starts_to_ignore or []
        self.event_ends_to_ignore = event_ends_to_ignore or []


def _install_fake_llama_index() -> dict[str, Any]:
    """Install fake llama_index modules and return them for cleanup."""
    callbacks_mod = SimpleNamespace(
        BaseCallbackHandler=FakeBaseCallbackHandler,
        CBEventType=FakeCBEventType,
        CallbackManager=MagicMock,
    )
    base_handler_mod = SimpleNamespace(
        BaseCallbackHandler=FakeBaseCallbackHandler,
    )
    core_mod = SimpleNamespace(callbacks=callbacks_mod)
    top_mod = SimpleNamespace(core=core_mod)

    fakes = {
        "llama_index": top_mod,
        "llama_index.core": core_mod,
        "llama_index.core.callbacks": callbacks_mod,
        "llama_index.core.callbacks.base_handler": base_handler_mod,
    }
    for k, v in fakes.items():
        sys.modules[k] = v  # type: ignore[assignment]
    return fakes


def _uninstall_fake_llama_index(fakes: dict[str, Any]) -> None:
    for k in fakes:
        sys.modules.pop(k, None)
    # Also remove the cached handler module so it re-evaluates
    sys.modules.pop("trulayer.instruments.llamaindex", None)


# Install fakes before importing the handler module
_fakes = _install_fake_llama_index()

import trulayer.instruments.llamaindex as _li_mod  # noqa: E402, PLC0415

importlib.reload(_li_mod)  # reload with fakes in place

from trulayer.instruments.llamaindex import TruLayerCallbackHandler  # noqa: E402, PLC0415

# Map the fake CBEventType to the one that the module actually imported
CBEventType = FakeCBEventType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(project_id: str = "proj-1") -> MagicMock:
    client = MagicMock()
    client._project_id = project_id
    client._batch = MagicMock()
    client._scrub_fn = None
    client._sample_rate = 1.0
    client._metadata_validator = None
    return client


def _make_handler() -> TruLayerCallbackHandler:
    client = _make_client()
    return TruLayerCallbackHandler(client)


# ---------------------------------------------------------------------------
# on_event_start — LLM
# ---------------------------------------------------------------------------


def test_on_event_start_llm_opens_span() -> None:
    handler = _make_handler()
    event_id = handler.on_event_start(
        CBEventType.LLM,
        payload={"messages": [{"role": "user", "content": "hello"}]},
        event_id="evt-1",
    )
    assert event_id == "evt-1"
    assert "evt-1" in handler._open_spans
    _, span_type, input_text = handler._open_spans["evt-1"]
    assert span_type == "llm"
    assert "hello" in input_text


# ---------------------------------------------------------------------------
# on_event_end — LLM (closes span with output)
# ---------------------------------------------------------------------------


def test_on_event_end_llm_closes_span_with_output() -> None:
    client = _make_client()
    handler = TruLayerCallbackHandler(client)

    with TraceContext(client, name="test-trace"):
        handler.on_event_start(
            CBEventType.LLM,
            payload={"messages": [{"role": "user", "content": "prompt"}]},
            event_id="evt-2",
        )
        handler.on_event_end(
            CBEventType.LLM,
            payload={"response": "the answer is 42"},
            event_id="evt-2",
        )

    payload = client._batch.enqueue.call_args[0][0]
    span = payload["spans"][0]
    assert span["name"] == "llamaindex.llm"
    assert span["span_type"] == "llm"
    assert "prompt" in span["input"]
    assert span["output"] == "the answer is 42"


# ---------------------------------------------------------------------------
# on_event_start — QUERY
# ---------------------------------------------------------------------------


def test_on_event_start_query_opens_retrieval_span() -> None:
    handler = _make_handler()
    event_id = handler.on_event_start(
        CBEventType.QUERY,
        payload={"query_str": "what is the meaning of life?"},
        event_id="evt-3",
    )
    assert event_id == "evt-3"
    _, span_type, input_text = handler._open_spans["evt-3"]
    assert span_type == "retrieval"
    assert input_text == "what is the meaning of life?"


# ---------------------------------------------------------------------------
# on_event_start — RETRIEVE
# ---------------------------------------------------------------------------


def test_on_event_start_retrieve_opens_retrieval_span() -> None:
    handler = _make_handler()
    handler.on_event_start(
        CBEventType.RETRIEVE,
        payload={"query_str": "search query"},
        event_id="evt-retrieve",
    )
    _, span_type, input_text = handler._open_spans["evt-retrieve"]
    assert span_type == "retrieval"
    assert input_text == "search query"


# ---------------------------------------------------------------------------
# on_event_end — QUERY (closes span with response output)
# ---------------------------------------------------------------------------


def test_on_event_end_query_closes_span_with_output() -> None:
    client = _make_client()
    handler = TruLayerCallbackHandler(client)

    with TraceContext(client, name="query-trace"):
        handler.on_event_start(
            CBEventType.QUERY,
            payload={"query_str": "my query"},
            event_id="evt-4",
        )
        handler.on_event_end(
            CBEventType.QUERY,
            payload={"response": "query result"},
            event_id="evt-4",
        )

    payload = client._batch.enqueue.call_args[0][0]
    span = payload["spans"][0]
    assert span["name"] == "llamaindex.retrieval"
    assert span["span_type"] == "retrieval"
    assert span["input"] == "my query"
    assert span["output"] == "query result"


# ---------------------------------------------------------------------------
# Unknown event type uses default span type
# ---------------------------------------------------------------------------


def test_unknown_event_type_uses_default() -> None:
    handler = _make_handler()
    handler.on_event_start(
        CBEventType.EMBEDDING,
        payload=None,
        event_id="evt-5",
    )
    _, span_type, _ = handler._open_spans["evt-5"]
    assert span_type == "default"


# ---------------------------------------------------------------------------
# on_event_end for unknown event_id is a no-op
# ---------------------------------------------------------------------------


def test_on_event_end_unknown_event_id_is_noop() -> None:
    handler = _make_handler()
    # Should not crash
    handler.on_event_end(
        CBEventType.LLM,
        payload={"response": "something"},
        event_id="nonexistent",
    )


# ---------------------------------------------------------------------------
# on_event_end outside a trace is a no-op
# ---------------------------------------------------------------------------


def test_on_event_end_outside_trace_is_noop() -> None:
    client = _make_client()
    handler = TruLayerCallbackHandler(client)
    handler.on_event_start(CBEventType.LLM, payload=None, event_id="evt-6")
    handler.on_event_end(CBEventType.LLM, payload=None, event_id="evt-6")
    client._batch.enqueue.assert_not_called()


# ---------------------------------------------------------------------------
# start_trace / end_trace are no-ops
# ---------------------------------------------------------------------------


def test_start_trace_and_end_trace_are_noops() -> None:
    handler = _make_handler()
    handler.start_trace(trace_id="t1")
    handler.end_trace(trace_id="t1", trace_map={"root": ["child"]})
    # No crash or side effects


# ---------------------------------------------------------------------------
# ImportError when llama_index is not installed
# ---------------------------------------------------------------------------


def test_import_error_when_llama_index_not_installed() -> None:
    """Verify that importing the module without llama_index raises ImportError."""
    saved: dict[str, Any] = {}
    keys_to_block = [
        "llama_index",
        "llama_index.core",
        "llama_index.core.callbacks",
        "llama_index.core.callbacks.base_handler",
    ]
    mod_key = "trulayer.instruments.llamaindex"

    for k in keys_to_block:
        saved[k] = sys.modules.pop(k, None)
    saved[mod_key] = sys.modules.pop(mod_key, None)

    # Block imports by setting them to None (Python treats None as "not importable")
    for k in keys_to_block:
        sys.modules[k] = None  # type: ignore[assignment]

    try:
        with pytest.raises(ImportError, match="llama_index is required"):
            importlib.import_module("trulayer.instruments.llamaindex")
    finally:
        # Clean up blocked entries
        for k in keys_to_block:
            sys.modules.pop(k, None)
        sys.modules.pop(mod_key, None)
        # Restore fakes
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v
        # Reload with fakes restored
        importlib.reload(_li_mod)
