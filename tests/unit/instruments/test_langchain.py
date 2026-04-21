"""Tests for the LangChain callback handler."""

from __future__ import annotations

import sys
import warnings
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from trulayer.instruments.langchain import (
    _extract_chat_input,
    _extract_model,
    _on_chat_model_start,
    _on_llm_end,
    _on_llm_error,
    _on_llm_start,
)
from trulayer.trace import TraceContext


def _make_client(project_id: str = "proj-1") -> MagicMock:
    client = MagicMock()
    client._project_id = project_id
    client._batch = MagicMock()
    client._scrub_fn = None
    client._sample_rate = 1.0
    client._metadata_validator = None
    return client


def _make_handler() -> Any:
    """Build a minimal handler-like object with the mixin state."""
    h = SimpleNamespace()
    h._tl_client = _make_client()
    h._tl_starts = {}
    return h


def _make_llm_result(
    text: str = "answer",
    prompt_tokens: int = 8,
    completion_tokens: int = 4,
) -> Any:
    gen = SimpleNamespace(text=text, message=None)
    return SimpleNamespace(
        generations=[[gen]],
        llm_output={
            "token_usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
        },
    )


def _make_chat_llm_result(text: str = "chat answer") -> Any:
    msg = SimpleNamespace(content=text)
    gen = SimpleNamespace(text=None, message=msg)
    return SimpleNamespace(
        generations=[[gen]],
        llm_output={"token_usage": {"prompt_tokens": 5, "completion_tokens": 3}},
    )


# ---------------------------------------------------------------------------
# _extract_model
# ---------------------------------------------------------------------------


def test_extract_model_from_serialized_model_name() -> None:
    serialized = {"kwargs": {"model_name": "gpt-4o"}}
    assert _extract_model(serialized, {}) == "gpt-4o"


def test_extract_model_from_serialized_model() -> None:
    serialized = {"kwargs": {"model": "claude-3"}}
    assert _extract_model(serialized, {}) == "claude-3"


def test_extract_model_from_invocation_params() -> None:
    serialized: dict[str, Any] = {}
    assert _extract_model(serialized, {"invocation_params": {"model_name": "gpt-3.5"}}) == "gpt-3.5"


def test_extract_model_empty() -> None:
    assert _extract_model({}, {}) == ""


# ---------------------------------------------------------------------------
# _extract_chat_input
# ---------------------------------------------------------------------------


def test_extract_chat_input_string_content() -> None:
    msg = SimpleNamespace(content="hello from user")
    assert _extract_chat_input([[msg]]) == "hello from user"


def test_extract_chat_input_multimodal() -> None:
    msg = SimpleNamespace(content=[{"type": "text", "text": "multimodal input"}])
    assert _extract_chat_input([[msg]]) == "multimodal input"


def test_extract_chat_input_empty() -> None:
    assert _extract_chat_input([]) == ""


def test_extract_chat_input_non_string_non_list_content() -> None:
    msg = SimpleNamespace(content=42)  # neither str nor list
    assert _extract_chat_input([[msg]]) == ""


def test_extract_chat_input_exception_returns_empty() -> None:
    class _Boom:
        @property
        def content(self) -> str:
            raise RuntimeError("boom")

    assert _extract_chat_input([[_Boom()]]) == ""


# ---------------------------------------------------------------------------
# on_llm_start / on_llm_end lifecycle
# ---------------------------------------------------------------------------


def test_on_llm_start_stores_entry() -> None:
    h = _make_handler()
    run_id = uuid4()
    _on_llm_start(h, {"kwargs": {"model_name": "gpt-4o"}}, ["hi there"], run_id=run_id)
    assert run_id in h._tl_starts
    _, model, input_text = h._tl_starts[run_id]
    assert model == "gpt-4o"
    assert input_text == "hi there"


def test_on_llm_end_records_span_inside_trace() -> None:
    client = _make_client()
    h = _make_handler()
    h._tl_client = client
    run_id = uuid4()

    with TraceContext(client, name="test"):
        _on_llm_start(h, {"kwargs": {"model_name": "gpt-4"}}, ["what is 2+2?"], run_id=run_id)
        _on_llm_end(h, _make_llm_result("4"), run_id=run_id)

    payload = client._batch.enqueue.call_args[0][0]
    span = payload["spans"][0]
    assert span["name"] == "langchain.llm"
    assert span["span_type"] == "llm"
    assert span["input"] == "what is 2+2?"
    assert span["output"] == "4"
    assert span["model"] == "gpt-4"
    assert span["prompt_tokens"] == 8
    assert span["completion_tokens"] == 4


def test_on_llm_end_outside_trace_is_noop() -> None:
    h = _make_handler()
    client = h._tl_client
    run_id = uuid4()
    _on_llm_start(h, {}, ["prompt"], run_id=run_id)
    _on_llm_end(h, _make_llm_result(), run_id=run_id)
    client._batch.enqueue.assert_not_called()


def test_on_llm_end_malformed_result_records_empty_output() -> None:
    """If result parsing raises, on_llm_end still records a span with empty output."""
    client = _make_client()
    h = _make_handler()
    h._tl_client = client
    run_id = uuid4()

    class _BadResult:
        @property
        def generations(self) -> object:
            raise AttributeError("no generations")

        llm_output: dict[str, Any] = {}

    with TraceContext(client, name="t"):
        _on_llm_start(h, {}, ["prompt"], run_id=run_id)
        _on_llm_end(h, _BadResult(), run_id=run_id)

    payload = client._batch.enqueue.call_args[0][0]
    assert payload["spans"][0]["output"] == ""


def test_on_llm_end_unknown_run_id_is_noop() -> None:
    h = _make_handler()
    _on_llm_end(h, _make_llm_result(), run_id=uuid4())
    # No crash expected


def test_on_llm_end_cleans_up_starts_entry() -> None:
    h = _make_handler()
    run_id = uuid4()
    _on_llm_start(h, {}, ["hi"], run_id=run_id)
    _on_llm_end(h, _make_llm_result(), run_id=run_id)
    assert run_id not in h._tl_starts


# ---------------------------------------------------------------------------
# on_chat_model_start
# ---------------------------------------------------------------------------


def test_on_chat_model_start_records_span() -> None:
    client = _make_client()
    h = _make_handler()
    h._tl_client = client
    run_id = uuid4()

    msg = SimpleNamespace(content="chat input")
    with TraceContext(client, name="chat-test"):
        _on_chat_model_start(h, {"kwargs": {"model_name": "claude-3"}}, [[msg]], run_id=run_id)
        _on_llm_end(h, _make_chat_llm_result("chat output"), run_id=run_id)

    payload = client._batch.enqueue.call_args[0][0]
    span = payload["spans"][0]
    assert span["input"] == "chat input"
    assert span["output"] == "chat output"
    assert span["model"] == "claude-3"


# ---------------------------------------------------------------------------
# on_llm_error
# ---------------------------------------------------------------------------


def test_on_llm_error_removes_entry() -> None:
    h = _make_handler()
    run_id = uuid4()
    _on_llm_start(h, {}, ["hi"], run_id=run_id)
    _on_llm_error(h, ValueError("boom"), run_id=run_id)
    assert run_id not in h._tl_starts


# ---------------------------------------------------------------------------
# instrument_langchain — missing dependency
# ---------------------------------------------------------------------------


def test_instrument_langchain_creates_handler() -> None:
    """instrument_langchain returns a handler subclassing the mocked BaseCallbackHandler."""
    from trulayer.instruments.langchain import instrument_langchain  # noqa: PLC0415

    class FakeBase:
        pass

    client = _make_client()
    with pytest.MonkeyPatch().context() as mp:
        fake_callbacks = MagicMock()
        fake_callbacks.BaseCallbackHandler = FakeBase
        fake_lc_core = MagicMock()
        fake_lc_core.callbacks = fake_callbacks
        mp.setitem(sys.modules, "langchain_core", fake_lc_core)
        mp.setitem(sys.modules, "langchain_core.callbacks", fake_callbacks)
        handler = instrument_langchain(client)

    assert isinstance(handler, FakeBase)
    assert handler._tl_client is client  # type: ignore[attr-defined]


def test_instrument_langchain_missing_dep_warns_and_raises() -> None:
    with pytest.raises(ImportError), warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        with (
            # Hide both possible import paths
            pytest.MonkeyPatch().context() as mp,
        ):
            mp.setitem(sys.modules, "langchain_core", None)
            mp.setitem(sys.modules, "langchain_core.callbacks", None)
            mp.setitem(sys.modules, "langchain", None)
            mp.setitem(sys.modules, "langchain.callbacks", None)

            import importlib  # noqa: PLC0415

            from trulayer.instruments import langchain as lc_mod  # noqa: PLC0415

            importlib.reload(lc_mod)
            lc_mod.instrument_langchain(_make_client())
