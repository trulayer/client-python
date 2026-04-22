"""Tests for the DSPy instrumentation."""

from __future__ import annotations

import sys
import types
from typing import Any, cast
from unittest.mock import MagicMock

from trulayer.instruments.dspy import instrument_dspy, uninstrument_dspy
from trulayer.trace import TraceContext

# ---------------------------------------------------------------------------
# Fake dspy module injected into sys.modules for testing
# ---------------------------------------------------------------------------


def _make_fake_dspy() -> types.ModuleType:
    mod = types.ModuleType("dspy")

    class Predict:
        def forward(self, **kwargs: Any) -> dict[str, Any]:
            return {"answer": "42"}

    mod.Predict = Predict  # type: ignore[attr-defined]
    return mod


def _install_fake_dspy() -> types.ModuleType:
    mod = _make_fake_dspy()
    sys.modules["dspy"] = mod
    return mod


def _cleanup_fake_dspy() -> None:
    sys.modules.pop("dspy", None)


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
# 1. instrument_dspy patches Predict.forward
# ---------------------------------------------------------------------------


def test_instrument_patches_forward() -> None:
    mod = _install_fake_dspy()
    try:
        client = _make_client()
        original = mod.Predict.forward
        with TraceContext(client, name="trace") as ctx:
            instrument_dspy(ctx)
        assert mod.Predict.forward is not original
        assert getattr(mod.Predict, "_tl_patched", False) is True
    finally:
        uninstrument_dspy()
        _cleanup_fake_dspy()


# ---------------------------------------------------------------------------
# 2. forward creates span with input kwargs
# ---------------------------------------------------------------------------


def test_forward_creates_span_with_input() -> None:
    mod = _install_fake_dspy()
    try:
        client = _make_client()
        with TraceContext(client, name="trace") as ctx:
            instrument_dspy(ctx)
            p = mod.Predict()
            result = p.forward(question="What is 6*7?")

        assert result == {"answer": "42"}
        spans = _get_spans(client)
        llm_spans = [s for s in spans if s["span_type"] == "llm"]
        assert len(llm_spans) == 1
        assert "Predict" in llm_spans[0]["name"]
        assert "What is 6*7?" in llm_spans[0]["input"]
    finally:
        uninstrument_dspy()
        _cleanup_fake_dspy()


# ---------------------------------------------------------------------------
# 3. forward records output
# ---------------------------------------------------------------------------


def test_forward_records_output() -> None:
    mod = _install_fake_dspy()
    try:
        client = _make_client()
        with TraceContext(client, name="trace") as ctx:
            instrument_dspy(ctx)
            p = mod.Predict()
            p.forward(question="test")

        spans = _get_spans(client)
        llm_spans = [s for s in spans if s["span_type"] == "llm"]
        assert len(llm_spans) == 1
        assert "42" in llm_spans[0]["output"]
    finally:
        uninstrument_dspy()
        _cleanup_fake_dspy()


# ---------------------------------------------------------------------------
# 4. uninstrument_dspy restores original
# ---------------------------------------------------------------------------


def test_uninstrument_restores_original() -> None:
    mod = _install_fake_dspy()
    try:
        original = mod.Predict.forward
        client = _make_client()
        with TraceContext(client, name="trace") as ctx:
            instrument_dspy(ctx)
        assert mod.Predict.forward is not original
        uninstrument_dspy()
        assert mod.Predict.forward is original
        assert getattr(mod.Predict, "_tl_patched", True) is False
    finally:
        _cleanup_fake_dspy()


# ---------------------------------------------------------------------------
# 5. double-call is idempotent (no double-wrapping)
# ---------------------------------------------------------------------------


def test_double_instrument_is_idempotent() -> None:
    mod = _install_fake_dspy()
    try:
        client = _make_client()
        with TraceContext(client, name="trace") as ctx:
            instrument_dspy(ctx)
            first_patched = mod.Predict.forward
            instrument_dspy(ctx)
            second_patched = mod.Predict.forward
        assert first_patched is second_patched
    finally:
        uninstrument_dspy()
        _cleanup_fake_dspy()


# ---------------------------------------------------------------------------
# 6. instrument_dspy with dspy not installed warns and returns
# ---------------------------------------------------------------------------


def test_instrument_dspy_no_package_warns() -> None:
    import warnings

    _cleanup_fake_dspy()
    client = _make_client()
    with TraceContext(client, name="trace") as ctx:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Ensure dspy is not importable
            sys.modules["dspy"] = None  # type: ignore[assignment]
            try:
                instrument_dspy(ctx)
            finally:
                sys.modules.pop("dspy", None)
        assert any("dspy" in str(warning.message).lower() for warning in w)


# ---------------------------------------------------------------------------
# 7. uninstrument_dspy with dspy not installed is a noop
# ---------------------------------------------------------------------------


def test_uninstrument_dspy_no_package() -> None:
    """uninstrument_dspy should not raise when dspy is not installed."""
    _cleanup_fake_dspy()
    sys.modules["dspy"] = None  # type: ignore[assignment]
    try:
        uninstrument_dspy()  # should not raise
    finally:
        sys.modules.pop("dspy", None)


# ---------------------------------------------------------------------------
# 8. forward raises → exception propagates + span records error + warns
# ---------------------------------------------------------------------------


def test_forward_exception_propagates_and_records_error() -> None:
    import warnings

    mod = _install_fake_dspy()

    class FailingPredict(mod.Predict):  # type: ignore[misc,name-defined]
        def forward(self, **kwargs: Any) -> Any:
            raise ValueError("forward failed")

    # Override Predict with the failing subclass
    mod.Predict = FailingPredict  # type: ignore[attr-defined]

    try:
        client = _make_client()
        with TraceContext(client, name="trace") as ctx:
            instrument_dspy(ctx)
            p = mod.Predict()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                import pytest

                with pytest.raises(ValueError, match="forward failed"):
                    p.forward(question="boom?")
            assert any("error during DSPy" in str(warning.message) for warning in w)

        spans = _get_spans(client)
        llm_spans = [s for s in spans if s["span_type"] == "llm"]
        assert len(llm_spans) == 1
        assert llm_spans[0]["metadata"]["error"] == "forward failed"
        assert llm_spans[0]["error"] is True
    finally:
        uninstrument_dspy()
        _cleanup_fake_dspy()
