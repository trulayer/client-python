"""Tests for local/offline sandbox mode (TRU-81)."""

from __future__ import annotations

import os
from unittest import mock

import pytest

from trulayer.local_batch import LocalBatchSender
from trulayer.testing import assert_sender, create_test_client

# ---------------------------------------------------------------------------
# LocalBatchSender unit tests
# ---------------------------------------------------------------------------


class TestLocalBatchSender:
    def test_local_batch_sender_stores_traces(self) -> None:
        """send() (enqueue) stores traces without raising."""
        sender = LocalBatchSender()
        payload = {"id": "t1", "spans": []}
        sender.enqueue(payload)
        assert len(sender.traces) == 1
        assert sender.traces[0]["id"] == "t1"

    def test_local_batch_sender_traces_property(self) -> None:
        """traces property returns flat list across multiple batches."""
        sender = LocalBatchSender()
        sender.enqueue({"id": "t1", "spans": []})
        sender.enqueue({"id": "t2", "spans": []})
        assert len(sender.traces) == 2
        assert [t["id"] for t in sender.traces] == ["t1", "t2"]

    def test_local_batch_sender_spans_property(self) -> None:
        """spans property returns flat spans across all traces."""
        sender = LocalBatchSender()
        sender.enqueue({
            "id": "t1",
            "spans": [
                {"name": "s1", "span_type": "llm"},
                {"name": "s2", "span_type": "tool"},
            ],
        })
        sender.enqueue({
            "id": "t2",
            "spans": [{"name": "s3", "span_type": "default"}],
        })
        assert len(sender.spans) == 3
        assert [s["name"] for s in sender.spans] == ["s1", "s2", "s3"]

    def test_local_batch_sender_clear(self) -> None:
        """clear() empties all stored state."""
        sender = LocalBatchSender()
        sender.enqueue({"id": "t1", "spans": [{"name": "s1"}]})
        assert len(sender.traces) == 1
        sender.clear()
        assert len(sender.traces) == 0
        assert len(sender.spans) == 0


# ---------------------------------------------------------------------------
# create_test_client
# ---------------------------------------------------------------------------


class TestCreateTestClient:
    def test_create_test_client_returns_tuple(self) -> None:
        """Returns (TruLayerClient, LocalBatchSender) pair."""
        from trulayer.client import TruLayerClient

        client, sender = create_test_client(project_name="test-proj")
        assert isinstance(sender, LocalBatchSender)
        assert isinstance(client, TruLayerClient)

    def test_create_test_client_captures_spans(self) -> None:
        """After trace + flush, sender.spans has data."""
        client, sender = create_test_client(project_name="test-proj")
        with client.trace("my-trace") as t, t.span("step-1", "llm") as span:
            span.set_input("hello")
            span.set_output("world")
        client.flush()
        assert len(sender.traces) == 1
        assert len(sender.spans) == 1
        assert sender.spans[0]["name"] == "step-1"


# ---------------------------------------------------------------------------
# TRULAYER_MODE=local env auto-detection
# ---------------------------------------------------------------------------


class TestLocalModeEnv:
    def test_trulayer_mode_local_env(self) -> None:
        """TruLayerClient with TRULAYER_MODE=local uses LocalBatchSender."""
        from trulayer.client import TruLayerClient

        with mock.patch.dict(os.environ, {"TRULAYER_MODE": "local"}):
            with pytest.warns(match="LOCAL mode"):
                client = TruLayerClient(api_key="", project_name="ci-test")
            assert isinstance(client._batch, LocalBatchSender)


# ---------------------------------------------------------------------------
# SenderAssertions / assert_sender
# ---------------------------------------------------------------------------


class TestSenderAssertions:
    def test_assert_sender_has_trace_passes(self) -> None:
        """has_trace() passes when at least one trace is present."""
        sender = LocalBatchSender()
        sender.enqueue({"id": "t1", "spans": []})
        # Should not raise
        assert_sender(sender).has_trace()

    def test_assert_sender_span_count_raises(self) -> None:
        """span_count(n) raises AssertionError on mismatch."""
        sender = LocalBatchSender()
        sender.enqueue({"id": "t1", "spans": [{"name": "s1"}]})
        with pytest.raises(AssertionError, match="Expected 5 spans, got 1"):
            assert_sender(sender).span_count(5)

    def test_assert_sender_has_span_named_raises(self) -> None:
        """has_span_named raises when span not found."""
        sender = LocalBatchSender()
        sender.enqueue({"id": "t1", "spans": [{"name": "alpha"}]})
        with pytest.raises(AssertionError, match="Expected span named 'beta'"):
            assert_sender(sender).has_span_named("beta")
