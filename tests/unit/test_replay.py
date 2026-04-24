"""Tests for TRULAYER_MODE=replay and flush_to_file (TRU-232)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import pytest

import trulayer
from trulayer.local_batch import LocalBatchSender


class TestFlushToFile:
    def test_writes_one_trace_per_line(self, tmp_path: Path) -> None:
        sender = LocalBatchSender()
        sender.enqueue({"id": "t1", "spans": [{"name": "s1"}]})
        sender.enqueue({"id": "t2", "spans": [{"name": "s2"}]})

        out = tmp_path / "traces.jsonl"
        sender.flush_to_file(str(out))

        lines = out.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["id"] == "t1"
        assert json.loads(lines[1])["id"] == "t2"

    def test_empty_sender_writes_empty_file(self, tmp_path: Path) -> None:
        sender = LocalBatchSender()
        out = tmp_path / "empty.jsonl"
        sender.flush_to_file(str(out))
        assert out.read_text(encoding="utf-8") == ""


class TestReplay:
    def test_replay_reads_jsonl(self, tmp_path: Path) -> None:
        src = tmp_path / "traces.jsonl"
        src.write_text(
            json.dumps({"id": "t1", "spans": [{"name": "s1"}]})
            + "\n"
            + json.dumps({"id": "t2", "spans": [{"name": "s2"}, {"name": "s3"}]})
            + "\n",
            encoding="utf-8",
        )

        sender = trulayer.replay(str(src))
        assert [t["id"] for t in sender.traces] == ["t1", "t2"]
        assert [s["name"] for s in sender.spans] == ["s1", "s2", "s3"]

    def test_replay_skips_blank_lines(self, tmp_path: Path) -> None:
        src = tmp_path / "traces.jsonl"
        src.write_text(
            "\n"
            + json.dumps({"id": "t1", "spans": []})
            + "\n\n"
            + json.dumps({"id": "t2", "spans": []})
            + "\n",
            encoding="utf-8",
        )
        sender = trulayer.replay(str(src))
        assert [t["id"] for t in sender.traces] == ["t1", "t2"]

    def test_replay_skips_malformed_lines_with_warning(self, tmp_path: Path) -> None:
        src = tmp_path / "traces.jsonl"
        src.write_text(
            json.dumps({"id": "good-1", "spans": []})
            + "\n"
            + "{not valid json\n"
            + json.dumps({"id": "good-2", "spans": []})
            + "\n",
            encoding="utf-8",
        )
        with pytest.warns(UserWarning, match="malformed replay line"):
            sender = trulayer.replay(str(src))
        assert [t["id"] for t in sender.traces] == ["good-1", "good-2"]

    def test_replay_skips_non_object_lines(self, tmp_path: Path) -> None:
        src = tmp_path / "traces.jsonl"
        src.write_text(
            json.dumps([1, 2, 3]) + "\n" + json.dumps({"id": "ok", "spans": []}) + "\n",
            encoding="utf-8",
        )
        with pytest.warns(UserWarning, match="expected JSON object"):
            sender = trulayer.replay(str(src))
        assert [t["id"] for t in sender.traces] == ["ok"]

    def test_replay_missing_file_warns_and_returns_empty(self, tmp_path: Path) -> None:
        missing = tmp_path / "does-not-exist.jsonl"
        with pytest.warns(UserWarning, match="replay file not found"):
            sender = trulayer.replay(str(missing))
        assert sender.traces == []

    def test_round_trip_capture_flush_replay(self, tmp_path: Path) -> None:
        """Capture traces → flush_to_file → replay → same spans."""
        client, sender = trulayer.create_test_client(project_name="rt-test")

        with client.trace("my-trace") as t:
            with t.span("step-1", "llm") as s1:
                s1.set_input("hello")
                s1.set_output("world")
            with t.span("step-2", "tool") as s2:
                s2.set_output("done")
        client.flush()

        assert len(sender.traces) == 1
        assert len(sender.spans) == 2
        original_span_names = sorted(s["name"] for s in sender.spans)

        out = tmp_path / "round_trip.jsonl"
        sender.flush_to_file(str(out))

        replayed = trulayer.replay(str(out))
        assert len(replayed.traces) == 1
        assert len(replayed.spans) == 2
        assert sorted(s["name"] for s in replayed.spans) == original_span_names
        # The replayed trace payload is byte-identical to what was captured.
        assert replayed.traces[0] == sender.traces[0]


class TestReplayEnvWiring:
    def test_trulayer_mode_replay_materializes_file(self, tmp_path: Path) -> None:
        src = tmp_path / "seed.jsonl"
        src.write_text(
            json.dumps({"id": "seed-1", "spans": [{"name": "seeded"}]}) + "\n",
            encoding="utf-8",
        )
        env = {
            "TRULAYER_MODE": "replay",
            "TRULAYER_REPLAY_FILE": str(src),
        }
        with mock.patch.dict(os.environ, env), pytest.warns(match="REPLAY mode"):
            client = trulayer.init(api_key="", project_name="replay-test")

        assert isinstance(client._batch, LocalBatchSender)
        assert [t["id"] for t in client._batch.traces] == ["seed-1"]

    def test_trulayer_mode_replay_without_file_warns(self) -> None:
        env = {"TRULAYER_MODE": "replay"}
        # Scrub TRULAYER_REPLAY_FILE if the harness env already set it.
        with mock.patch.dict(os.environ, env, clear=False):
            os.environ.pop("TRULAYER_REPLAY_FILE", None)
            with pytest.warns(match="TRULAYER_REPLAY_FILE"):
                client = trulayer.init(api_key="", project_name="no-file")
            assert isinstance(client._batch, LocalBatchSender)
            assert client._batch.traces == []
