from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass
class CapturedBatch:
    traces: list[dict[str, Any]]
    sent_at: str  # ISO timestamp


class LocalBatchSender:
    """In-memory trace collector — no network I/O, no API key required.

    Drop-in replacement for :class:`~trulayer.batch.BatchSender` when running
    in local / offline mode (``TRULAYER_MODE=local``).  All traces are kept in
    memory so tests and CI jobs can assert on what would have been sent without
    hitting the real API.
    """

    def __init__(self) -> None:
        self._batches: list[CapturedBatch] = []

    # -- BatchSender-compatible interface ------------------------------------

    def start(self) -> None:
        """No-op — nothing to start in local mode."""

    def enqueue(self, item: dict[str, Any]) -> None:
        """Store a single trace payload without sending. Never raises."""
        self._batches.append(
            CapturedBatch(
                traces=[item],
                sent_at=datetime.now(UTC).isoformat(),
            )
        )
        if os.environ.get("TRULAYER_LOCAL_VERBOSE") == "1":
            print(
                f"[trulayer:local] trace {item.get('trace_id', item.get('id'))} "
                f"— {len(item.get('spans', []))} span(s)"
            )

    def shutdown(self, timeout: float = 5.0) -> None:  # noqa: ARG002
        """No-op — nothing to flush in local mode."""

    # -- Inspection helpers --------------------------------------------------

    @property
    def traces(self) -> list[dict[str, Any]]:
        """All captured traces across all batches (flat)."""
        return [t for b in self._batches for t in b.traces]

    @property
    def spans(self) -> list[dict[str, Any]]:
        """All captured spans across all batches (flat)."""
        return [s for t in self.traces for s in t.get("spans", [])]

    def clear(self) -> None:
        self._batches.clear()
