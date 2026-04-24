"""Replay captured traces from a JSONL file.

This module powers ``TRULAYER_MODE=replay`` and the public
:func:`trulayer.replay` helper. Given a JSONL file previously written by
:meth:`trulayer.local_batch.LocalBatchSender.flush_to_file`, it re-emits each
trace through an in-memory :class:`LocalBatchSender` as if the trace had just
been captured by the SDK.

The replay path is strictly non-throwing: malformed lines are logged via
``warnings.warn`` and skipped. A missing file is surfaced as a single warning
and produces an empty sender.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from trulayer.local_batch import LocalBatchSender


def replay(file: str) -> LocalBatchSender:
    """Read ``file`` as JSONL and re-emit each trace into a new
    :class:`LocalBatchSender`.

    Args:
        file: Path to a JSONL file where each line is a serialized trace
            payload (the shape produced by ``LocalBatchSender.flush_to_file``).

    Returns:
        A :class:`LocalBatchSender` populated with the replayed traces. Inspect
        ``sender.traces`` and ``sender.spans`` to assert on them.

    Notes:
        * Blank lines are ignored.
        * Lines that fail to parse as JSON or that aren't JSON objects emit a
          ``UserWarning`` and are skipped — never a crash.
        * A missing input file emits a warning and returns an empty sender.
    """
    sender = LocalBatchSender()
    sender.start()

    path = Path(file)
    if not path.exists():
        warnings.warn(
            f"trulayer: replay file not found: {file}",
            stacklevel=2,
        )
        return sender

    try:
        fh = path.open("r", encoding="utf-8")
    except OSError as exc:
        warnings.warn(
            f"trulayer: replay file could not be opened ({exc})",
            stacklevel=2,
        )
        return sender

    with fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.warn(
                    f"trulayer: skipping malformed replay line {lineno}: {exc}",
                    stacklevel=2,
                )
                continue
            if not isinstance(payload, dict):
                warnings.warn(
                    f"trulayer: skipping replay line {lineno}: "
                    f"expected JSON object, got {type(payload).__name__}",
                    stacklevel=2,
                )
                continue
            sender.enqueue(payload)

    return sender


__all__ = ["replay"]
