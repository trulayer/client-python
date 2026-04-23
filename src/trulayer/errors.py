"""Typed exceptions raised by the TruLayer SDK.

The SDK is designed to never raise into user code at runtime — batch sender
failures are surfaced through :mod:`warnings`. These exception types are
provided so callers who want to observe and react to specific, non-retryable
failure modes (e.g. a misconfigured API key) can do so.
"""

from __future__ import annotations

from typing import Any, Literal

InvalidAPIKeyCode = Literal["invalid_api_key", "api_key_expired"]

_INVALID_API_KEY_MESSAGE = "API key is invalid or has expired — check your configuration."


class TruLayerError(Exception):
    """Base class for all TruLayer SDK exceptions."""


class InvalidAPIKeyError(TruLayerError):
    """Raised when the TruLayer API rejects a request with HTTP 401 and an
    error code of ``invalid_api_key`` or ``api_key_expired``.

    These are permanent configuration errors — retrying with the same
    credentials has no chance of succeeding, so the SDK halts pending and
    future requests for the lifetime of the client instance.

    Recommended handling: catch during startup smoke requests, log the
    failure, and surface an actionable message to the operator.
    """

    def __init__(self, code: InvalidAPIKeyCode) -> None:
        super().__init__(_INVALID_API_KEY_MESSAGE)
        self.code: InvalidAPIKeyCode = code


def parse_invalid_api_key_payload(body: Any) -> InvalidAPIKeyCode | None:
    """Return the matched non-retryable code if ``body`` represents an invalid
    or expired API key response, else ``None``.

    Accepts either an ``error`` or ``code`` field for forward compatibility
    with future API error schemas.
    """
    if not isinstance(body, dict):
        return None
    raw = body.get("error") or body.get("code")
    if raw == "invalid_api_key":
        return "invalid_api_key"
    if raw == "api_key_expired":
        return "api_key_expired"
    return None


__all__ = [
    "TruLayerError",
    "InvalidAPIKeyError",
    "InvalidAPIKeyCode",
    "parse_invalid_api_key_payload",
]
