from __future__ import annotations


def new_id() -> str:
    """Generate a UUIDv7 string. Uses Python 3.14+ built-in or uuid7 backport."""
    try:
        import uuid as _uuid
        return str(_uuid.uuid7())  # type: ignore[attr-defined]  # Python 3.14+
    except AttributeError:  # pragma: no cover
        from uuid7 import uuid7  # pragma: no cover  # noqa: PLC0415
        return str(uuid7())  # pragma: no cover
