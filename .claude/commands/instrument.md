---
description: Scaffold a new auto-instrumentation patch for a Python AI provider. Usage: /instrument <provider> — e.g. /instrument openai
---

Scaffold an auto-instrumentation module for a Python AI provider SDK. The argument is: $ARGUMENTS

Parse the argument as: <provider>
- provider: lowercase provider name (e.g. openai, anthropic, langchain, cohere)

Generate this file:

**`src/trulayer/instruments/<provider>.py`**

```python
"""Auto-instrumentation for <Provider>."""

from __future__ import annotations

import functools
import time
import warnings
from typing import Any

from trulayer.client import TruLayerClient
from trulayer.model import SpanData


_original_create = None
_patched = False


def patch(client: TruLayerClient) -> None:
    """Monkey-patch <Provider> to wrap completions in a TruLayer span."""
    global _original_create, _patched
    if _patched:
        return
    try:
        import <provider>  # noqa: PLC0415
    except ImportError:
        warnings.warn("trulayer: <provider> not installed, skipping instrumentation")
        return

    _original_create = <provider>.resources.completions.Completions.create

    @functools.wraps(_original_create)
    def _patched_create(self, *args: Any, **kwargs: Any) -> Any:
        start = time.monotonic()
        try:
            result = _original_create(self, *args, **kwargs)
        except Exception:
            # Never raise from SDK instrumentation
            return _original_create(self, *args, **kwargs)

        elapsed_ms = int((time.monotonic() - start) * 1000)

        try:
            span = SpanData(
                name="<provider>.completion",
                input=str(kwargs.get("messages", "")),
                output=str(result),
                latency_ms=elapsed_ms,
                model=kwargs.get("model", ""),
            )
            client._batch.enqueue(span)
        except Exception as e:
            warnings.warn(f"trulayer: failed to record span: {e}")

        return result

    <provider>.resources.completions.Completions.create = _patched_create
    _patched = True


def unpatch() -> None:
    """Restore original <Provider> method."""
    global _original_create, _patched
    if not _patched or _original_create is None:
        return
    import <provider>  # noqa: PLC0415
    <provider>.resources.completions.Completions.create = _original_create
    _patched = False
```

Rules:
- `patch()` must be idempotent — calling it twice should be a no-op
- `unpatch()` must fully restore the original method
- Never raise exceptions into user code — catch internally and `warnings.warn`
- Never log prompt/response content at INFO level — only at DEBUG
- Type annotations on all functions — `mypy --strict` must pass

After generating:
1. Export `patch` and `unpatch` from `src/trulayer/__init__.py` as `instrument_<provider>` / `uninstrument_<provider>`.
2. Add a unit test in `tests/unit/instruments/test_<provider>.py` that mocks the provider and verifies span enqueue is called.
3. Run `uv run mypy .` to verify type correctness.
