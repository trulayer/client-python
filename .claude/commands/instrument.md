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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trulayer.client import TruLayerClient

_original_create: Any = None
_patched = False


def instrument_<provider>(client: TruLayerClient) -> None:
    """
    Monkey-patch <provider> to wrap completions in a TruLayer span.
    Idempotent — calling twice is a no-op.
    """
    global _original_create, _patched
    if _patched:
        return
    try:
        import <provider>  # noqa: PLC0415
    except ImportError:
        warnings.warn("trulayer: <provider> not installed, skipping instrumentation", stacklevel=2)
        return

    _original_create = <provider>.resources.chat.completions.Completions.create

    @functools.wraps(_original_create)
    def _patched_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.monotonic()
        result = _original_create(self, *args, **kwargs)
        _record_span(client, kwargs, result, time.monotonic() - start)
        return result

    <provider>.resources.chat.completions.Completions.create = _patched_create
    _patched = True


def uninstrument_<provider>() -> None:
    """Restore original <Provider> method. Idempotent."""
    global _original_create, _patched
    if not _patched or _original_create is None:
        return
    try:
        import <provider>  # noqa: PLC0415
        <provider>.resources.chat.completions.Completions.create = _original_create
    except ImportError:
        pass
    _patched = False


def _record_span(
    client: TruLayerClient,
    kwargs: dict[str, Any],
    result: Any,
    elapsed: float,
) -> None:
    try:
        from trulayer.trace import current_trace  # noqa: PLC0415

        trace = current_trace()
        if trace is None:
            return

        output = ""
        prompt_tokens: int | None = None
        completion_tokens: int | None = None

        try:
            output = result.choices[0].message.content or ""
            if result.usage:
                prompt_tokens = result.usage.prompt_tokens
                completion_tokens = result.usage.completion_tokens
        except Exception:
            pass

        messages = kwargs.get("messages", [])
        input_text = messages[-1].get("content", "") if messages else ""

        with trace.span("<provider>.chat", span_type="llm") as span:
            span.set_input(str(input_text))
            span.set_output(output)
            span.set_model(kwargs.get("model", ""))
            span.set_tokens(prompt=prompt_tokens, completion=completion_tokens)
            span._data.latency_ms = int(elapsed * 1000)
    except Exception as e:
        warnings.warn(f"trulayer: failed to record span: {e}", stacklevel=2)
```

Rules:
- `instrument_<provider>()` must be idempotent — calling twice is a no-op
- `uninstrument_<provider>()` must fully restore the original method
- Record spans via `current_trace()` + `with trace.span(name, span_type=...) as span:` — never call `client._batch.enqueue()` or instantiate `SpanData` directly
- `_record_span` must never raise into user code — wrap entirely in try/except with `warnings.warn`
- Never log prompt/response content at INFO level — only at DEBUG
- Type annotations on all functions — `mypy --strict` must pass

After generating:
1. Export `instrument_<provider>` and `uninstrument_<provider>` from `src/trulayer/__init__.py`.
2. Add a unit test in `tests/unit/instruments/test_<provider>.py` that mocks the provider and verifies a span is recorded on the active trace.
3. Run `uv run mypy .` to verify type correctness.
