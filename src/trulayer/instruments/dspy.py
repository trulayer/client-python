"""DSPy instrumentation for TruLayer auto-instrumentation."""

from __future__ import annotations

import functools
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trulayer.trace import TraceContext

_original_forward: Any = None


def instrument_dspy(
    trace_ctx: TraceContext,
    *,
    capture_inputs: bool = True,
    capture_outputs: bool = True,
) -> None:
    """
    Globally patch ``dspy.Predict.forward()`` so every DSPy LM call creates
    a TruLayer span. Call once at startup; idempotent (checks ``_tl_patched`` flag).
    """
    global _original_forward
    try:
        import dspy  # noqa: PLC0415
    except ImportError as exc:
        warnings.warn(
            f"trulayer: dspy is required for DSPy instrumentation: {exc}",
            stacklevel=2,
        )
        return

    if getattr(dspy.Predict, "_tl_patched", False):
        return

    _original_forward = dspy.Predict.forward

    @functools.wraps(_original_forward)
    def instrumented_forward(self: Any, **kwargs: Any) -> Any:
        class_name = type(self).__name__
        span_name = f"dspy:{class_name}"
        with trace_ctx.span(span_name, span_type="llm") as span:
            if capture_inputs:
                span.set_input(str(kwargs))
            try:
                result = _original_forward(self, **kwargs)
                if capture_outputs:
                    span.set_output(str(result))
                return result
            except Exception as exc:
                span.set_metadata(error=str(exc))
                warnings.warn(
                    f"trulayer: error during DSPy forward instrumentation: {exc}",
                    stacklevel=2,
                )
                raise

    dspy.Predict.forward = instrumented_forward
    dspy.Predict._tl_patched = True


def uninstrument_dspy() -> None:
    """Restore original ``dspy.Predict.forward``."""
    global _original_forward
    try:
        import dspy  # noqa: PLC0415
    except ImportError:
        return

    if _original_forward is not None and getattr(dspy.Predict, "_tl_patched", False):
        dspy.Predict.forward = _original_forward
        dspy.Predict._tl_patched = False
        _original_forward = None
