"""Haystack v2 Pipeline instrumentation for TruLayer auto-instrumentation."""

from __future__ import annotations

import functools
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trulayer.trace import TraceContext


def instrument_haystack(
    pipeline: Any,
    trace_ctx: TraceContext,
    *,
    capture_inputs: bool = True,
    capture_outputs: bool = True,
) -> Any:
    """
    Wrap a Haystack Pipeline so ``.run()`` is recorded as a trace with one
    child span per component execution.

    Returns the same pipeline instance (mutated in place via monkey-patching).
    Never raises into user code — catches instrumentation setup errors and warns.
    """
    try:
        _patch_run(pipeline, trace_ctx, capture_inputs, capture_outputs)
        _patch_components(pipeline, trace_ctx)
    except Exception as exc:
        warnings.warn(
            f"trulayer: failed to instrument Haystack pipeline: {exc}",
            stacklevel=2,
        )
    return pipeline


def _patch_run(
    pipeline: Any,
    trace_ctx: TraceContext,
    capture_inputs: bool,
    capture_outputs: bool,
) -> None:
    original_run = pipeline.run

    @functools.wraps(original_run)
    def instrumented_run(*args: Any, **kwargs: Any) -> Any:
        with trace_ctx.span("haystack:pipeline", span_type="chain") as span:
            if capture_inputs:
                inputs = args[0] if args else kwargs.get("data", kwargs)
                span.set_input(str(inputs))
            try:
                result = original_run(*args, **kwargs)
                if capture_outputs:
                    span.set_output(str(result))
                return result
            except Exception as exc:
                span.set_metadata(error=str(exc))
                warnings.warn(
                    f"trulayer: error during Haystack pipeline run instrumentation: {exc}",
                    stacklevel=2,
                )
                raise

    pipeline.run = instrumented_run


def _patch_components(pipeline: Any, trace_ctx: TraceContext) -> None:
    """Wrap each component's run method with a child span."""
    graph = getattr(pipeline, "graph", None)
    if graph is None:
        return

    nodes = getattr(graph, "nodes", None)
    if nodes is None:
        return

    for component_name in nodes:
        node_data = graph.nodes[component_name]
        component = node_data.get("instance", None)
        if component is None:
            continue

        original_run = getattr(component, "run", None)
        if original_run is None:
            continue

        @functools.wraps(original_run)
        def instrumented_component_run(
            *args: Any,
            _orig: Any = original_run,
            _name: str = component_name,
            **kwargs: Any,
        ) -> Any:
            with trace_ctx.span(f"haystack:{_name}", span_type="other") as span:
                try:
                    result = _orig(*args, **kwargs)
                    return result
                except Exception as exc:
                    span.set_metadata(error=str(exc))
                    raise

        component.run = instrumented_component_run
