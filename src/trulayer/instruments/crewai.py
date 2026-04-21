"""CrewAI Crew instrumentation for TruLayer auto-instrumentation."""

from __future__ import annotations

import functools
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trulayer.trace import TraceContext


def instrument_crewai(
    crew: Any,
    trace_ctx: TraceContext,
    *,
    capture_inputs: bool = True,
    capture_outputs: bool = True,
) -> Any:
    """
    Wrap a CrewAI Crew so ``.kickoff()`` is recorded as a trace with one
    child span per agent task execution.

    Returns the same crew instance (mutated in place via monkey-patching).
    Never raises into user code — catches instrumentation setup errors and warns.
    """
    try:
        _patch_kickoff(crew, trace_ctx, capture_inputs, capture_outputs)
        _patch_agents(crew, trace_ctx)
    except Exception as exc:
        warnings.warn(
            f"trulayer: failed to instrument CrewAI crew: {exc}",
            stacklevel=2,
        )
    return crew


def _patch_kickoff(
    crew: Any,
    trace_ctx: TraceContext,
    capture_inputs: bool,
    capture_outputs: bool,
) -> None:
    original_kickoff = crew.kickoff

    @functools.wraps(original_kickoff)
    def instrumented_kickoff(*args: Any, **kwargs: Any) -> Any:
        with trace_ctx.span("crewai:kickoff", span_type="agent") as span:
            if capture_inputs:
                inputs = getattr(crew, "inputs", None)
                if inputs is not None:
                    span.set_input(str(inputs))
                elif kwargs:
                    span.set_input(str(kwargs))
            try:
                result = original_kickoff(*args, **kwargs)
                if capture_outputs:
                    raw = getattr(result, "raw", None)
                    if raw is not None:
                        span.set_output(str(raw))
                    else:
                        span.set_output(str(result))
                return result
            except Exception as exc:
                span.set_metadata(error=str(exc))
                warnings.warn(
                    f"trulayer: error during CrewAI kickoff instrumentation: {exc}",
                    stacklevel=2,
                )
                raise

    crew.kickoff = instrumented_kickoff


def _patch_agents(crew: Any, trace_ctx: TraceContext) -> None:
    """Wrap each agent's execute_task method with a child span."""
    agents = getattr(crew, "agents", None)
    if not agents:
        return

    for agent in agents:
        original_execute = getattr(agent, "execute_task", None)
        if original_execute is None:
            continue

        @functools.wraps(original_execute)
        def instrumented_execute(
            *args: Any,
            _orig: Any = original_execute,
            _agent: Any = agent,
            **kwargs: Any,
        ) -> Any:
            task = args[0] if args else kwargs.get("task")
            desc = ""
            if task is not None:
                desc = str(getattr(task, "description", ""))[:40]
            span_name = f"crewai:task:{desc}" if desc else "crewai:task"
            with trace_ctx.span(span_name, span_type="agent") as span:
                try:
                    result = _orig(*args, **kwargs)
                    return result
                except Exception as exc:
                    span.set_metadata(error=str(exc))
                    raise

        agent.execute_task = instrumented_execute
