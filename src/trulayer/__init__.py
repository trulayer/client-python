"""TruLayer AI — trace capture SDK for Python."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from trulayer.client import TruLayerClient
from trulayer.instruments.anthropic import instrument_anthropic, uninstrument_anthropic
from trulayer.instruments.langchain import instrument_langchain
from trulayer.instruments.openai import instrument_openai, uninstrument_openai
from trulayer.instruments.autogen import instrument_autogen
from trulayer.instruments.crewai import instrument_crewai
from trulayer.instruments.dspy import instrument_dspy, uninstrument_dspy
from trulayer.instruments.haystack import instrument_haystack
from trulayer.instruments.pydanticai import instrument_pydanticai
from trulayer.local_batch import LocalBatchSender
from trulayer.model import EventData, FeedbackData, SpanData, TraceData
from trulayer.testing import SenderAssertions, assert_sender, create_test_client
from trulayer.trace import TraceContext, current_trace

_global_client: TruLayerClient | None = None


def init(
    api_key: str,
    project_name: str | None = None,
    endpoint: str = "https://api.trulayer.ai",
    batch_size: int = 50,
    flush_interval: float = 2.0,
    sample_rate: float = 1.0,
    scrub_fn: Callable[[str], str] | None = None,
    metadata_validator: Callable[[dict[str, Any]], None] | None = None,
    project_id: str | None = None,  # deprecated alias
) -> TruLayerClient:
    """
    Initialize the global TruLayer client. Call once at application startup.

    `project_id` is a deprecated alias for `project_name` and will be removed
    in 0.3.x.

    Returns the client for explicit use or assignment.
    """
    global _global_client
    _global_client = TruLayerClient(
        api_key=api_key,
        project_name=project_name,
        project_id=project_id,
        endpoint=endpoint,
        batch_size=batch_size,
        flush_interval=flush_interval,
        sample_rate=sample_rate,
        scrub_fn=scrub_fn,
        metadata_validator=metadata_validator,
    )
    return _global_client


def get_client() -> TruLayerClient:
    if _global_client is None:
        raise RuntimeError("trulayer.init() has not been called")
    return _global_client


def instrument_llamaindex() -> TruLayerCallbackHandler:
    """
    Return a LlamaIndex callback handler that records query and LLM events as TruLayer spans.

    Pass the returned handler to a LlamaIndex ``CallbackManager``:

        from llama_index.core import Settings
        handler = trulayer.instrument_llamaindex()
        Settings.callback_manager.handlers.append(handler)

    Requires ``llama-index-core``. Install with: ``pip install llama-index-core``.
    """
    from trulayer.instruments.llamaindex import TruLayerCallbackHandler  # noqa: PLC0415

    return TruLayerCallbackHandler(get_client())


if TYPE_CHECKING:
    from trulayer.instruments.llamaindex import TruLayerCallbackHandler


__all__ = [
    "init",
    "get_client",
    "TruLayerClient",
    "TraceContext",
    "current_trace",
    "TraceData",
    "SpanData",
    "EventData",
    "FeedbackData",
    "LocalBatchSender",
    "create_test_client",
    "assert_sender",
    "SenderAssertions",
    "instrument_openai",
    "uninstrument_openai",
    "instrument_anthropic",
    "uninstrument_anthropic",
    "instrument_langchain",
    "instrument_llamaindex",
    "instrument_pydanticai",
    "instrument_crewai",
    "instrument_dspy",
    "uninstrument_dspy",
    "instrument_haystack",
    "instrument_autogen",
]
