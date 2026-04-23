# TruLayer AI — Python SDK

> **Status: Alpha.** APIs are pre-`1.0.0` and may change between minor releases.
> Pin a specific version in production until `1.0.0` ships.

Python SDK for instrumenting AI applications and sending traces to TruLayer AI.

- Documentation: https://docs.trulayer.ai
- Source: https://github.com/trulayer/client-python
- Issues: https://github.com/trulayer/client-python/issues

## Installation

```bash
pip install trulayer
# or
uv add trulayer
```

## Quick Start

```python
import trulayer
from openai import OpenAI

trulayer.init(api_key="tl_...")

client = OpenAI()

# Auto-instrumented — traces sent automatically
with trulayer.trace(name="my-agent", session_id="user-123"):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}]
    )
```

## Manual Instrumentation

```python
import trulayer

trulayer.init(api_key="tl_...", project_name="my-project")

with trulayer.trace(
    name="rag-pipeline",
    external_id="req-42",  # link to your own request id for idempotent ingest
) as trace:
    trace.set_model("gpt-4o")  # rolled-up model for the trace

    with trace.span(name="retrieve", type="retrieval") as span:
        docs = retrieve(query)
        span.set_output({"count": len(docs)})

    with trace.span(name="generate", type="llm") as span:
        result = llm.complete(prompt)
        span.set_output(result)
        span.set_metadata({"model": "gpt-4o", "tokens": 512})

    trace.set_cost(0.0042)  # optional rolled-up cost in USD
    # latency_ms is auto-derived from start to end of the trace block
```

## Auto-Instrumentation

```python
# Patch OpenAI client globally
trulayer.instrument_openai()

# Patch Anthropic client
trulayer.instrument_anthropic()

# Patch LangChain
trulayer.instrument_langchain()
```

## Async Support

```python
async with trulayer.atrace(name="async-agent") as trace:
    async with trace.aspan(name="fetch") as span:
        result = await async_llm_call()
```

## Configuration

```python
trulayer.init(
    api_key="tl_...",
    project_name="my-project",
    endpoint="https://api.trulayer.ai",
    batch_size=50,                   # Events per batch
    flush_interval=2.0,              # Seconds between flushes
    timeout=5.0,                     # HTTP timeout
    debug=False,                     # Log trace data locally
)
```

## Error Handling

The SDK is fire-and-forget: transient HTTP failures are retried with exponential backoff (up to 3 attempts) and eventually surfaced via `warnings.warn`. User code is never interrupted by network errors.

One failure mode is **non-retryable** and surfaced as a typed exception: if the TruLayer API responds with HTTP 401 and an error code of `invalid_api_key` or `api_key_expired`, the SDK:

- Raises `InvalidAPIKeyError` internally (no retries, no backoff).
- Drops all queued traces and rejects subsequent `enqueue` calls on that client instance.
- Emits a single `UserWarning` identifying the failure.

These are permanent configuration errors — retrying cannot succeed, so the SDK halts to avoid wasting requests.

```python
import sys
import trulayer
from trulayer import InvalidAPIKeyError

tl = trulayer.init(api_key="tl_...", project_name="my-project")

# Recommended: fail fast at startup with a lightweight probe trace.
try:
    with tl.trace("startup-probe"):
        pass
    tl.flush()
    if tl._batch.fatal_error is not None:
        raise tl._batch.fatal_error
except InvalidAPIKeyError as err:
    print(err, file=sys.stderr)  # "API key is invalid or has expired — check your configuration."
    sys.exit(1)
```

`InvalidAPIKeyError` exposes a `code` attribute (`"invalid_api_key"` or `"api_key_expired"`) for programmatic handling and subclasses `TruLayerError` (also exported).

## Tech Stack

- Python 3.11+
- `httpx` for async HTTP
- `pydantic` for data validation
- Supports sync and async contexts
- Compatible with: OpenAI, Anthropic, LangChain, LlamaIndex

## Development

```bash
uv sync --dev
uv run pytest                    # Tests
uv run pytest --cov              # Coverage (target: >90%)
uv run ruff check .              # Lint
uv run ruff format .             # Format
uv run mypy .                    # Type check
```

## Links

- [Documentation](https://docs.trulayer.ai)
- [API Reference](https://docs.trulayer.ai/api-reference)

