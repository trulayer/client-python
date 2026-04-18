# TruLayer AI — Python SDK

Python SDK for instrumenting AI applications and sending traces to TruLayer AI.

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

trulayer.init(api_key="tl_...", project="my-project")

with trulayer.trace(name="rag-pipeline") as trace:
    with trace.span(name="retrieve", type="retrieval") as span:
        docs = retrieve(query)
        span.set_output({"count": len(docs)})

    with trace.span(name="generate", type="llm") as span:
        result = llm.complete(prompt)
        span.set_output(result)
        span.set_metadata({"model": "gpt-4o", "tokens": 512})
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
    project="my-project",           # Project name
    environment="production",        # Environment tag
    endpoint="https://api.trulayer.ai",
    batch_size=50,                   # Events per batch
    flush_interval=2.0,              # Seconds between flushes
    timeout=5.0,                     # HTTP timeout
    debug=False,                     # Log trace data locally
)
```

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

## Linear Issues

- [TRU-8](https://linear.app/omnimoda/issue/TRU-8) — SDK Design
- [TRU-15](https://linear.app/omnimoda/issue/TRU-15) — Python SDK Implementation
