---
name: trulayer-python-sdk
description: Use this skill when writing, debugging, or integrating the trulayer Python package — including init, trace context managers, span context managers, feedback, auto-instrumentation of OpenAI/Anthropic/LangChain/LlamaIndex/PydanticAI/CrewAI/DSPy/Haystack/AutoGen, redaction, and batch/flush behavior.
---

# TruLayer Python SDK (`trulayer`)

Authoritative reference for the `trulayer` pip package. Every example below mirrors the real public surface — do not invent kwargs or methods that are not listed here.

## Install

```bash
pip install trulayer
# or
uv add trulayer
```

Requires **Python 3.11+**.

Optional provider extras are NOT shipped — install the providers you use directly (`pip install openai anthropic langchain-core llama-index-core` etc.). Instrumentation hooks are no-ops with a warning if the underlying library is missing.

## Init and config

```python
import trulayer

trulayer.init(
    api_key="tl_live_...",
    project_name="my-project",
)
```

`init()` constructs a global `TruLayerClient`, registers an `atexit` flush, and returns the client.

Full signature:

```python
def init(
    api_key: str,
    project_name: str | None = None,
    endpoint: str = "https://api.trulayer.ai",
    batch_size: int = 50,
    flush_interval: float = 2.0,
    sample_rate: float = 1.0,                 # 0.0–1.0, sampled per-trace
    scrub_fn: Callable[[str], str] | None = None,   # applied to input/output/error_message
    metadata_validator: Callable[[dict], None] | None = None,
    redactor: Redactor | None = None,         # see Redaction section
    project_id: str | None = None,            # DEPRECATED alias, removed in 0.3.x
) -> TruLayerClient
```

Multi-project / explicit client:

```python
from trulayer import TruLayerClient

client = TruLayerClient(api_key="tl_...", project_name="proj-a")
```

`trulayer.get_client()` returns the global client (raises `RuntimeError` if `init()` was never called).

### Local mode (CI / offline)

Set `TRULAYER_MODE=local` in the environment. The client uses an in-memory `LocalBatchSender` instead of HTTP, no API key validation, and `project_name` defaults to `"local"`. A warning is emitted on construction.

## Tracing

Traces and spans are **context managers** — exiting the `with` block is what finalizes the data and queues it onto the batch sender. Forgetting `with` means nothing ships.

### Single trace, single span

```python
import trulayer

trulayer.init(api_key="tl_...", project_name="my-project")
client = trulayer.get_client()

with client.trace("checkout.summarize") as trace:
    trace.set_input("user query")
    with trace.span("openai.call", span_type="llm") as span:
        span.set_input("user query")
        # ... call your LLM ...
        span.set_output("model response")
        span.set_model("gpt-4o")
        span.set_tokens(prompt=120, completion=64)
    trace.set_output("final answer")
```

### Trace API

```python
client.trace(
    name: str | None = None,
    session_id: str | None = None,
    external_id: str | None = None,
    tags: list[str] | None = None,           # e.g. ["env:prod", "model:gpt-4o"]
    metadata: dict[str, Any] | None = None,
)
```

Methods on the trace object inside the `with`:

- `trace.set_input(value: str)`
- `trace.set_output(value: str)`
- `trace.set_model(model: str)`
- `trace.set_cost(cost: float)`
- `trace.set_metadata(**kwargs)`
- `trace.add_tag(tag: str)`
- `trace.span(name: str, span_type: str = "default") -> SpanContext`

### Span API

```python
with trace.span("retriever", span_type="retrieval") as span:
    span.set_input("query")
    span.set_output("docs")
    span.set_model("bge-m3")
    span.set_tokens(prompt=10, completion=0)
    span.set_metadata(top_k=5)
```

Common `span_type` values: `"llm"`, `"retrieval"`, `"tool"`, `"default"`.

### Async usage

Both `TraceContext` and `SpanContext` implement `__aenter__` / `__aexit__`, so `async with` works identically:

```python
async with client.trace("api.handler") as trace:
    async with trace.span("openai.call", span_type="llm") as span:
        ...
```

### Current trace lookup

```python
from trulayer import current_trace

t = current_trace()  # the active TraceContext in this contextvar, or None
```

This is what auto-instrumentation hooks use under the hood.

## Auto-instrumentation

### OpenAI

```python
from openai import OpenAI
import trulayer

trulayer.init(api_key="tl_...", project_name="my-project")
trulayer.instrument_openai(trulayer.get_client())

client = OpenAI()
with trulayer.get_client().trace("answer-question"):
    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
    )
```

Notes:

- Signature: `instrument_openai(client: TruLayerClient) -> None`. Pass the **TruLayer** client, not the OpenAI client.
- Patches `openai.resources.chat.completions.Completions.create` and the async equivalent **in place** (no wrapped object returned).
- Idempotent — second call is a no-op.
- Streaming responses (`stream=True`, sync and async) are wrapped — the span is finalized when the iterator is exhausted or raises.
- Spans only attach when there is an active `trace.trace(...)` context; bare OpenAI calls are passed through untouched.
- `trulayer.uninstrument_openai()` restores the originals (use in test teardown).

### Anthropic

```python
import anthropic
import trulayer

trulayer.init(api_key="tl_...", project_name="my-project")
trulayer.instrument_anthropic(trulayer.get_client())

client = anthropic.Anthropic()
with trulayer.get_client().trace("answer-question"):
    client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        messages=[{"role": "user", "content": "hello"}],
    )
```

Same shape as OpenAI: pass the TruLayer client, in-place patching, idempotent, streaming-aware, sync + async. `trulayer.uninstrument_anthropic()` reverses it.

### LangChain

```python
import trulayer
from langchain_openai import ChatOpenAI

trulayer.init(api_key="tl_...", project_name="my-project")
handler = trulayer.instrument_langchain(trulayer.get_client())

llm = ChatOpenAI(callbacks=[handler])
with trulayer.get_client().trace("rag-query"):
    llm.invoke("hello")
```

`instrument_langchain(client)` **returns** a `BaseCallbackHandler` subclass — pass it to any LangChain LLM, chat model, or chain via `callbacks=[...]`. Works with `langchain_core` v0.1+.

### LlamaIndex

```python
import trulayer
from llama_index.core import Settings

trulayer.init(api_key="tl_...", project_name="my-project")
handler = trulayer.instrument_llamaindex()  # uses the global client
Settings.callback_manager.handlers.append(handler)
```

`instrument_llamaindex()` takes no arguments — it pulls the global client. Returns a `TruLayerCallbackHandler` to attach to a `CallbackManager`. Requires `llama-index-core>=0.10`.

### PydanticAI

```python
import trulayer
from pydantic_ai import Agent

trulayer.init(api_key="tl_...", project_name="my-project")
agent = Agent("openai:gpt-4o")

with trulayer.get_client().trace("agent-run") as trace:
    trulayer.instrument_pydanticai(agent, trace)
    result = agent.run_sync("hello")
```

Signature: `instrument_pydanticai(agent, trace_ctx, *, capture_input=True, capture_output=True, run_name=None) -> Agent`. Mutates `agent` in place and returns it. Patches `.run`, `.run_sync`, `.run_stream`, and tools.

### CrewAI

```python
with trulayer.get_client().trace("crew-kickoff") as trace:
    trulayer.instrument_crewai(crew, trace)
    crew.kickoff()
```

Signature: `instrument_crewai(crew, trace_ctx, *, capture_inputs=True, capture_outputs=True) -> Crew`.

### Haystack v2

```python
with trulayer.get_client().trace("pipeline-run") as trace:
    trulayer.instrument_haystack(pipeline, trace)
    pipeline.run({"query": "hello"})
```

Signature: `instrument_haystack(pipeline, trace_ctx, *, capture_inputs=True, capture_outputs=True) -> Pipeline`.

### AutoGen

```python
with trulayer.get_client().trace("autogen-chat") as trace:
    trulayer.instrument_autogen(agent, trace)
    agent.initiate_chat(other, message="hi")
```

Signature: `instrument_autogen(agent, trace_ctx, *, capture_messages=True) -> ConversableAgent`.

### DSPy

```python
with trulayer.get_client().trace("dspy-program") as trace:
    trulayer.instrument_dspy(trace)
    program(question="hello")
trulayer.uninstrument_dspy()  # optional teardown
```

Signature: `instrument_dspy(trace_ctx, *, capture_inputs=True, capture_outputs=True) -> None`. Globally patches `dspy.Predict.forward`. Idempotent. Pair with `uninstrument_dspy()` in tests.

> **Stability note:** DSPy, Haystack, and AutoGen instrumentation track upstream APIs that evolve quickly — treat these three as experimental. The OpenAI, Anthropic, and LangChain hooks are stable.

## Feedback

Submit user feedback against a captured trace:

```python
client.feedback(
    trace_id="018f...",
    label="thumbs_up",            # any short string label
    score=1.0,                    # optional float
    comment="great answer",       # optional
    metadata={"reviewer": "me"},  # optional dict
)
```

Synchronous, fire-and-forget — never raises into your code (warns on failure).

## Tags and metadata

```python
with client.trace(
    "checkout",
    tags=["env:prod", "model:gpt-4o"],
    metadata={"user_id": "u_123"},
) as trace:
    trace.add_tag("variant:v2")
    trace.set_metadata(experiment="prompt-a")
```

Do **not** put raw PII or secrets in `metadata` without a `scrub_fn` or `redactor` configured on `init()`.

## Redaction

Two ways, used independently or together.

### Quick: `scrub_fn`

A user-supplied `Callable[[str], str]` applied to `input`, `output`, and `error_message` on every trace and span before enqueue:

```python
import re
trulayer.init(
    api_key="tl_...",
    project_name="my-project",
    scrub_fn=lambda s: re.sub(r"\d{16}", "<CARD>", s),
)
```

### Rich: `Redactor`

```python
from trulayer import Redactor, Rule, BUILTIN_PACKS

redactor = Redactor(
    packs=["standard", "secrets"],
    rules=[Rule(name="employee_id", pattern=r"EMP-\d{6}")],
)
trulayer.init(api_key="tl_...", project_name="my-project", redactor=redactor)
```

Built-in pack names: `"standard"`, `"strict"`, `"phi"`, `"finance"`, `"secrets"` (see `BUILTIN_PACKS`).

Pseudonymization (HMAC-SHA256) instead of static replacement:

```python
Redactor(packs=["standard"], pseudonymize=True, pseudonymize_salt="my-secret")
```

One-shot helper (compiles fresh each call — prefer a long-lived `Redactor` for hot paths):

```python
from trulayer import redact
clean = redact("email a@b.com", packs=["standard"])
```

Field-targeted redaction on a span dict:

```python
redactor.redact_span(span_dict, fields=("input", "output", "metadata.user.email"))
```

## Sampling

`init(sample_rate=0.1)` keeps ~10% of traces. The decision is made on `__enter__` of each `TraceContext`. Spans inside a non-sampled trace are still recorded in memory but never enqueued.

## Flush and shutdown

```python
client.flush(timeout=5.0)     # flush buffered events, blocking; restarts the sender
client.shutdown(timeout=5.0)  # flush then stop the sender (atexit calls this)
```

`init()` registers an `atexit` handler so normal process exit flushes automatically. Long-running async workers and short-lived scripts that abort abnormally should call `flush()` explicitly before returning.

> **There is no `async_flush()`.** `flush()` is the only flush API; it is a synchronous wrapper safe to call from sync or async code (it talks to the batch sender thread).

## Staging vs prod

```python
trulayer.init(
    api_key="tl_test_...",
    project_name="my-project",
    endpoint="https://api.staging.trulayer.ai",
)
```

Use a staging API key issued in workspace settings.

## Testing

```python
from trulayer.testing import create_test_client, assert_sender

client, sender = create_test_client()           # returns (client, LocalBatchSender)
with client.trace("unit-test") as trace:
    with trace.span("step", "llm") as span:
        span.set_input("hi")
        span.set_output("bye")
client.flush()

assert_sender(sender).has_trace().span_count(1).has_span_named("step")
```

`create_test_client(**kwargs)` accepts the same kwargs as `TruLayerClient` (e.g. `project_name="..."`) and wires a `LocalBatchSender` so no network call is made and no API key is required.

`SenderAssertions` chain methods: `.has_trace(trace_id=None)`, `.span_count(n)`, `.has_span_named(name)`. Each raises `AssertionError` on mismatch and returns `self` for chaining.

For end-to-end tests against a deployed environment, set `TRULAYER_MODE=local` to short-circuit the network with no code changes.

## Common mistakes the agent must avoid

1. **`redact=` does not exist.** Use `scrub_fn=` for a callable, or `redactor=` for a `Redactor` instance. Passing `redact=` is a `TypeError`.
2. **`async_flush()` does not exist.** Use `client.flush(timeout=...)`. It is safe from async code.
3. **`instrument_openai` / `instrument_anthropic` take the TruLayer client, not the provider client and not a trace.** They patch globally in place; do not assign their return value.
4. **Provider-agnostic instrumentation hooks (`pydanticai`, `crewai`, `haystack`, `autogen`, `dspy`) take a `TraceContext` as their second argument** and must be called *inside* a `with client.trace(...) as trace:` block. `instrument_langchain` and `instrument_llamaindex` are the exceptions — they return a callback handler.
5. **`create_test_client()` returns a tuple `(client, sender)`,** not just a client. Pass the `sender` to `assert_sender()`, not `client._batch`.
6. **`project_id=` is deprecated** (still works, emits `DeprecationWarning`, removed in 0.3.x). Use `project_name=`.
7. **Calling `trulayer.init()` more than once per process** spawns extra batch sender threads. Call it once at startup; everywhere else use `trulayer.get_client()`.
8. **Forgetting `with` on a trace or span.** Exiting the context manager is what finalizes timing and queues the payload — without `with`, nothing is sent.
9. **Calling `instrument_openai` / `instrument_anthropic` after the provider client has been used in a hot loop.** Patch once at startup, before any LLM calls.
10. **Putting PII in `metadata` or `tags` without configuring `scrub_fn` or `redactor`.** These fields are passed through verbatim.
