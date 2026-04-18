# Python SDK — Implementation Tasks

Track implementation progress for the TruLayer AI Python SDK.

## Status Legend

- [ ] Not started
- [~] In progress
- [x] Done

---

## Phase 1: Core SDK — [TRU-15](https://linear.app/omnimoda/issue/TRU-15)

### Project Setup
- [ ] Initialize `pyproject.toml` (name: `trulayer`, Python 3.11+)
- [ ] Configure `uv` workspace
- [ ] Set up `ruff` (lint + format), `mypy` (strict), `pytest`
- [ ] Configure CI: lint, type-check, test with coverage gate (>90%)
- [ ] Set up `src/trulayer/` layout

### Core Models — [TRU-8](https://linear.app/omnimoda/issue/TRU-8)
- [ ] `TraceData` Pydantic model (id, project, session, metadata, spans)
- [ ] `SpanData` Pydantic model (id, trace_id, type, input, output, metadata, timing)
- [ ] `EventData` model (level, message, metadata)
- [ ] `FeedbackData` model

### Client & Init
- [ ] `TruLayerClient` class (holds config, batch sender reference)
- [ ] `trulayer.init(api_key, project, environment, endpoint, ...)` global initializer
- [ ] Thread-local client registry (support multiple clients in tests)

### Trace & Span Context Managers
- [ ] `trulayer.trace(name, session_id, metadata)` — sync context manager
- [ ] `trace.span(name, type, metadata)` — sync span context manager
- [ ] `trulayer.atrace(...)` — async context manager variant
- [ ] `trace.aspan(...)` — async span context manager
- [ ] Auto-capture start/end time, exception status
- [ ] `.set_input()`, `.set_output()`, `.set_metadata()` on span

### Batch Sender
- [ ] `asyncio.Queue`-based batch buffer
- [ ] Dedicated background thread running the async flush loop
- [ ] Flush on `batch_size` threshold (default: 50)
- [ ] Flush on `flush_interval` timer (default: 2s)
- [ ] `atexit` shutdown hook with 5s drain timeout
- [ ] HTTP retry with exponential backoff (3 retries)
- [ ] Drop + warn on max retries exceeded (never raise)

### Auto-Instrumentation
- [ ] `instrument_openai(client)` — wrap `chat.completions.create` and `completions.create`
- [ ] `instrument_anthropic(client)` — wrap `messages.create`
- [ ] `instrument_langchain()` — LangChain callback handler
- [ ] `uninstrument_openai()` / etc. — unpatch for test isolation

### Feedback API
- [ ] `trulayer.feedback(trace_id, score, label, comment)` — submit feedback

---

## Phase 2: V1 Enhancements

- [ ] `instrument_llamaindex()` — LlamaIndex integration
- [ ] Streaming support: capture token-by-token for streaming completions
- [ ] Sampling rate configuration (e.g., only send 10% of traces)
- [ ] PII scrubbing / redaction hooks (user-provided callable)
- [ ] Custom metadata schema validation

---

## Engineering Checklist (per PR)

- [ ] Unit tests written (>90% coverage for new code)
- [ ] `uv run mypy .` passes with no errors
- [ ] `uv run ruff check .` passes
- [ ] Public API changes reflected in README
- [ ] Breaking changes bump minor version
