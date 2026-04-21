# Python SDK — Implementation Tasks

**Due Date: 2026-04-30**

Track implementation progress for the TruLayer AI Python SDK.

## Status Legend

- [ ] Not started
- [~] In progress
- [x] Done

---

## Phase 1: Core SDK

### Project Setup

- [x] Initialize `pyproject.toml` (name: `trulayer`, Python 3.11+)
- [x] Configure `uv` workspace
- [x] Set up `ruff` (lint + format), `mypy` (strict), `pytest`
- [x] Configure CI: lint, type-check, test with coverage gate (>90%)
- [x] Set up `src/trulayer/` layout

### Core Models

- [x] `TraceData` Pydantic model (id, project, session, metadata, spans)
- [x] `SpanData` Pydantic model (id, trace_id, type, input, output, metadata, timing)
- [x] `EventData` model (level, message, metadata)
- [x] `FeedbackData` model

### Client & Init

- [x] `TruLayerClient` class (holds config, batch sender reference)
- [x] `trulayer.init(api_key, project, endpoint, ...)` global initializer
- [x] `contextvars`-based trace registry (safe across threads and async)

### Trace & Span Context Managers

- [x] `trulayer.trace(name, session_id, metadata)` — sync + async context manager
- [x] `trace.span(name, type, metadata)` — sync + async span context manager
- [x] Auto-capture start/end time, exception status
- [x] `.set_input()`, `.set_output()`, `.set_metadata()` on span and trace

### Batch Sender

- [x] Thread-safe queue (producer: any thread; consumer: dedicated async loop thread)
- [x] Flush on `batch_size` threshold (default: 50)
- [x] Flush on `flush_interval` timer (default: 2s)
- [x] `atexit` shutdown hook with 5s drain timeout
- [x] HTTP retry with exponential backoff (3 retries)
- [x] Drop + warn on max retries exceeded (never raise)

### Auto-Instrumentation

- [x] `instrument_openai(client)` — wrap `chat.completions.create` (sync + async)
- [x] `instrument_anthropic(client)` — wrap `messages.create` (sync + async)
- [x] `instrument_langchain()` — LangChain callback handler (`TruLayerCallbackHandler`; handles `on_llm_start`, `on_chat_model_start`, `on_llm_end`, `on_llm_error`)
- [x] `uninstrument_openai()` / `uninstrument_anthropic()` — unpatch for test isolation

### Feedback API

- [x] `client.feedback(trace_id, label, score, comment)` — submit feedback

---

## Phase 2: V1 Enhancements

- [x] **`instrument_llamaindex()` — LlamaIndex integration** ← top priority
- [x] **`instrument_pydanticai()` — PydanticAI Agent instrumentation** (TRU-80)
- [x] Streaming support: sync + async generator wrappers for OpenAI and Anthropic (`stream=True`)
- [x] Sampling rate configuration (e.g., only send 10% of traces)
- [x] PII scrubbing / redaction hooks (user-provided callable)
- [x] Custom metadata schema validation
- [x] **Local/offline sandbox mode** (`TRULAYER_MODE=local`, `LocalBatchSender`, `create_test_client`, `assert_sender`) (TRU-81)
- [x] **`instrument_crewai()` — CrewAI Crew instrumentation** (TRU-85)
- [x] **`instrument_dspy()` / `uninstrument_dspy()` — DSPy Predict instrumentation** (TRU-85)
- [x] **`instrument_haystack()` — Haystack v2 Pipeline instrumentation** (TRU-85)
- [x] **`instrument_autogen()` — AutoGen ConversableAgent instrumentation** (TRU-85)

---

## Engineering Checklist (per PR)

- [ ] Unit tests written (>90% coverage for new code)
- [ ] `uv run mypy .` passes with no errors
- [ ] `uv run ruff check .` passes
- [ ] Public API changes reflected in README
- [ ] Breaking changes bump minor version
