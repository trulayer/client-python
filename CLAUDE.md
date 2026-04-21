# CLAUDE.md — Python SDK (client-python)

## Project Purpose

The `trulayer` Python package. Provides trace capture, span instrumentation, and auto-instrumentation hooks for OpenAI, Anthropic, and LangChain. Designed for minimal latency overhead and easy integration.

## Tech Stack

- Python 3.11+
- `httpx` — async HTTP client for sending batches to the ingestion API
- `pydantic` v2 — data validation and serialization
- `uv` — package and environment management
- `pytest` + `pytest-asyncio` — testing
- `ruff` — lint and format
- `mypy` — static type checking

## Definition of Done

A task is **not done** until all of the following are true — in order:

1. **Tests pass** — `uv run pytest` green, `uv run mypy src/`, `uv run ruff check` zero errors.
2. **Docs updated** — **Documentation is part of Done.** Any PR that changes public SDK exports must include the corresponding update to the public docs in the same PR. "I'll update docs in a follow-up" is not acceptable. If the docs change is too big to ship with the code, split the code change first.
3. **Committed on a feature branch** — all changed files committed on a branch named `feat/...` or `fix/...`. **Never commit directly to `main`.**
4. **PR opened** — `gh pr create` targeting `main` with a summary of what changed and why.
5. **PR merged** — `gh pr merge --squash`. Work on the next task cannot begin until this PR is merged.
5. **Working tree clean** — after merge, `git status` must show nothing to commit. No uncommitted modifications, no untracked feature files. If something is still local, commit it or delete it — leaving code in a branch without a PR is the same as it not existing.

**Direct pushes to `main` are forbidden.** Every change must go through a pull request.

## CI is gating

Every pull request must pass CI before it can be merged. If CI fails, the engineer who opened the PR owns the fix — not a reviewer, not a follow-up task. Don't merge with failing CI. Don't bypass with `--admin` or `--no-verify`. If a check is flaky, fix it or remove it — don't skip it.

## Key Commands

```bash
uv sync --dev               # Install all dependencies including dev
uv run pytest               # Run tests
uv run pytest --cov         # Tests with coverage (target: >90%)
uv run pytest -x            # Stop on first failure
uv run ruff check .         # Lint
uv run ruff format .        # Format
uv run mypy .               # Type check
uv build                    # Build sdist + wheel
```

## Project Layout

```text
src/trulayer/
  __init__.py           → public API exports + init()
  client.py             → TruLayerClient (init, flush, shutdown)
  trace.py              → Trace and Span context managers
  batch.py              → async batch sender (queue + flush loop)
  model.py              → Pydantic models (TraceData, SpanData, etc.)
  instruments/
    openai.py           → OpenAI auto-instrumentation patch
    anthropic.py        → Anthropic auto-instrumentation patch
    langchain.py        → LangChain callback handler
tests/
  unit/                 → pytest unit tests (no network)
  integration/          → tests against a real/mock TruLayer API
```

## Coding Conventions

- Type annotations on all functions — `mypy --strict` must pass
- `async` for all I/O in the batch sender; sync public API wraps async via an internal event loop thread
- Context managers for traces (`with trulayer.trace(...)`) and spans (`with trace.span(...)`)
- Thread-safe batch queue using `asyncio.Queue` (producer: any thread; consumer: dedicated loop thread)
- Graceful shutdown: `atexit` handler flushes remaining batches
- Never raise exceptions from SDK code into user code — catch internally, log with `warnings.warn`
- Do not log user data (prompt/response content) at default log levels — only at `DEBUG`

## Batch Sender Behavior

- Events are queued and flushed every `flush_interval` seconds (default: 2s) or when `batch_size` is reached (default: 50)
- On shutdown, flush is attempted with a 5s timeout
- HTTP failures retry up to 3 times with exponential backoff
- After max retries, events are dropped with a warning (never block user code)

## ID Generation

All `trace_id` and `span_id` values are **UUIDv7**, generated via the `uuid` package:

```python
import uuid
trace_id = str(uuid.uuid7())  # requires Python 3.14+ or the `uuid7` backport package
```

## Auto-Instrumentation

Patching strategy: monkey-patch the provider client's internal completion method to wrap calls in a span. Patches must be reversible (`unpatch()`). Never modify user's client instance globally without their opt-in.

## Testing

- Unit tests mock `httpx.AsyncClient` — no real network calls
- Test trace/span context managers, batching logic, retry behavior
- Integration tests in `tests/integration/` against a local mock server
- Coverage target: **90%** (SDK reliability requires high coverage)
- All async tests use `pytest-asyncio` with `asyncio_mode = "auto"`

## Publishing

- Package name: `trulayer`
- Built with `uv build` → publishes to PyPI via CI
- Version follows semver — bump in `pyproject.toml`

## Public Repository Policy

This repository ships to TruLayer customers. Do not introduce references to internal code, internal repositories (e.g. the TruLayer API service or dashboard), internal planning documents, internal Linear issue content, or internal architectural details. Refer to the platform as "TruLayer" or "the TruLayer API" — not as specific internal components. If in doubt, leave it out.
