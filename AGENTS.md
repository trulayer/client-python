# Codex instructions

This is the public TruLayer Python SDK repo. `CLAUDE.md` is the detailed source of truth; read it before making any non-trivial change.

## Scope

- `trulayer` Python package for trace capture, span instrumentation, batching, redaction, replay, testing helpers, and provider/framework instrumentation.
- Public customer-facing repo. Do not expose private service names, repo paths, planning issues, or private architecture.

## Working rules

- Make changes on a feature/fix branch and open a PR to `main`. Never commit directly to `main`.
- SDK code must not raise into user application code during normal telemetry failures.
- Do not log prompt, response, or user payload data at default log levels.
- Keep public exports documented when they change.
- Preserve API compatibility unless a semver-breaking release is intentional and documented.

## Verification

Run before opening a PR:

```bash
uv run pytest
uv run mypy .
uv run ruff check .
```

For packaging changes, also run:

```bash
uv build
```
