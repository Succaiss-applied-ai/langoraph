# AGENTS.md

Guidance for Codex and other coding agents working in this repository.

## Project

`langoraph` is a small Go module that provides typed graph execution primitives and an OpenAI-compatible LLM client. Keep the public API conservative and backward compatible unless the task explicitly asks for a breaking change.

## Local Checks

Run the narrowest useful set while iterating, then the full gate before handing off:

```bash
gofmt -w $(git ls-files '*.go')
go vet ./...
go test ./...
go test -race ./...
```

`make ci` runs the same local gate.

## Testing Rules

- Unit tests must not require provider API keys.
- Integration tests that call real providers must stay opt-in and be selected explicitly with `go test ./... -run Integration -v`.
- Preserve the parallel execution contract pinned by `parallel_semantics_test.go`: wait-all, deterministic first error, serialized `ErrorRecorder`, and shared state by pointer.
- For LLM changes, cover both non-streaming and streaming/SSE behavior where practical.

## Style

- Prefer simple Go over new abstractions.
- Keep comments for concurrency, protocol parsing, and retry behavior where the intent is otherwise easy to miss.
- Do not introduce generated files or vendored dependencies unless requested.
