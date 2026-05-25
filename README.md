# langoraph

[![CI](https://github.com/Succaiss-applied-ai/langoraph/actions/workflows/ci.yml/badge.svg)](https://github.com/Succaiss-applied-ai/langoraph/actions/workflows/ci.yml)
[![Security](https://github.com/Succaiss-applied-ai/langoraph/actions/workflows/security.yml/badge.svg)](https://github.com/Succaiss-applied-ai/langoraph/actions/workflows/security.yml)

`langoraph` is a Go library for building typed, stateful graph workflows with deterministic parallel execution. It is inspired by [LangGraph](https://github.com/langchain-ai/langgraph) and includes an OpenAI-compatible LLM client for DashScope, DeepSeek, OpenAI, and similar providers.

## Features

- Typed graph runtime with fixed, conditional, and parallel edges.
- Deterministic fan-out semantics for graph branches, slices, and batches.
- Ordered concurrent map/reduce helper through `Fanout`.
- OpenAI-compatible chat client with streaming, first-token timeouts, and structured-output retry helpers.
- Unit-testable runtime behavior with no provider API keys required.

## Installation

```bash
go get github.com/Succaiss-applied-ai/langoraph
```

`langoraph` supports the Go version declared in [`go.mod`](go.mod).

## Quick Start

```go
package main

import (
	"context"
	"strings"

	"github.com/Succaiss-applied-ai/langoraph"
)

type State struct {
	Input  string
	Output string
}

func main() {
	g := langoraph.NewGraph[State]()

	g.AddNode("normalize", func(_ context.Context, s *State) error {
		s.Output = strings.TrimSpace(s.Input)
		return nil
	})
	g.AddNode("uppercase", func(_ context.Context, s *State) error {
		s.Output = strings.ToUpper(s.Output)
		return nil
	})

	g.AddEdge(langoraph.START, "normalize")
	g.AddEdge("normalize", "uppercase")
	g.AddEdge("uppercase", langoraph.END)

	state := &State{Input: " hello "}
	if err := g.Run(context.Background(), state); err != nil {
		panic(err)
	}
}
```

## Core API

### Conditional routing

```go
g.AddConditionalEdge(
	"classify",
	func(_ context.Context, s *State) string {
		if s.Output == "" {
			return "empty"
		}
		return "ready"
	},
	map[string]string{
		"empty": "fallback",
		"ready": "process",
	},
)
```

### Parallel branches

```go
g.AddParallelEdge(
	"prepare",
	[]string{"branch_a", "branch_b", "branch_c"},
	"aggregate",
)
```

### Fanout

```go
results, err := langoraph.Fanout(
	context.Background(),
	[]string{"a", "b", "c"},
	func(ctx context.Context, item string) (string, error) {
		return strings.ToUpper(item), nil
	},
)
```

`Fanout` returns results in input order, regardless of goroutine completion order.

## Parallel Semantics

`Fanout`, `Graph.AddParallelEdge`, and `RunAll` share the same contract:

1. All parallel work runs to completion unless the caller cancels the context.
2. The first non-nil error is returned in input or branch order.
3. `ErrorRecorder.RecordError` calls are serialized by the runtime.
4. Parallel graph branches share the same state pointer.

These rules are covered by `parallel_semantics_test.go`.

## LLM Client

The `llm` package supports OpenAI-compatible chat providers.

| Provider | API key | Base URL | Model |
| --- | --- | --- | --- |
| DashScope | `DASHSCOPE_API_KEY` | `DASHSCOPE_BASE_URL` | `DASHSCOPE_MODEL` |
| DeepSeek | `DEEPSEEK_API_KEY` | `DEEPSEEK_BASE_URL` | - |
| OpenAI | `OPENAI_API_KEY` | `OPENAI_BASE_URL` | - |

```go
import "github.com/Succaiss-applied-ai/langoraph/llm"

client, err := llm.NewClient(llm.Config{
	Stream:         true,
	TimeoutSeconds: 180,
})
if err != nil {
	return err
}

resp, err := client.Chat(ctx, []llm.Message{
	{Role: "user", Content: "Return JSON: {\"answer\":\"...\"}"},
})
if err != nil {
	return err
}
_ = resp
```

Structured-output helpers are available through `ChatStructured` and `ChatStructuredWithFeedback`.

## Testing

```bash
make ci
```

Equivalent commands:

```bash
go vet ./...
go test ./...
go test -race ./...
```

Provider integration tests are opt-in:

```bash
go test ./... -run Integration -v
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for pull request expectations and local checks.

## License

[MIT](LICENSE)
