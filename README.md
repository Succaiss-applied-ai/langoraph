# langoraph

A Go implementation of [LangGraph](https://github.com/langchain-ai/langgraph) primitives — typed pipelines, concurrent fan-out, and a plug-and-play LLM client that works with DashScope (Qwen), DeepSeek, and OpenAI out of the box.

## Features

- **`Pipeline[S]`** — run a fixed sequence of typed nodes against shared state, with optional per-node error recording
- **`RunAll`** — execute one pipeline against many independent states concurrently
- **`Fanout`** — concurrently process a slice of items and collect results in input order
- **`llm` package** — unified `Client` interface backed by any OpenAI-compatible provider

## Installation

```bash
go get github.com/zhaocun/langoraph
```

Requires **Go 1.21+**.

## Quick Start

### Pipeline

```go
package main

import (
    "context"
    "fmt"

    langoraph "github.com/zhaocun/langoraph"
)

type State struct {
    Input  string
    Result string
}

func main() {
    var p langoraph.Pipeline[State]

    p.AddNode("step1", func(_ context.Context, s *State) error {
        s.Input = "hello"
        return nil
    })
    p.AddNode("step2", func(_ context.Context, s *State) error {
        s.Result = s.Input + " world"
        return nil
    })

    state := &State{}
    if err := p.Run(context.Background(), state); err != nil {
        panic(err)
    }
    fmt.Println(state.Result) // hello world
}
```

### Fanout

```go
results, err := langoraph.Fanout(
    context.Background(),
    []string{"item1", "item2", "item3"},
    func(ctx context.Context, item string) (string, error) {
        return "[processed] " + item, nil
    },
)
```

Results are returned in the same order as the input slice, regardless of which goroutine finishes first.

### LLM Client

Set one of the following environment variables, then call `llm.NewClient`:

| Provider   | API Key Env            | Base URL Env             | Model Env          |
|------------|------------------------|---------------------------|--------------------|
| DashScope  | `DASHSCOPE_API_KEY`    | `DASHSCOPE_BASE_URL`      | `DASHSCOPE_MODEL`  |
| DeepSeek   | `DEEPSEEK_API_KEY`     | `DEEPSEEK_BASE_URL`       | —                  |
| OpenAI     | `OPENAI_API_KEY`       | `OPENAI_BASE_URL`         | —                  |

When `Provider` is empty, the first key found wins (DashScope → DeepSeek → OpenAI).

```go
import "github.com/zhaocun/langoraph/llm"

client, err := llm.NewClient(llm.Config{Temperature: 0.7, TimeoutSeconds: 30})
if err != nil {
    log.Fatal(err)
}

// Structured JSON output
var out struct {
    Answer string `json:"answer"`
}
err = llm.ChatStructured(ctx, client, "What is 1+1? Reply JSON: {\"answer\":\"...\"}", &out)
```

## Running Tests

Unit tests (no API key required):

```bash
go test ./...
```

Integration tests (requires an API key):

```bash
go test ./... -run Integration -v
```

## License

[MIT](LICENSE)
