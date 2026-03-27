# langoraph

A Go implementation of [LangGraph](https://github.com/langchain-ai/langgraph) primitives — typed pipelines, conditional/parallel graph execution, concurrent fan-out, and a plug-and-play LLM client that works with DashScope (Qwen), DeepSeek, and OpenAI out of the box.

## Features

- **`Pipeline[S]`** — run a fixed sequence of typed nodes against shared state, with optional per-node error recording
- **`Graph[S]`** — execute nodes connected by directed edges with support for conditional routing and parallel fan-out
- **`RunAll`** — execute one pipeline against many independent states concurrently
- **`Fanout`** — concurrently process a slice of items and collect results in input order
- **`llm` package** — unified `Client` interface backed by any OpenAI-compatible provider

## Installation

```bash
go get github.com/Succaiss-applied-ai/langoraph
```

Requires **Go 1.22+**.

## Quick Start

### Pipeline (linear)

```go
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
_ = p.Run(context.Background(), state)
fmt.Println(state.Result) // hello world
```

### Graph (conditional edges)

```go
g := langoraph.NewGraph[State]()

g.AddNode("classify", classifyFn)
g.AddNode("positive", positiveFn)
g.AddNode("negative", negativeFn)

g.AddEdge(langoraph.START, "classify")
g.AddConditionalEdge("classify",
    func(_ context.Context, s *State) string {
        if s.Sentiment >= 0 { return "pos" }
        return "neg"
    },
    map[string]string{"pos": "positive", "neg": "negative"},
)
g.AddEdge("positive", langoraph.END)
g.AddEdge("negative", langoraph.END)

_ = g.Run(context.Background(), &State{})
```

### Graph (parallel fan-out + join)

```go
g := langoraph.NewGraph[State]()

g.AddNode("prepare", prepareFn)
g.AddNode("branch_a", branchAFn)
g.AddNode("branch_b", branchBFn)
g.AddNode("branch_c", branchCFn)
g.AddNode("aggregate", aggregateFn)

g.AddEdge(langoraph.START, "prepare")
g.AddParallelEdge("prepare", []string{"branch_a", "branch_b", "branch_c"}, "aggregate")
g.AddEdge("aggregate", langoraph.END)

_ = g.Run(context.Background(), &State{})
```

Branches run concurrently on the shared state. Callers must ensure branch functions do not race on overlapping fields.

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
import "github.com/Succaiss-applied-ai/langoraph/llm"

client, err := llm.NewClient(llm.Config{Temperature: 0.7, TimeoutSeconds: 30})
if err != nil {
    log.Fatal(err)
}

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
