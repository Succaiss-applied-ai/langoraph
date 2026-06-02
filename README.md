# langoraph

[![CI](https://github.com/Succaiss-applied-ai/langoraph/actions/workflows/ci.yml/badge.svg)](https://github.com/Succaiss-applied-ai/langoraph/actions/workflows/ci.yml)
[![Security](https://github.com/Succaiss-applied-ai/langoraph/actions/workflows/security.yml/badge.svg)](https://github.com/Succaiss-applied-ai/langoraph/actions/workflows/security.yml)

A Go implementation of [LangGraph](https://github.com/langchain-ai/langgraph) primitives — typed graphs with conditional/parallel execution, concurrent fan-out, and a streaming-capable LLM client that works with DashScope (Qwen), DeepSeek, and OpenAI out of the box.

The parallel-execution semantics are deliberately matched to LangGraph's runtime so you can port a Python `StateGraph` to Go without surprises.

## Features

- **`Graph[S]`** — typed nodes connected by edges, with conditional routing and parallel fan-out (LangGraph parity)
- **`RunAll`** — drive one graph against many independent states concurrently
- **`Fanout`** — concurrently process a slice of items and collect results in input order
- **`llm` package** — unified `Client` interface for any OpenAI-compatible provider, with optional SSE streaming, first-token watchdog (with reasoning-content heartbeat), and a feedback-driven structured-output retry loop

## Installation

```bash
go get github.com/Succaiss-applied-ai/langoraph
```

Requires the Go version declared in [`go.mod`](go.mod).

## Quick Start

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

### Fanout (dynamic per-item parallelism)

```go
results, err := langoraph.Fanout(
    context.Background(),
    []string{"item1", "item2", "item3"},
    func(ctx context.Context, item string) (string, error) {
        return "[processed] " + item, nil
    },
)
```

Results are returned in the same order as the input slice, regardless of which goroutine finishes first. This is the Go equivalent of LangGraph's dynamic `Send()` fan-out plus an `operator.add` reducer.

## Parallel semantics — LangGraph parity

`Fanout`, `Graph.AddParallelEdge` and `RunAll` all share one contract that mirrors LangGraph's Pregel runtime:

1. **Wait-all.** Every parallel branch / item / state runs to its natural completion, even if a peer errors. The shared `ctx` is **never** cancelled by langoraph itself — external cancellation still aborts every in-flight branch.
2. **Deterministic first error.** After all branches return, the first non-nil error in the original input order is returned. This stays the same across runs even though scheduling is non-deterministic.
3. **`ErrorRecorder` is serialised.** When `S` implements `RecordError(name, err)`, parallel branches still all run, and `RecordError` calls are funnelled through an internal mutex so user-supplied recorders can stay racy-but-simple (e.g. a slice append).
4. **Branches share `*S` by reference.** Just like LangGraph's typed-dict reducers, the safe pattern is for each branch to mutate disjoint fields, or to return into per-key buckets that an aggregator node merges later.

The contract is pinned by `parallel_semantics_test.go` — touch any of the four points above and that file lights up.

### The "per-L0 fan-out" pattern

LangGraph's `_per_l0_fan_out` (asyncio.gather inside a node body) ports cleanly to `Fanout` inside a node:

```go
g.AddNode("map_jd_l3", func(ctx context.Context, s *MapState) error {
    grouped := groupByL0(s.JDChunks)
    type result struct {
        L0    string
        Items []JDItem
    }
    results, err := langoraph.Fanout(ctx, grouped, func(ctx context.Context, g group) (result, error) {
        items, err := mapOneL0(ctx, g)
        return result{L0: g.L0, Items: items}, err
    })
    if err != nil {
        return err
    }
    for _, r := range results {
        s.ByL0[r.L0] = r.Items
    }
    return nil
})
```

## LLM Client

Set one of the following environment variables, then call `llm.NewClient`:

| Provider   | API Key Env            | Base URL Env              | Model Env          |
|------------|------------------------|---------------------------|--------------------|
| DashScope  | `DASHSCOPE_API_KEY`    | `DASHSCOPE_BASE_URL`      | `DASHSCOPE_MODEL`  |
| DeepSeek   | `DEEPSEEK_API_KEY`     | `DEEPSEEK_BASE_URL`       | —                  |
| OpenAI     | `OPENAI_API_KEY`       | `OPENAI_BASE_URL`         | —                  |

When `Provider` is empty, the first key found wins (DashScope → DeepSeek → OpenAI).

```go
import "github.com/Succaiss-applied-ai/langoraph/llm"

client, err := llm.NewClient(llm.Config{
    Temperature:    0.7,
    TimeoutSeconds: 30,
})
if err != nil {
    log.Fatal(err)
}

var out struct {
    Answer string `json:"answer"`
}
_ = llm.ChatStructured(ctx, client, "What is 1+1? Reply JSON: {\"answer\":\"...\"}", &out)
```

### Streaming + first-token watchdog (thinking models)

DashScope/DeepSeek "thinking" models (`qwen-deepseek`, `deepseek-v4-pro/flash`, ...) routinely take several seconds to surface their first content token while the reasoning trace is being produced. Non-streaming requests sit on the connection until the model finishes thinking and frequently exceed the HTTP timeout.

`langoraph` ships with a streaming SSE client that:

- Opens a `text/event-stream` connection with `stream_options.include_usage=true`.
- Runs a **first-token watchdog**: if no `content` (or `reasoning_content`) delta arrives within `FirstTokenTimeout`, the response body is closed and the call surfaces a typed `*llm.FirstTokenTimeoutError` (also a `net.Error.Timeout()=true`).
- Treats any `reasoning_content` delta as proof-of-life heartbeat, so reasoning bursts do not falsely time out.
- Retries up to `FirstTokenMaxRetries` extra SSE attempts on first-token timeout before bubbling the error.
- Extracts streaming `usage` (prompt / completion / reasoning tokens) from the final usage-only chunk.

Configure once at the client level:

```go
client, _ := llm.NewClient(llm.Config{
    Stream:                true,
    FirstTokenTimeout:     8 * time.Second,
    FirstTokenMaxRetries:  2,
    Temperature:           0.1,
    TimeoutSeconds:        180,
})
```

Or override per-call via functional options:

```go
resp, err := client.Chat(ctx, msgs,
    llm.WithStream(true),
    llm.WithFirstTokenTimeout(2*time.Second),
    llm.WithFirstTokenMaxRetries(0),
    llm.WithTemperature(0.0),
    llm.WithMaxTokens(4096),
    llm.WithSeed(42),
)
```

Available options: `WithStream`, `WithTemperature`, `WithTopP`, `WithSeed`, `WithMaxTokens`, `WithEnableThinking`, `WithReasoningEffort`, `WithFirstTokenTimeout`, `WithFirstTokenMaxRetries`.

### Feedback-driven validation retry

Mirrors Python's `call_llm_json_with_validation`: send a structured-output prompt, validate the parsed JSON, and on failure re-prompt the model with the validator's feedback message appended to the chat history.

```go
type Selection struct {
    IDs []string
}

allow := map[string]bool{"L0_a": true, "L0_b": true}

validator := func(payload map[string]any) llm.ValidatorVerdict[Selection] {
    raw, ok := payload["ids"].([]any)
    if !ok {
        return llm.ValidatorVerdict[Selection]{
            OK: false, Feedback: `请只输出 {"ids":["L0_a", ...]}。`,
        }
    }
    var keep []string
    var unknown []string
    for _, v := range raw {
        s, _ := v.(string)
        if allow[s] {
            keep = append(keep, s)
        } else {
            unknown = append(unknown, s)
        }
    }
    if len(unknown) > 0 {
        return llm.ValidatorVerdict[Selection]{
            OK:         false,
            Normalised: Selection{IDs: keep},
            Feedback:   fmt.Sprintf("以下 id 不在白名单内：%v", unknown),
        }
    }
    return llm.ValidatorVerdict[Selection]{OK: true, Normalised: Selection{IDs: keep}}
}

outcome, err := llm.ChatStructuredWithFeedback(
    ctx, client, msgs,
    "L0Selection", schemaMap,   // schemaName + schema → json_schema mode
    validator, 3,               // up to 3 feedback retries
    llm.WithStream(true),       // any ChatOption flows through
)
fmt.Println(outcome.Result.IDs, outcome.RetryCount, outcome.SchemaMode)
```

The verdict semantics match Python:

- `OK=true` → return immediately.
- `OK=false, Feedback==""` → validator chose to accept the partial; return without retry.
- `OK=false, Feedback!=""` → append `(assistant: last raw output, user: feedback)` and re-prompt.
- After `maxRetries` exhausts, the most recent `Normalised` payload is returned with `RetryCount` reflecting how many feedback retries actually fired.

## Running Tests

Unit tests (no API key required):

```bash
go test ./... -short -race
```

Full local gate:

```bash
make ci
```

Integration tests (requires an API key in `.env` or env vars):

```bash
go test ./... -run Integration -v
```

## License

[MIT](LICENSE)
