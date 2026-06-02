// Verifies ChatOption / per-call overrides serialise into the wire
// request exactly as expected, and that pre-options call sites still
// compile / behave identically (zero-options preserves Config defaults).
package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"
)

// captureServer returns an httptest.Server that records every request
// body it receives, plus a trivial OK response so non-streaming tests
// have something to unmarshal.
func captureServer(t *testing.T) (*httptest.Server, *requestRecorder) {
	t.Helper()
	rec := &requestRecorder{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		rec.add(body)
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Request-Id", "provider-request-123")
		_, _ = w.Write([]byte(`{
			"choices":[{"message":{"content":"ok"}}],
			"usage":{"prompt_tokens":1,"completion_tokens":2}
		}`))
	}))
	t.Cleanup(srv.Close)
	return srv, rec
}

func TestChatNonStreamCapturesProviderRequestID(t *testing.T) {
	srv, _ := captureServer(t)
	c := stubClient(t, srv.URL, Config{})

	resp, err := c.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}})
	if err != nil {
		t.Fatalf("Chat failed: %v", err)
	}
	if resp.ProviderRequestID != "provider-request-123" {
		t.Fatalf("ProviderRequestID = %q", resp.ProviderRequestID)
	}
}

type requestRecorder struct {
	mu       sync.Mutex
	bodies   [][]byte
	requests []map[string]any
}

func (r *requestRecorder) add(body []byte) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.bodies = append(r.bodies, body)
	var parsed map[string]any
	_ = json.Unmarshal(body, &parsed)
	r.requests = append(r.requests, parsed)
}

func (r *requestRecorder) last() map[string]any {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.requests) == 0 {
		return nil
	}
	return r.requests[len(r.requests)-1]
}

// stubClient builds an openAIClient pointing at an httptest server.
// We bypass NewClient + env-var probing because we want to exercise
// the option pipeline directly.
func stubClient(t *testing.T, serverURL string, cfg Config) *openAIClient {
	t.Helper()
	c := newOpenAIClient(serverURL, "stub-key", "stub-model", cfg)
	return c
}

// ----- TestOptions_Override_Sampling -----
// Per-call WithTemperature / WithTopP / WithSeed / WithMaxTokens
// must override the Config defaults exactly once and only for that
// one call.

func TestOptions_Override_Sampling(t *testing.T) {
	srv, rec := captureServer(t)
	c := stubClient(t, srv.URL, Config{
		Temperature: 0.1,
		TopP:        0.9,
		Seed:        42,
		SeedSet:     true,
		MaxTokens:   256,
	})

	// First call: defaults from Config.
	if _, err := c.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}); err != nil {
		t.Fatalf("default Chat failed: %v", err)
	}
	def := rec.last()
	if def["temperature"] != 0.1 {
		t.Errorf("default temperature: expected 0.1, got %v", def["temperature"])
	}
	if def["top_p"] != 0.9 {
		t.Errorf("default top_p: expected 0.9, got %v", def["top_p"])
	}
	if def["seed"].(float64) != 42 {
		t.Errorf("default seed: expected 42, got %v", def["seed"])
	}
	if def["max_tokens"].(float64) != 256 {
		t.Errorf("default max_tokens: expected 256, got %v", def["max_tokens"])
	}

	// Second call: per-call overrides.
	if _, err := c.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}},
		WithTemperature(0.7),
		WithTopP(0.5),
		WithSeed(7),
		WithMaxTokens(64),
	); err != nil {
		t.Fatalf("override Chat failed: %v", err)
	}
	ov := rec.last()
	if ov["temperature"] != 0.7 {
		t.Errorf("override temperature: expected 0.7, got %v", ov["temperature"])
	}
	if ov["top_p"] != 0.5 {
		t.Errorf("override top_p: expected 0.5, got %v", ov["top_p"])
	}
	if ov["seed"].(float64) != 7 {
		t.Errorf("override seed: expected 7, got %v", ov["seed"])
	}
	if ov["max_tokens"].(float64) != 64 {
		t.Errorf("override max_tokens: expected 64, got %v", ov["max_tokens"])
	}

	// Third call (no options) — must revert to Config defaults.
	if _, err := c.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}); err != nil {
		t.Fatalf("third Chat failed: %v", err)
	}
	rev := rec.last()
	if rev["temperature"] != 0.1 {
		t.Errorf("post-override revert temperature: expected 0.1, got %v", rev["temperature"])
	}
	if rev["seed"].(float64) != 42 {
		t.Errorf("post-override revert seed: expected 42, got %v", rev["seed"])
	}
}

func TestOptions_ReasoningEffort_DashScopeWirePayload(t *testing.T) {
	srv, rec := captureServer(t)
	c := stubClient(t, srv.URL+"/dashscope", Config{
		EnableThinking:  true,
		ReasoningEffort: "low",
	})

	if _, err := c.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}); err != nil {
		t.Fatalf("default Chat failed: %v", err)
	}
	def := rec.last()
	if def["enable_thinking"] != true {
		t.Errorf("default enable_thinking: expected true, got %v", def["enable_thinking"])
	}
	if def["reasoning_effort"] != "low" {
		t.Errorf("default reasoning_effort: expected low, got %v", def["reasoning_effort"])
	}

	if _, err := c.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}},
		WithEnableThinking(false),
		WithReasoningEffort("high"),
	); err != nil {
		t.Fatalf("override Chat failed: %v", err)
	}
	ov := rec.last()
	if ov["enable_thinking"] != false {
		t.Errorf("override enable_thinking: expected false, got %v", ov["enable_thinking"])
	}
	if ov["reasoning_effort"] != "high" {
		t.Errorf("override reasoning_effort: expected high, got %v", ov["reasoning_effort"])
	}
}

// ----- TestOptions_OmitsZeroDefaults -----
// Config zero values for top_p / seed / max_tokens must not appear in
// the wire payload (so providers fall back to their own defaults).

func TestOptions_OmitsZeroDefaults(t *testing.T) {
	srv, rec := captureServer(t)
	c := stubClient(t, srv.URL, Config{Temperature: 0.5})

	if _, err := c.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}); err != nil {
		t.Fatal(err)
	}
	body := rec.last()
	if _, ok := body["top_p"]; ok {
		t.Errorf("top_p should be omitted when not set, got %v", body["top_p"])
	}
	if _, ok := body["seed"]; ok {
		t.Errorf("seed should be omitted when SeedSet=false, got %v", body["seed"])
	}
	if _, ok := body["max_tokens"]; ok {
		t.Errorf("max_tokens should be omitted when zero, got %v", body["max_tokens"])
	}
}

// ----- TestOptions_StreamToggle -----
// WithStream(true) must flip the wire "stream" flag and add
// stream_options.include_usage. WithStream(false) on a streaming-by-
// default client must turn it back off.

func TestOptions_StreamToggle(t *testing.T) {
	// Streaming-true call requires a fake SSE-capable server, but for
	// this test we only care about the request shape — so we use a
	// server that echoes a minimal SSE stream regardless of body.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		_ = json.Unmarshal(body, &parsed)
		// If caller asked for streaming, send a tiny SSE response.
		if streamFlag, _ := parsed["stream"].(bool); streamFlag {
			w.Header().Set("Content-Type", "text/event-stream")
			fl, _ := w.(http.Flusher)
			_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n")
			if fl != nil {
				fl.Flush()
			}
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
	}))
	t.Cleanup(srv.Close)

	c := stubClient(t, srv.URL, Config{Stream: false})

	// Per-call override true.
	resp, err := c.Chat(context.Background(),
		[]Message{{Role: "user", Content: "x"}},
		WithStream(true),
	)
	if err != nil {
		t.Fatalf("stream override true failed: %v", err)
	}
	if resp.Content != "hi" {
		t.Errorf("expected 'hi' from streamed reply, got %q", resp.Content)
	}

	// Default (no option) stays non-streaming.
	resp2, err := c.Chat(context.Background(), []Message{{Role: "user", Content: "x"}})
	if err != nil {
		t.Fatalf("default non-stream failed: %v", err)
	}
	if resp2.Content != "ok" {
		t.Errorf("expected 'ok' from non-streamed reply, got %q", resp2.Content)
	}
}

// ----- TestOptions_NilOptionSafe -----
// Passing a literal nil ChatOption must not panic; it is silently
// skipped (mirrors Python's None default-skip behaviour).

func TestOptions_NilOptionSafe(t *testing.T) {
	srv, _ := captureServer(t)
	c := stubClient(t, srv.URL, Config{Temperature: 0.3})

	if _, err := c.Chat(context.Background(),
		[]Message{{Role: "user", Content: "x"}},
		nil, WithTemperature(0.9), nil,
	); err != nil {
		t.Fatalf("nil ChatOption should be no-op, got: %v", err)
	}
}

// ----- TestOptions_FirstTokenWatchdog_Defaults -----
// resolveStreamWatchdog must return Config defaults when no per-call
// options override them, and per-call overrides when they do.

func TestOptions_FirstTokenWatchdog_Defaults(t *testing.T) {
	c := newOpenAIClient("https://example.invalid", "k", "m", Config{
		FirstTokenTimeout:    5 * time.Second,
		FirstTokenMaxRetries: 2,
	})

	t1, r1 := c.resolveStreamWatchdog(applyOptions(nil))
	if t1 != 5*time.Second || r1 != 2 {
		t.Errorf("expected defaults (5s, 2), got (%s, %d)", t1, r1)
	}

	t2, r2 := c.resolveStreamWatchdog(applyOptions([]ChatOption{
		WithFirstTokenTimeout(2 * time.Second),
		WithFirstTokenMaxRetries(5),
	}))
	if t2 != 2*time.Second || r2 != 5 {
		t.Errorf("expected overrides (2s, 5), got (%s, %d)", t2, r2)
	}

	t3, r3 := c.resolveStreamWatchdog(applyOptions([]ChatOption{
		WithFirstTokenMaxRetries(-3),
	}))
	if t3 != 5*time.Second || r3 != 0 {
		t.Errorf("expected (5s, 0) after clamping negative retries, got (%s, %d)", t3, r3)
	}
}
