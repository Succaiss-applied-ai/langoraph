// SSE streaming tests with a fake httptest server. Covers:
//
//   - Happy-path: content deltas concatenate, usage is captured.
//   - Reasoning heartbeat: a reasoning_content delta disarms the
//     first-token watchdog so subsequent content arrives without a
//     timeout, mirroring DashScope thinking-mode behaviour.
//   - First-token timeout: stalled upstream surfaces as the typed
//     FirstTokenTimeoutError.
//   - First-token retry budget: WithFirstTokenMaxRetries(N) opens N+1
//     SSE attempts before bubbling the timeout up.
//   - Concurrent streaming: many goroutines streaming simultaneously
//     against the same client share the keep-alive pool without
//     interfering — exercised under -race.
package llm

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// streamingFlusher writes SSE events with controllable inter-event
// delays. Helpers below build common scenarios on top of it.
type streamingFlusher struct {
	w  http.ResponseWriter
	fl http.Flusher
}

func newSSEWriter(w http.ResponseWriter) *streamingFlusher {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	fl, _ := w.(http.Flusher)
	if fl != nil {
		fl.Flush()
	}
	return &streamingFlusher{w: w, fl: fl}
}

func (s *streamingFlusher) write(payload string) {
	_, _ = io.WriteString(s.w, "data: "+payload+"\n\n")
	if s.fl != nil {
		s.fl.Flush()
	}
}

func (s *streamingFlusher) done() {
	_, _ = io.WriteString(s.w, "data: [DONE]\n\n")
	if s.fl != nil {
		s.fl.Flush()
	}
}

func contentDelta(text string) string {
	return fmt.Sprintf(`{"choices":[{"index":0,"delta":{"content":%q}}]}`, text)
}

func reasoningDelta(text string) string {
	return fmt.Sprintf(`{"choices":[{"index":0,"delta":{"reasoning_content":%q}}]}`, text)
}

func usageOnly(prompt, completion, reasoning int) string {
	return fmt.Sprintf(
		`{"choices":[],"usage":{"prompt_tokens":%d,"completion_tokens":%d,"completion_tokens_details":{"reasoning_tokens":%d}}}`,
		prompt, completion, reasoning,
	)
}

// streamServer wires a handler that streams a sequence of events with
// configurable per-event delays.
func streamServer(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	return srv
}

// ----- TestStreaming_HappyPath -----

func TestStreaming_HappyPath(t *testing.T) {
	srv := streamServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("X-DashScope-Request-Id", "stream-request-123")
		s := newSSEWriter(w)
		s.write(contentDelta("Hello"))
		s.write(contentDelta(", "))
		s.write(contentDelta("world!"))
		s.write(usageOnly(11, 22, 0))
		s.done()
	})

	c := stubClient(t, srv.URL, Config{
		Stream:               true,
		FirstTokenTimeout:    2 * time.Second,
		FirstTokenMaxRetries: 0,
	})

	resp, err := c.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}})
	if err != nil {
		t.Fatalf("happy-path stream failed: %v", err)
	}
	if resp.Content != "Hello, world!" {
		t.Errorf("content: expected 'Hello, world!', got %q", resp.Content)
	}
	if resp.InputTokens != 11 || resp.OutputTokens != 22 {
		t.Errorf("usage: expected (11,22), got (%d,%d)", resp.InputTokens, resp.OutputTokens)
	}
	if resp.ProviderRequestID != "stream-request-123" {
		t.Errorf("ProviderRequestID: expected stream-request-123, got %q", resp.ProviderRequestID)
	}
}

// ----- TestStreaming_ReasoningHeartbeat -----
// A reasoning_content delta arriving before any content delta must
// disarm the watchdog so a slow content phase still succeeds.

func TestStreaming_ReasoningHeartbeat(t *testing.T) {
	srv := streamServer(t, func(w http.ResponseWriter, _ *http.Request) {
		s := newSSEWriter(w)
		// Send reasoning heartbeat shortly after open.
		time.Sleep(80 * time.Millisecond)
		s.write(reasoningDelta("thinking..."))
		// Then content arrives well after the original 200ms watchdog
		// window — heartbeat should have disarmed it.
		time.Sleep(400 * time.Millisecond)
		s.write(contentDelta("answer"))
		s.write(usageOnly(7, 8, 9))
		s.done()
	})

	c := stubClient(t, srv.URL, Config{
		Stream:               true,
		FirstTokenTimeout:    200 * time.Millisecond,
		FirstTokenMaxRetries: 0,
	})

	resp, err := c.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}})
	if err != nil {
		t.Fatalf("reasoning heartbeat path failed: %v", err)
	}
	if resp.Content != "answer" {
		t.Errorf("content: expected 'answer', got %q", resp.Content)
	}
	if resp.ThinkingContent != "thinking..." {
		t.Errorf("thinking: expected 'thinking...', got %q", resp.ThinkingContent)
	}
	if resp.ReasoningTokens != 9 {
		t.Errorf("reasoning_tokens: expected 9, got %d", resp.ReasoningTokens)
	}
}

// ----- TestStreaming_FirstTokenTimeout -----

func TestStreaming_FirstTokenTimeout(t *testing.T) {
	// Connect, then never send any data — watchdog must fire.
	srv := streamServer(t, func(w http.ResponseWriter, r *http.Request) {
		_ = newSSEWriter(w)
		select {
		case <-time.After(2 * time.Second):
		case <-r.Context().Done():
		}
	})

	c := stubClient(t, srv.URL, Config{
		Stream:               true,
		FirstTokenTimeout:    150 * time.Millisecond,
		FirstTokenMaxRetries: 0,
	})

	_, err := c.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}})
	if err == nil {
		t.Fatal("expected first-token timeout, got nil")
	}
	var ftt *FirstTokenTimeoutError
	if !errors.As(err, &ftt) {
		t.Fatalf("expected *FirstTokenTimeoutError, got %T: %v", err, err)
	}
	if ftt.Budget != 150*time.Millisecond {
		t.Errorf("Budget: expected 150ms, got %s", ftt.Budget)
	}
}

// ----- TestStreaming_FirstTokenTimeout_RetryBudget -----
// WithFirstTokenMaxRetries(N) → exactly N+1 SSE attempts before the
// typed timeout bubbles up.

func TestStreaming_FirstTokenTimeout_RetryBudget(t *testing.T) {
	var attempts int32
	srv := streamServer(t, func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&attempts, 1)
		_ = newSSEWriter(w)
		select {
		case <-time.After(2 * time.Second):
		case <-r.Context().Done():
		}
	})

	c := stubClient(t, srv.URL, Config{
		Stream:               true,
		FirstTokenTimeout:    100 * time.Millisecond,
		FirstTokenMaxRetries: 2, // 1 initial + 2 retries = 3 attempts
	})

	_, err := c.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}})
	var ftt *FirstTokenTimeoutError
	if !errors.As(err, &ftt) {
		t.Fatalf("expected *FirstTokenTimeoutError, got %T: %v", err, err)
	}
	if got := atomic.LoadInt32(&attempts); got != 3 {
		t.Errorf("expected 3 SSE attempts, got %d", got)
	}
}

// ----- TestStreaming_FirstTokenTimeout_PerCallOverride -----
// WithFirstTokenTimeout(5ms) overrides a generous Config default.

func TestStreaming_FirstTokenTimeout_PerCallOverride(t *testing.T) {
	srv := streamServer(t, func(w http.ResponseWriter, r *http.Request) {
		_ = newSSEWriter(w)
		select {
		case <-time.After(2 * time.Second):
		case <-r.Context().Done():
		}
	})

	c := stubClient(t, srv.URL, Config{
		Stream:            true,
		FirstTokenTimeout: 10 * time.Second,
	})

	start := time.Now()
	_, err := c.Chat(context.Background(),
		[]Message{{Role: "user", Content: "hi"}},
		WithFirstTokenTimeout(80*time.Millisecond),
	)
	elapsed := time.Since(start)

	var ftt *FirstTokenTimeoutError
	if !errors.As(err, &ftt) {
		t.Fatalf("expected first-token timeout, got: %v", err)
	}
	if elapsed > 1*time.Second {
		t.Errorf("expected fast timeout via per-call override, took %s", elapsed)
	}
}

// ----- TestStreaming_ChunkIdleTimeout -----
// Once the stream is established, any long gap between two chunks must
// fail fast instead of waiting for the total request deadline.

func TestStreaming_ChunkIdleTimeout(t *testing.T) {
	srv := streamServer(t, func(w http.ResponseWriter, r *http.Request) {
		s := newSSEWriter(w)
		s.write(contentDelta("first"))
		select {
		case <-time.After(2 * time.Second):
		case <-r.Context().Done():
		}
	})

	c := stubClient(t, srv.URL, Config{
		Stream:               true,
		FirstTokenTimeout:    time.Second,
		FirstTokenMaxRetries: 0,
		ChunkIdleTimeout:     120 * time.Millisecond,
	})

	start := time.Now()
	_, err := c.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}})
	elapsed := time.Since(start)

	var idle *ChunkIdleTimeoutError
	if !errors.As(err, &idle) {
		t.Fatalf("expected *ChunkIdleTimeoutError, got %T: %v", err, err)
	}
	if idle.Budget != 120*time.Millisecond {
		t.Errorf("Budget: expected 120ms, got %s", idle.Budget)
	}
	if elapsed > time.Second {
		t.Errorf("expected fast idle timeout, took %s", elapsed)
	}
}

// ----- TestStreaming_NoTimeoutWhenDisabled -----
// FirstTokenTimeout<=0 means watchdog is disabled; the request proceeds
// (and finishes when the upstream sends [DONE]) without any timeout.

func TestStreaming_NoTimeoutWhenDisabled(t *testing.T) {
	srv := streamServer(t, func(w http.ResponseWriter, _ *http.Request) {
		s := newSSEWriter(w)
		// Big delay before first content — no watchdog must trip.
		time.Sleep(150 * time.Millisecond)
		s.write(contentDelta("ok"))
		s.done()
	})

	c := stubClient(t, srv.URL, Config{Stream: true, FirstTokenTimeout: 0})
	resp, err := c.Chat(context.Background(), []Message{{Role: "user", Content: "x"}})
	if err != nil {
		t.Fatalf("disabled-watchdog stream failed: %v", err)
	}
	if resp.Content != "ok" {
		t.Errorf("content: expected 'ok', got %q", resp.Content)
	}
}

// ----- TestStreaming_ConcurrentSafety -----
// 8 goroutines stream simultaneously through one client; -race must
// not fire and every reply must be intact.

func TestStreaming_ConcurrentSafety(t *testing.T) {
	srv := streamServer(t, func(w http.ResponseWriter, r *http.Request) {
		s := newSSEWriter(w)
		s.write(contentDelta("payload-"))
		s.write(contentDelta(r.URL.Path))
		s.done()
	})

	c := stubClient(t, srv.URL, Config{Stream: true, FirstTokenTimeout: 1 * time.Second})

	const n = 8
	var wg sync.WaitGroup
	results := make([]string, n)
	errs := make([]error, n)
	for i := 0; i < n; i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			resp, err := c.Chat(context.Background(), []Message{{Role: "user", Content: "x"}})
			if err != nil {
				errs[i] = err
				return
			}
			results[i] = resp.Content
		}()
	}
	wg.Wait()
	for i := 0; i < n; i++ {
		if errs[i] != nil {
			t.Errorf("worker %d failed: %v", i, errs[i])
		}
		if results[i] != "payload-/chat/completions" {
			t.Errorf("worker %d: expected 'payload-/chat/completions', got %q", i, results[i])
		}
	}
}

// ----- TestStreaming_ChatSchema_Streaming -----
// ChatSchema honours streaming when WithStream(true) is passed.

func TestStreaming_ChatSchema_Streaming(t *testing.T) {
	srv := streamServer(t, func(w http.ResponseWriter, _ *http.Request) {
		s := newSSEWriter(w)
		s.write(contentDelta(`{"a":1}`))
		s.done()
	})

	c := stubClient(t, srv.URL, Config{Stream: true, FirstTokenTimeout: 1 * time.Second})

	resp, err := c.ChatSchema(context.Background(),
		[]Message{{Role: "user", Content: "x"}},
		"AnswerSchema",
		map[string]any{"type": "object"},
	)
	if err != nil {
		t.Fatalf("streaming ChatSchema failed: %v", err)
	}
	if resp.Content != `{"a":1}` {
		t.Errorf("content: expected JSON, got %q", resp.Content)
	}
}
