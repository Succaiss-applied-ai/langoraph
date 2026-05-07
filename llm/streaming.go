package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// FirstTokenTimeoutError is returned by the streaming chat path when
// the watchdog fires before any content (or reasoning_content) delta
// arrives. It is a TimeoutError so callers can use ``errors.Is(err,
// context.DeadlineExceeded)``-style probing without a special branch.
//
// Mirrors Python's ``LLMFirstTokenTimeout`` from
// ``app/llm/dashscope_client.py``.
type FirstTokenTimeoutError struct {
	Budget time.Duration
	Model  string
}

func (e *FirstTokenTimeoutError) Error() string {
	return fmt.Sprintf("llm: first content token did not arrive within %s (model=%s)", e.Budget, e.Model)
}

// Timeout reports true so this error type integrates with net.Error /
// generic timeout-detection callers without an explicit type assertion.
func (e *FirstTokenTimeoutError) Timeout() bool { return true }

// streamChunk is the OpenAI/DashScope SSE delta wire shape we care about.
type streamChunk struct {
	Choices []struct {
		Index int `json:"index"`
		Delta struct {
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
			Role             string `json:"role"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens            int `json:"prompt_tokens"`
		CompletionTokens        int `json:"completion_tokens"`
		TotalTokens             int `json:"total_tokens"`
		CompletionTokensDetails *struct {
			ReasoningTokens int `json:"reasoning_tokens"`
		} `json:"completion_tokens_details"`
	} `json:"usage"`
}

// chatStream opens a streaming chat-completions SSE connection and
// collects the full reply, applying a first-token watchdog and an
// in-place retry budget on first-token timeouts.
//
// Goroutine safety: every call constructs its own request body, HTTP
// request, watchdog timer and SSE parser state — concurrent invocations
// share only the package-level Transport / connection pool.
func (c *openAIClient) chatStream(
	ctx context.Context,
	req chatRequest,
	firstTokenTimeout time.Duration,
	firstTokenMaxRetries int,
) (*Response, error) {
	maxAttempts := firstTokenMaxRetries + 1
	if maxAttempts < 1 {
		maxAttempts = 1
	}

	var lastTimeout *FirstTokenTimeoutError
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		resp, err := c.streamOnce(ctx, req, firstTokenTimeout)
		if err == nil {
			return resp, nil
		}
		var ftt *FirstTokenTimeoutError
		if errors.As(err, &ftt) {
			lastTimeout = ftt
			if attempt < maxAttempts {
				// Honour external cancellation between attempts.
				if cerr := ctx.Err(); cerr != nil {
					return nil, cerr
				}
				continue
			}
			return nil, ftt
		}
		return nil, err
	}
	// Defensive: loop only exits via return.
	if lastTimeout != nil {
		return nil, lastTimeout
	}
	return nil, fmt.Errorf("llm: streaming exited without a result (unexpected)")
}

// streamOnce performs one SSE round-trip, returning either a populated
// Response, a *FirstTokenTimeoutError, or any other transport / parse
// error.
func (c *openAIClient) streamOnce(
	ctx context.Context,
	req chatRequest,
	firstTokenTimeout time.Duration,
) (*Response, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("llm: marshal stream request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("llm: build stream request: %w", err)
	}
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		// A timeout that bites before the first byte (i.e. no SSE
		// connection ever opened) is reported as a first-token
		// timeout so callers' retry budget kicks in identically to
		// the "connected but no delta" branch below.
		if isFirstTokenTimeoutErr(ctx, err, firstTokenTimeout) {
			return nil, &FirstTokenTimeoutError{Budget: firstTokenTimeout, Model: c.model}
		}
		return nil, fmt.Errorf("llm: http stream request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("llm: provider returned %d: %s", resp.StatusCode, string(raw))
	}

	wd := newFirstTokenWatchdog(resp.Body, firstTokenTimeout)
	defer wd.stop()

	parts, reasoningParts, usage, parseErr := parseSSEStream(resp.Body, wd)

	// Watchdog wins: surface as a typed first-token timeout. The CAS
	// inside the watchdog guarantees exactly one of {timer fires,
	// first delta seen} succeeds, so this branch is unambiguous.
	if wd.timedOut() {
		return nil, &FirstTokenTimeoutError{Budget: firstTokenTimeout, Model: c.model}
	}

	if parseErr != nil && !errors.Is(parseErr, io.EOF) {
		// The watchdog might have closed the body mid-read after first
		// token arrived — that should never happen because we stop the
		// timer on first-token success, but guard for it.
		if errors.Is(parseErr, io.ErrUnexpectedEOF) || errors.Is(parseErr, errBodyClosed) {
			if wd.timedOut() {
				return nil, &FirstTokenTimeoutError{Budget: firstTokenTimeout, Model: c.model}
			}
		}
		return nil, fmt.Errorf("llm: stream read: %w", parseErr)
	}

	r := &Response{
		Content:         strings.Join(parts, ""),
		ThinkingContent: strings.Join(reasoningParts, ""),
	}
	if usage != nil {
		r.InputTokens = usage.PromptTokens
		r.OutputTokens = usage.CompletionTokens
		if usage.CompletionTokensDetails != nil {
			r.ReasoningTokens = usage.CompletionTokensDetails.ReasoningTokens
		}
	}
	return r, nil
}

// errBodyClosed is the sentinel we surface when a watchdog-triggered
// body Close races with the SSE parser's Read.
var errBodyClosed = errors.New("llm: response body closed by watchdog")

// firstTokenWatchdog aborts the SSE response body if the first content
// (or reasoning_content) delta does not arrive within ``timeout``.
// Exactly one of {timer fires, first-token-arrived} wins via an atomic
// CAS, so callers cannot observe a half-armed state.
type firstTokenWatchdog struct {
	timer    *time.Timer
	armed    atomic.Bool // true while the timer is still racing
	tripped  atomic.Bool // true after the timer fired (first-token timeout)
	body     io.Closer
	disabled bool
	once     sync.Once
}

// newFirstTokenWatchdog returns a watchdog that will Close ``body``
// after ``timeout`` unless ``signalFirstToken`` is called first.
// A non-positive timeout disables the watchdog entirely (all signal
// calls become no-ops, ``timedOut`` is always false).
func newFirstTokenWatchdog(body io.Closer, timeout time.Duration) *firstTokenWatchdog {
	wd := &firstTokenWatchdog{body: body}
	if timeout <= 0 {
		wd.disabled = true
		return wd
	}
	wd.armed.Store(true)
	wd.timer = time.AfterFunc(timeout, func() {
		// Exactly one of {timer fires, signalFirstToken} wins.
		if !wd.armed.CompareAndSwap(true, false) {
			return
		}
		wd.tripped.Store(true)
		_ = body.Close()
	})
	return wd
}

// signalFirstToken disarms the watchdog. Idempotent and goroutine-safe.
func (w *firstTokenWatchdog) signalFirstToken() {
	if w.disabled {
		return
	}
	if !w.armed.CompareAndSwap(true, false) {
		return
	}
	if w.timer != nil {
		w.timer.Stop()
	}
}

// stop releases watchdog resources after the request lifecycle
// completes. Calling it more than once is safe.
func (w *firstTokenWatchdog) stop() {
	w.once.Do(func() {
		if w.timer != nil {
			w.timer.Stop()
		}
	})
}

// timedOut reports whether the watchdog fired before the first token.
func (w *firstTokenWatchdog) timedOut() bool {
	if w.disabled {
		return false
	}
	return w.tripped.Load()
}

// isFirstTokenTimeoutErr reports whether a transport error captured
// before any SSE delta was received should be treated as a first-token
// timeout. We only honour this when a positive timeout is configured;
// otherwise the caller wanted the raw transport error to surface.
func isFirstTokenTimeoutErr(ctx context.Context, err error, timeout time.Duration) bool {
	if timeout <= 0 || err == nil {
		return false
	}
	if ctx.Err() != nil {
		// External cancellation; not a first-token timeout.
		return false
	}
	var ne interface{ Timeout() bool }
	if errors.As(err, &ne) && ne.Timeout() {
		return true
	}
	return false
}

// parseSSEStream consumes the OpenAI / DashScope event-stream wire
// format and returns:
//
//   - parts: every ``delta.content`` chunk, in arrival order, ready to
//     join into the final response Content.
//   - reasoningParts: every ``delta.reasoning_content`` chunk for the
//     thinking-model trace (kept separate so callers don't accidentally
//     concatenate them into the user-facing Content).
//   - usage: the populated usage block from the final usage-only chunk
//     (DashScope sends one when ``stream_options.include_usage=true``);
//     nil when the provider does not emit it.
//
// The watchdog is disarmed the first time **either** a content delta
// or a reasoning_content delta is observed — matching the Python
// "thinking-model heartbeat" path so deepseek-v4-pro / qwen-deepseek
// thinking bursts do not falsely time out.
func parseSSEStream(body io.Reader, wd *firstTokenWatchdog) (
	parts []string, reasoningParts []string, usage *streamChunkUsage, err error,
) {
	scanner := bufio.NewScanner(body)
	// Reasoning-mode chunks can be large (10s of KB per delta on
	// thinking models). 1MB per line is a generous ceiling that still
	// catches runaway responses.
	scanner.Buffer(make([]byte, 0, 64*1024), 1<<20)

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" || payload == "[DONE]" {
			if payload == "[DONE]" {
				break
			}
			continue
		}
		var chunk streamChunk
		if jerr := json.Unmarshal([]byte(payload), &chunk); jerr != nil {
			// A single malformed chunk is non-fatal — providers
			// occasionally interleave keep-alive comments. Drop it.
			continue
		}
		// Usage-only chunk: choices is empty, usage is populated.
		if chunk.Usage != nil {
			u := streamChunkUsage{
				PromptTokens:     chunk.Usage.PromptTokens,
				CompletionTokens: chunk.Usage.CompletionTokens,
				TotalTokens:      chunk.Usage.TotalTokens,
			}
			if chunk.Usage.CompletionTokensDetails != nil {
				u.CompletionTokensDetails = &struct {
					ReasoningTokens int
				}{ReasoningTokens: chunk.Usage.CompletionTokensDetails.ReasoningTokens}
			}
			usage = &u
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		delta := chunk.Choices[0].Delta
		if delta.Content != "" {
			wd.signalFirstToken()
			parts = append(parts, delta.Content)
		} else if delta.ReasoningContent != "" {
			// Thinking-model heartbeat: any non-empty reasoning delta
			// proves the upstream is alive even though no content has
			// arrived yet. Disarm the watchdog so reasoning bursts do
			// not falsely time out.
			wd.signalFirstToken()
			reasoningParts = append(reasoningParts, delta.ReasoningContent)
		} else if chunk.Choices[0].FinishReason != "" {
			// Some providers emit a finish_reason without ever sending
			// a content delta (e.g. tool-only responses or empty
			// answers). Treat that as "first token received" so we
			// don't time out on a legitimately empty completion.
			wd.signalFirstToken()
		}
	}
	if serr := scanner.Err(); serr != nil {
		// Watchdog-triggered Close shows up as "use of closed network
		// connection" or similar — fold it into a sentinel so the
		// outer chatStream can branch on the typed timeout cleanly.
		if isClosedConnError(serr) {
			return parts, reasoningParts, usage, errBodyClosed
		}
		return parts, reasoningParts, usage, serr
	}
	return parts, reasoningParts, usage, nil
}

// streamChunkUsage is a lightweight private mirror of the wire usage
// shape so we don't expose chatResponse internals via parseSSEStream.
type streamChunkUsage struct {
	PromptTokens            int
	CompletionTokens        int
	TotalTokens             int
	CompletionTokensDetails *struct {
		ReasoningTokens int
	}
}

func isClosedConnError(err error) bool {
	if err == nil {
		return false
	}
	s := err.Error()
	return strings.Contains(s, "use of closed network connection") ||
		strings.Contains(s, "http: read on closed response body") ||
		strings.Contains(s, "context canceled")
}
