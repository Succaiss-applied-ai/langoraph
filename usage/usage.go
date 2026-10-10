// Package modelusage records provider-reported usage once per physical HTTP
// request. It never estimates counts from text or treats absent usage as zero.
package usage

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"crypto/rand"
)

const maximumCapture = 8 << 20

type Scope struct {
	RunID, ThreadID, TenantID, TaskID, Workload, TraceID, ParentCallID, DocumentID string
}

type scopeKey struct{}

func WithScope(ctx context.Context, scope Scope) context.Context {
	previous, _ := ctx.Value(scopeKey{}).(Scope)
	if scope.RunID == "" {
		scope.RunID = previous.RunID
	}
	if scope.ThreadID == "" {
		scope.ThreadID = previous.ThreadID
	}
	if scope.TenantID == "" {
		scope.TenantID = previous.TenantID
	}
	if scope.TaskID == "" {
		scope.TaskID = previous.TaskID
	}
	if scope.TraceID == "" {
		scope.TraceID = previous.TraceID
	}
	if scope.Workload == "" {
		scope.Workload = previous.Workload
	}
	if scope.ParentCallID == "" {
		scope.ParentCallID = previous.ParentCallID
	}
	if scope.DocumentID == "" {
		scope.DocumentID = previous.DocumentID
	}
	return context.WithValue(ctx, scopeKey{}, scope)
}

type Metadata struct {
	Provider, Model, Operation string
}

// Wrap copies the client, preserving its transport, redirects, cookies and
// deadlines. The observer does not drain responses or change their bytes.
func Wrap(client *http.Client, metadata Metadata, logger *slog.Logger) *http.Client {
	copy := *client
	base := copy.Transport
	if base == nil {
		base = http.DefaultTransport
	}
	if logger == nil {
		logger = slog.New(slog.NewJSONHandler(os.Stdout, nil))
	}
	copy.Transport = &transport{base: base, metadata: metadata, logger: logger}
	return &copy
}

type transport struct {
	base     http.RoundTripper
	metadata Metadata
	logger   *slog.Logger
}

func (t *transport) RoundTrip(req *http.Request) (*http.Response, error) {
	meta := t.metadata
	scope, _ := req.Context().Value(scopeKey{}).(Scope)
	// Read only a replayable copy. Request content is never retained in logs.
	if req.GetBody != nil {
		if body, err := req.GetBody(); err == nil {
			var wire struct {
				Model    string `json:"model"`
				Metadata struct {
					RunID    string `json:"run_id"`
					ThreadID string `json:"thread_id"`
				} `json:"metadata"`
			}
			if json.NewDecoder(io.LimitReader(body, maximumCapture)).Decode(&wire) == nil {
				if wire.Model != "" {
					meta.Model = wire.Model
				}
				if scope.RunID == "" {
					scope.RunID = wire.Metadata.RunID
				}
				if scope.ThreadID == "" {
					scope.ThreadID = wire.Metadata.ThreadID
				}
			}
			_ = body.Close()
		}
	}
	r := &record{logger: t.logger, ctx: req.Context(), started: time.Now(), fields: []any{
		"call_id", rand.Text(), "provider", meta.Provider, "model", meta.Model,
		"operation", meta.Operation, "run_id", scope.RunID, "thread_id", scope.ThreadID,
		"tenant_id", scope.TenantID, "task_id", scope.TaskID, "workload", scope.Workload, "trace_id", scope.TraceID, "parent_call_id", scope.ParentCallID, "document_id", scope.DocumentID,
	}}
	r.logger.InfoContext(r.ctx, "模型请求开始", append(r.fields, "event", "model.request.started")...)
	response, err := t.base.RoundTrip(req)
	if err != nil {
		r.finish("transport_error")
		return response, err
	}
	r.httpStatus = response.StatusCode
	r.requestID = response.Header.Get("x-request-id")
	for _, header := range []string{"x-dashscope-request-id", "x-acs-request-id", "x-tc-requestid", "x-tc-request-id", "request-id"} {
		if r.requestID == "" {
			r.requestID = response.Header.Get(header)
		}
	}
	if len(r.requestID) > 256 {
		r.requestID = ""
	}
	if response.Body == nil {
		r.finish("empty_body")
	} else {
		r.stream = strings.Contains(strings.ToLower(response.Header.Get("Content-Type")), "text/event-stream")
		response.Body = &bodyObserver{ReadCloser: response.Body, record: r, stream: r.stream}
	}
	return response, nil
}

type record struct {
	logger                *slog.Logger
	ctx                   context.Context
	started               time.Time
	fields                []any
	httpStatus            int
	requestID, responseID string
	usage                 map[string]any
	invalid               bool
	stream, final         bool
	once                  sync.Once
}

func (r *record) observe(data []byte) {
	if bytes.Equal(bytes.TrimSpace(data), []byte("[DONE]")) {
		r.final = true
		return
	}
	var wire struct {
		Type     string          `json:"type"`
		ID       string          `json:"id"`
		Usage    json.RawMessage `json:"usage"`
		Response struct {
			ID    string          `json:"id"`
			Usage json.RawMessage `json:"usage"`
		} `json:"response"`
	}
	if json.Unmarshal(data, &wire) != nil {
		return
	}
	if wire.Type == "response.completed" || wire.Type == "response.incomplete" || wire.Type == "response.failed" {
		r.final = true
	}
	if wire.ID != "" && len(wire.ID) <= 256 {
		r.responseID = wire.ID
	}
	if wire.Response.ID != "" && len(wire.Response.ID) <= 256 {
		r.responseID = wire.Response.ID
	}
	raw := wire.Usage
	if len(wire.Response.Usage) > 0 {
		raw = wire.Response.Usage
	}
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return
	}
	var usage map[string]any
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	if decoder.Decode(&usage) != nil {
		r.invalid = true
		return
	}
	// Preserve all numeric provider usage details, including audio/image/cache
	// breakdowns, but never strings, model output or arbitrary error bodies.
	r.usage, r.invalid = numericUsage(usage, 0)
}

func numericUsage(value map[string]any, depth int) (map[string]any, bool) {
	out := map[string]any{}
	invalid := false
	if depth > 8 {
		return out, true
	}
	for key, item := range value {
		if len(key) > 80 {
			continue
		}
		safe := true
		for _, c := range key {
			if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '_') {
				safe = false
			}
		}
		if !safe {
			continue
		}
		switch item := item.(type) {
		case json.Number:
			n, err := item.Int64()
			if err != nil || n < 0 {
				invalid = true
			} else {
				out[key] = n
			}
		case map[string]any:
			nested, bad := numericUsage(item, depth+1)
			invalid = invalid || bad
			if len(nested) > 0 {
				out[key] = nested
			}
		case nil:
		default:
			// Non-numeric token counters are invalid, not zero.
			if strings.Contains(key, "token") {
				invalid = true
			}
		}
	}
	return out, invalid
}

func counter(value map[string]any, paths ...string) any {
	for _, path := range paths {
		var item any = value
		for _, part := range strings.Split(path, ".") {
			object, ok := item.(map[string]any)
			if !ok {
				item = nil
				break
			}
			item = object[part]
		}
		if n, ok := item.(int64); ok {
			return n
		}
	}
	return nil
}

func (r *record) finish(bodyState string) {
	r.once.Do(func() {
		input := counter(r.usage, "prompt_tokens", "input_tokens")
		output := counter(r.usage, "completion_tokens", "output_tokens")
		total := counter(r.usage, "total_tokens")
		state := "unknown"
		if len(r.usage) > 0 {
			state = "partial"
			if total != nil && (input != nil && output != nil || strings.HasPrefix(r.operation(), "embedding")) {
				state = "reported"
			}
		}
		if state == "reported" && r.stream && !r.final {
			state = "partial"
		}
		if r.invalid {
			state = "invalid"
		}
		fields := append(append([]any{}, r.fields...),
			"event", "model.token_usage", "usage_source", "provider_response", "usage_status", state,
			"input_tokens", input, "output_tokens", output, "total_tokens", total,
			"cached_input_tokens", counter(r.usage, "prompt_cache_hit_tokens", "prompt_tokens_details.cached_tokens", "input_tokens_details.cached_tokens"),
			"cache_miss_input_tokens", counter(r.usage, "prompt_cache_miss_tokens"),
			"reasoning_tokens", counter(r.usage, "completion_tokens_details.reasoning_tokens", "output_tokens_details.reasoning_tokens", "reasoning_tokens"),
			"provider_usage", r.usage, "provider_request_id", r.requestID, "provider_response_id", r.responseID,
			"http_status", r.httpStatus, "body_state", bodyState, "usage_final", !r.stream || r.final, "duration_ms", time.Since(r.started).Milliseconds(),
		)
		r.logger.InfoContext(r.ctx, "模型Token用量", fields...)
	})
}

func (r *record) operation() string {
	for i := 0; i+1 < len(r.fields); i += 2 {
		if r.fields[i] == "operation" {
			v, _ := r.fields[i+1].(string)
			return v
		}
	}
	return ""
}

type bodyObserver struct {
	io.ReadCloser
	mu            sync.Mutex
	record        *record
	stream        bool
	buffer, event []byte
	overflow      bool
}

func (b *bodyObserver) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	b.mu.Lock()
	defer b.mu.Unlock()
	if n > 0 {
		b.capture(p[:n])
	}
	if err != nil {
		b.flush()
		state := "read_error"
		if err == io.EOF {
			state = "eof"
		}
		b.record.finish(state)
	}
	return n, err
}

func (b *bodyObserver) capture(data []byte) {
	if b.overflow {
		return
	}
	if len(b.buffer)+len(data) > maximumCapture {
		b.overflow = true
		b.buffer = nil
		b.record.invalid = true
		return
	}
	b.buffer = append(b.buffer, data...)
	if !b.stream {
		return
	}
	for {
		index := bytes.IndexByte(b.buffer, '\n')
		if index < 0 {
			break
		}
		line := bytes.TrimSuffix(b.buffer[:index], []byte("\r"))
		b.line(line)
		b.buffer = b.buffer[index+1:]
	}
}

func (b *bodyObserver) line(line []byte) {
	if len(line) == 0 {
		b.record.observe(b.event)
		b.event = nil
		return
	}
	if bytes.HasPrefix(line, []byte("data:")) {
		data := bytes.TrimPrefix(line[5:], []byte(" "))
		if len(b.event)+len(data)+1 > maximumCapture {
			b.record.invalid = true
			b.event = nil
			return
		}
		if len(b.event) > 0 {
			b.event = append(b.event, '\n')
		}
		b.event = append(b.event, data...)
	}
}

func (b *bodyObserver) flush() {
	if b.stream {
		if len(b.buffer) > 0 {
			b.line(bytes.TrimSuffix(b.buffer, []byte("\r")))
		}
		b.record.observe(b.event)
	} else {
		b.record.observe(b.buffer)
	}
	b.buffer, b.event = nil, nil
}

func (b *bodyObserver) Close() error {
	err := b.ReadCloser.Close()
	b.mu.Lock()
	defer b.mu.Unlock()
	b.flush()
	b.record.finish("closed")
	return err
}
