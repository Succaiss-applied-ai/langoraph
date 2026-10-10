package usage

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"testing"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func records(t *testing.T, buffer *bytes.Buffer) []map[string]any {
	t.Helper()
	var out []map[string]any
	decoder := json.NewDecoder(bytes.NewReader(buffer.Bytes()))
	decoder.UseNumber()
	for decoder.More() {
		var value map[string]any
		if err := decoder.Decode(&value); err != nil {
			t.Fatal(err)
		}
		out = append(out, value)
	}
	return out
}

func TestProviderCountsAndMissingAreDifferent(t *testing.T) {
	for _, tc := range []struct {
		name, response, status, operation string
		total                             any
	}{
		{"reported-zero", `{"id":"r","usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}`, "reported", "chat", json.Number("0")},
		{"missing", `{"id":"r","choices":[]}`, "unknown", "chat", nil},
		{"null", `{"usage":null}`, "unknown", "chat", nil},
		{"partial", `{"usage":{"prompt_tokens":19}}`, "partial", "chat", nil},
		{"embedding", `{"usage":{"total_tokens":33}}`, "reported", "embedding", json.Number("33")},
		{"invalid", `{"usage":{"prompt_tokens":"42","total_tokens":-1}}`, "invalid", "chat", nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var logs bytes.Buffer
			client := Wrap(&http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": []string{"application/json"}}, Body: io.NopCloser(strings.NewReader(tc.response))}, nil
			})}, Metadata{Provider: "test", Model: "m", Operation: tc.operation}, slog.New(slog.NewJSONHandler(&logs, nil)))
			req, _ := http.NewRequestWithContext(WithScope(context.Background(), Scope{RunID: "run", ThreadID: "thread"}), "POST", "https://provider.invalid/chat", strings.NewReader(`{"model":"actual-model","messages":[{"content":"SECRET-PROMPT"}]}`))
			req.Header.Set("Authorization", "Bearer SECRET-KEY")
			resp, err := client.Do(req)
			if err != nil {
				t.Fatal(err)
			}
			got, err := io.ReadAll(resp.Body)
			if err != nil || string(got) != tc.response {
				t.Fatalf("response was changed: %s, %v", got, err)
			}
			_ = resp.Body.Close()
			_ = resp.Body.Close()
			out := records(t, &logs)
			if len(out) != 2 {
				t.Fatalf("one start and one terminal record required, got %v", out)
			}
			u := out[1]
			if u["usage_status"] != tc.status || u["total_tokens"] != tc.total || u["run_id"] != "run" || u["model"] != "actual-model" {
				t.Fatalf("wrong usage: %v", u)
			}
			if out[0]["call_id"] != u["call_id"] || strings.Contains(logs.String(), "SECRET") {
				t.Fatal("correlation lost or request content leaked")
			}
		})
	}
}

func TestSSELatestUsageWithoutDoubleCounting(t *testing.T) {
	stream := "data: {\"id\":\"r\",\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":3,\"total_tokens\":15}}\r\n\r\n" +
		"data: {\"id\":\"r\",\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":5,\"total_tokens\":17,\"prompt_tokens_details\":{\"cached_tokens\":8},\"completion_tokens_details\":{\"reasoning_tokens\":2},\"debug\":\"SECRET-OUTPUT\"}}\n\n" +
		"data: [DONE]\n\n"
	var logs bytes.Buffer
	client := Wrap(&http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": []string{"text/event-stream"}}, Body: io.NopCloser(&oneByteReader{Reader: strings.NewReader(stream)})}, nil
	})}, Metadata{Operation: "chat"}, slog.New(slog.NewJSONHandler(&logs, nil)))
	req, _ := http.NewRequest("POST", "https://provider.invalid", nil)
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	got, err := io.ReadAll(resp.Body)
	if err != nil || string(got) != stream {
		t.Fatal("SSE bytes changed")
	}
	_ = resp.Body.Close()
	u := records(t, &logs)[1]
	if u["total_tokens"] != json.Number("17") || u["reasoning_tokens"] != json.Number("2") || u["cached_input_tokens"] != json.Number("8") || strings.Contains(logs.String(), "SECRET") {
		t.Fatalf("usage repeated or leaked: %v", u)
	}
}

type oneByteReader struct{ io.Reader }

func (r *oneByteReader) Read(p []byte) (int, error) { return r.Reader.Read(p[:1]) }

func TestResponsesAPIAndTransportFailure(t *testing.T) {
	var logs bytes.Buffer
	count := 0
	client := Wrap(&http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		count++
		if count == 1 {
			return nil, errors.New("SECRET-TRANSPORT-ERROR")
		}
		return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": []string{"text/event-stream"}}, Body: io.NopCloser(strings.NewReader("data: {\"type\":\"response.completed\",\"response\":{\"id\":\"r\",\"usage\":{\"input_tokens\":10,\"output_tokens\":6,\"total_tokens\":16,\"output_tokens_details\":{\"reasoning_tokens\":4}}}}\n\n"))}, nil
	})}, Metadata{Operation: "responses"}, slog.New(slog.NewJSONHandler(&logs, nil)))
	for i := 0; i < 2; i++ {
		req, _ := http.NewRequest("POST", "https://provider.invalid", nil)
		resp, err := client.Do(req)
		if err == nil {
			_, _ = io.Copy(io.Discard, resp.Body)
			_ = resp.Body.Close()
		}
	}
	out := records(t, &logs)
	if len(out) != 4 || out[1]["usage_status"] != "unknown" || out[1]["input_tokens"] != nil || out[3]["total_tokens"] != json.Number("16") || out[0]["call_id"] == out[2]["call_id"] || strings.Contains(logs.String(), "SECRET") {
		t.Fatalf("lost attempts: %v", out)
	}
}

func TestCloseBeforeUsageIsUnknown(t *testing.T) {
	var logs bytes.Buffer
	client := Wrap(&http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": []string{"text/event-stream"}}, Body: io.NopCloser(strings.NewReader("data: {\"usage\":{\"prompt_tokens\":100}}\n\n"))}, nil
	})}, Metadata{}, slog.New(slog.NewJSONHandler(&logs, nil)))
	req, _ := http.NewRequest("POST", "https://provider.invalid", nil)
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	_ = resp.Body.Close()
	u := records(t, &logs)[1]
	if u["usage_status"] != "unknown" || u["total_tokens"] != nil || u["body_state"] != "closed" {
		t.Fatalf("early close invented usage: %v", u)
	}
}
