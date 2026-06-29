package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"
)

func providerCaptureServer(t *testing.T) (*httptest.Server, *providerRecorder) {
	t.Helper()
	rec := &providerRecorder{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		rec.add(r.Header.Get("Authorization"), body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"choices":[{"message":{"content":"ok"}}],
			"usage":{"prompt_tokens":1,"completion_tokens":2}
		}`))
	}))
	t.Cleanup(srv.Close)
	return srv, rec
}

type providerRecorder struct {
	authorizations []string
	requests       []map[string]any
}

func (r *providerRecorder) add(auth string, body []byte) {
	r.authorizations = append(r.authorizations, auth)
	var parsed map[string]any
	_ = json.Unmarshal(body, &parsed)
	r.requests = append(r.requests, parsed)
}

func (r *providerRecorder) lastRequest() map[string]any {
	if len(r.requests) == 0 {
		return nil
	}
	return r.requests[len(r.requests)-1]
}

func (r *providerRecorder) lastAuthorization() string {
	if len(r.authorizations) == 0 {
		return ""
	}
	return r.authorizations[len(r.authorizations)-1]
}

func TestNewClientExplicitConfigOverridesEnv(t *testing.T) {
	t.Setenv("ARK_API_KEY", "env-ark-key")
	t.Setenv("ARK_MODEL", "env-ark-model")
	srv, rec := providerCaptureServer(t)

	client, err := NewClient(Config{
		Provider: "doubao",
		BaseURL:  srv.URL,
		APIKey:   "explicit-key",
		Model:    "explicit-model",
	})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	if _, err := client.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}); err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	if rec.lastAuthorization() != "Bearer explicit-key" {
		t.Fatalf("Authorization = %q", rec.lastAuthorization())
	}
	if got := rec.lastRequest()["model"]; got != "explicit-model" {
		t.Fatalf("model = %v, want explicit-model", got)
	}
}

func TestNewClientDoubaoAliasUsesArkEnv(t *testing.T) {
	srv, rec := providerCaptureServer(t)
	t.Setenv("ARK_API_KEY", "ark-key")
	t.Setenv("ARK_BASE_URL", srv.URL)
	t.Setenv("ARK_MODEL", "doubao-test-model")

	client, err := NewClient(Config{Provider: "doubao"})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	if _, err := client.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}); err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	if rec.lastAuthorization() != "Bearer ark-key" {
		t.Fatalf("Authorization = %q", rec.lastAuthorization())
	}
	if got := rec.lastRequest()["model"]; got != "doubao-test-model" {
		t.Fatalf("model = %v, want doubao-test-model", got)
	}
}

func TestNewClientDeepSeekUsesDashScopeCompatibleEndpoint(t *testing.T) {
	srv, rec := providerCaptureServer(t)
	t.Setenv("DASHSCOPE_API_KEY", "dashscope-key")
	t.Setenv("DASHSCOPE_BASE_URL", srv.URL)

	client, err := NewClient(Config{Provider: "deepseek"})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	if _, err := client.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}); err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	if rec.lastAuthorization() != "Bearer dashscope-key" {
		t.Fatalf("Authorization = %q", rec.lastAuthorization())
	}
	if got := rec.lastRequest()["model"]; got != "deepseek-chat" {
		t.Fatalf("model = %v, want deepseek-chat", got)
	}
}

func TestNewClientTokenPlanUsesEnv(t *testing.T) {
	srv, rec := providerCaptureServer(t)
	t.Setenv("TOKENPLAN_API_KEY", "tokenplan-key")
	t.Setenv("TOKENPLAN_BASE_URL", srv.URL)
	t.Setenv("TOKENPLAN_MODEL", "deepseek-v4-flash-202605")

	client, err := NewClient(Config{Provider: "tokenplan"})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	if _, err := client.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}); err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	if rec.lastAuthorization() != "Bearer tokenplan-key" {
		t.Fatalf("Authorization = %q", rec.lastAuthorization())
	}
	if got := rec.lastRequest()["model"]; got != "deepseek-v4-flash-202605" {
		t.Fatalf("model = %v, want deepseek-v4-flash-202605", got)
	}
}

func TestTokenPlanBaseURLSelectsTokenPlanProvider(t *testing.T) {
	p := providerForExplicitConfig("", "https://tokenhub.tencentmaas.com/plan/v3")
	if p == nil || p.name != "tokenplan" {
		t.Fatalf("provider = %#v, want tokenplan", p)
	}
}

func TestArkEndpointGetsThinkingExtensions(t *testing.T) {
	c := newOpenAIClient("https://ark.cn-beijing.volces.com/api/v3", "k", "m", Config{
		EnableThinking:  true,
		ReasoningEffort: "max",
	})
	req := c.buildRequest([]Message{{Role: "user", Content: "hi"}}, applyOptions(nil))
	if req.EnableThinking == nil || *req.EnableThinking != true {
		t.Fatalf("EnableThinking = %#v, want true", req.EnableThinking)
	}
	if req.ReasoningEffort == nil || *req.ReasoningEffort != "max" {
		t.Fatalf("ReasoningEffort = %#v, want max", req.ReasoningEffort)
	}
}

func TestTokenPlanEndpointGetsThinkingExtensions(t *testing.T) {
	c := newOpenAIClient("https://tokenhub.tencentmaas.com/plan/v3", "k", "m", Config{
		EnableThinking:  true,
		ReasoningEffort: "high",
	})
	req := c.buildRequest([]Message{{Role: "user", Content: "hi"}}, applyOptions(nil))
	if req.EnableThinking == nil || *req.EnableThinking != true {
		t.Fatalf("EnableThinking = %#v, want true", req.EnableThinking)
	}
	if req.ReasoningEffort == nil || *req.ReasoningEffort != "high" {
		t.Fatalf("ReasoningEffort = %#v, want high", req.ReasoningEffort)
	}
}

func TestLiveTokenPlanDeepSeek(t *testing.T) {
	if os.Getenv("RUN_TOKENPLAN_LIVE") != "1" {
		t.Skip("set RUN_TOKENPLAN_LIVE=1 and TOKENPLAN_API_KEY or TENCENT_TOKENPLAN_API_KEY")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()

	client, err := NewClient(Config{
		Provider:       "tokenplan",
		TimeoutSeconds: 45,
		Temperature:    0,
	})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}

	resp, err := client.Chat(ctx, []Message{{Role: "user", Content: "reply with ok"}})
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if resp == nil || resp.Content == "" {
		t.Fatalf("empty response: %#v", resp)
	}
}
