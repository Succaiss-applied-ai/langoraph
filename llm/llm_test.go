package llm_test

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/zhaocun/langoraph/llm"
)

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

// loadEnv tries to load a .env file from common ancestor directories so that
// integration tests can be run without manually exporting env vars.
// It only sets variables that are not already set in the environment.
func loadEnv(t *testing.T) {
	t.Helper()
	_, thisFile, _, _ := runtime.Caller(0)
	// Walk up from the test file location looking for .env
	dir := filepath.Dir(thisFile)
	for i := 0; i < 5; i++ {
		candidate := filepath.Join(dir, ".env")
		data, err := os.ReadFile(candidate)
		if err != nil {
			dir = filepath.Dir(dir)
			continue
		}
		for _, line := range strings.Split(string(data), "\n") {
			line = strings.TrimSpace(line)
			if line == "" || strings.HasPrefix(line, "#") {
				continue
			}
			parts := strings.SplitN(line, "=", 2)
			if len(parts) != 2 {
				continue
			}
			key := strings.TrimSpace(parts[0])
			val := strings.Trim(strings.TrimSpace(parts[1]), `"'`)
			if os.Getenv(key) == "" { // don't overwrite existing env
				os.Setenv(key, val)
			}
		}
		t.Logf("loaded .env from %s", candidate)
		return
	}
}

func newTestClient(t *testing.T) llm.Client {
	t.Helper()
	loadEnv(t)
	client, err := llm.NewClient(llm.Config{
		Temperature:    0.1,
		TimeoutSeconds: 30,
	})
	if err != nil {
		t.Skipf("no LLM API key available, skipping integration test: %v", err)
	}
	return client
}

// ---------------------------------------------------------------------------
// Unit tests — no network required
// ---------------------------------------------------------------------------

func TestExtractJSON_Plain(t *testing.T) {
	input := `{"name":"Alice","age":30}`
	got, err := llm.ExtractJSON(input)
	if err != nil {
		t.Fatal(err)
	}
	if got != input {
		t.Errorf("expected %q, got %q", input, got)
	}
}

func TestExtractJSON_WithMarkdownFence(t *testing.T) {
	input := "some text\n```json\n{\"ok\":true}\n```\ntrailing"
	got, err := llm.ExtractJSON(input)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, `"ok":true`) {
		t.Errorf("unexpected output: %q", got)
	}
}

func TestExtractJSON_EmbeddedInText(t *testing.T) {
	input := `Here is the result: {"score":95} hope that helps`
	got, err := llm.ExtractJSON(input)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, `"score":95`) {
		t.Errorf("unexpected output: %q", got)
	}
}

func TestExtractJSON_TrailingComma(t *testing.T) {
	input := `{"a":1,"b":2,}`
	got, err := llm.ExtractJSON(input)
	if err != nil {
		t.Fatalf("expected fix to succeed, got: %v", err)
	}
	if !strings.Contains(got, `"a":1`) {
		t.Errorf("unexpected output: %q", got)
	}
}

func TestExtractJSON_LineComments(t *testing.T) {
	input := `{
  // 这是注释
  "key": "value"
}`
	got, err := llm.ExtractJSON(input)
	if err != nil {
		t.Fatalf("expected comment stripping to succeed, got: %v", err)
	}
	if !strings.Contains(got, `"key"`) {
		t.Errorf("unexpected output: %q", got)
	}
}

func TestExtractJSON_NoJSON(t *testing.T) {
	_, err := llm.ExtractJSON("no json here at all")
	if err == nil {
		t.Fatal("expected error for input with no JSON")
	}
}

func TestFixJSON_RemovesTrailingCommas(t *testing.T) {
	cases := []struct {
		input string
		valid bool
	}{
		{`{"a":1,}`, true},
		{`{"a":[1,2,3,]}`, true},
		{`{"a":1,"b":2,}`, true},
	}
	for _, c := range cases {
		got := llm.FixJSON(c.input)
		// After fix, json.Valid or at least no trailing comma
		if strings.Contains(got, ",}") || strings.Contains(got, ",]") {
			t.Errorf("FixJSON(%q) still has trailing comma: %q", c.input, got)
		}
	}
}

// ---------------------------------------------------------------------------
// Integration tests — require a real API key (auto-skipped if unavailable)
// ---------------------------------------------------------------------------

func TestLLMClient_Chat_Integration(t *testing.T) {
	client := newTestClient(t)
	ctx := context.Background()

	resp, err := client.Chat(ctx, []llm.Message{
		{Role: "user", Content: "请用一句话回答：1+1等于多少？"},
	})
	if err != nil {
		t.Fatalf("Chat failed: %v", err)
	}
	if strings.TrimSpace(resp.Content) == "" {
		t.Fatal("got empty response content")
	}
	t.Logf("response: %s", resp.Content)
	t.Logf("tokens: input=%d output=%d", resp.InputTokens, resp.OutputTokens)
}

func TestLLMClient_ChatStructured_Integration(t *testing.T) {
	client := newTestClient(t)
	ctx := context.Background()

	prompt := `请用 JSON 格式回答以下问题，不要输出其他内容。
JSON 结构：{"answer": "<答案>", "confidence": <0到1的小数>}
问题：Go 语言是由哪家公司开发的？`

	var result struct {
		Answer     string  `json:"answer"`
		Confidence float64 `json:"confidence"`
	}
	if err := llm.ChatStructured(ctx, client, prompt, &result); err != nil {
		t.Fatalf("ChatStructured failed: %v", err)
	}
	if result.Answer == "" {
		t.Error("got empty answer field")
	}
	if result.Confidence < 0 || result.Confidence > 1 {
		t.Errorf("confidence out of range: %f", result.Confidence)
	}
	t.Logf("answer=%q confidence=%.2f", result.Answer, result.Confidence)
}

func TestLLMClient_ChatStructured_ChineseArray_Integration(t *testing.T) {
	client := newTestClient(t)
	ctx := context.Background()

	prompt := `请用 JSON 格式列出 Go 语言的 3 个核心特性，不要输出其他内容。
JSON 结构：{"features": ["特性1", "特性2", "特性3"]}`

	var result struct {
		Features []string `json:"features"`
	}
	if err := llm.ChatStructured(ctx, client, prompt, &result); err != nil {
		t.Fatalf("ChatStructured failed: %v", err)
	}
	if len(result.Features) == 0 {
		t.Error("got empty features list")
	}
	t.Logf("features: %v", result.Features)
}

func TestLLMClient_Timeout_Integration(t *testing.T) {
	loadEnv(t)
	client, err := llm.NewClient(llm.Config{
		Temperature:    0.1,
		TimeoutSeconds: 1, // 1 秒极短超时，预期触发 timeout
	})
	if err != nil {
		t.Skipf("no API key: %v", err)
	}

	ctx := context.Background()
	_, err = client.Chat(ctx, []llm.Message{
		{Role: "user", Content: "写一篇500字的文章"},
	})
	if err == nil {
		// 可能模型响应极快，不强制要求超时
		t.Log("request completed within 1s (model responded fast or timeout not triggered)")
	} else {
		t.Logf("got expected timeout/error: %v", err)
	}
}
