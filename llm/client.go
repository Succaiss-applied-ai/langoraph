// Package llm provides a unified LLM client interface and OpenAI-compatible
// implementations for DashScope (Qwen), DeepSeek, and OpenAI.
//
// All providers share the same OpenAI Chat Completions wire format;
// only the base URL and API key differ.
package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"
)

// Message is a single chat turn.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Response is the parsed LLM reply.
type Response struct {
	Content         string
	ThinkingContent string // reasoning_content from Qwen thinking mode
	InputTokens     int
	OutputTokens    int
	ReasoningTokens int
}

// Client is the interface every LLM provider must satisfy.
type Client interface {
	// Chat sends messages and returns the model reply.
	Chat(ctx context.Context, messages []Message) (*Response, error)
	// ChatJSON is like Chat but requests JSON-mode output.
	ChatJSON(ctx context.Context, messages []Message) (*Response, error)
}

// ---- OpenAI-compatible client ----

type openAIClient struct {
	baseURL        string
	apiKey         string
	model          string
	temperature    float64
	timeoutSeconds int
	enableThinking bool
	httpClient     *http.Client
}

// openAI wire types (minimal subset we need)
type chatRequest struct {
	Model          string         `json:"model"`
	Messages       []Message      `json:"messages"`
	Temperature    float64        `json:"temperature,omitempty"`
	ResponseFormat *responseFormat `json:"response_format,omitempty"`
	Stream         bool           `json:"stream"`
	ExtraBody      map[string]any `json:"extra_body,omitempty"`
}

type responseFormat struct {
	Type string `json:"type"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
		} `json:"message"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		CompletionTokensDetails struct {
			ReasoningTokens int `json:"reasoning_tokens"`
		} `json:"completion_tokens_details"`
	} `json:"usage"`
}

// sharedTransport is a package-level HTTP transport shared across all LLM clients.
// This enables TCP+TLS connection reuse (keep-alive), matching the behaviour of
// Python's httpx connection pool.  Without sharing, each openAIClient would open
// its own fresh connections and trigger DashScope's per-IP new-connection rate limit
// under heavy concurrency.
var sharedTransport = &http.Transport{
	MaxIdleConns:        64,
	MaxIdleConnsPerHost: 16,
	IdleConnTimeout:     90 * time.Second,
}

func newOpenAIClient(baseURL, apiKey, model string, temperature float64, timeoutSeconds int, enableThinking bool) *openAIClient {
	return &openAIClient{
		baseURL:        strings.TrimRight(baseURL, "/"),
		apiKey:         apiKey,
		model:          model,
		temperature:    temperature,
		timeoutSeconds: timeoutSeconds,
		enableThinking: enableThinking,
		// No Timeout on the http.Client itself: callers pass a context with deadline,
		// which is the correct Go idiom.  Setting Timeout here would race with the
		// context deadline and produce confusing error messages.
		httpClient: &http.Client{Transport: sharedTransport},
	}
}

func (c *openAIClient) chat(ctx context.Context, messages []Message, jsonMode bool) (*Response, error) {
	req := chatRequest{
		Model:       c.model,
		Messages:    messages,
		Temperature: c.temperature,
		Stream:      false,
	}
	if jsonMode {
		req.ResponseFormat = &responseFormat{Type: "json_object"}
	}
	// DashScope / DeepSeek support enable_thinking via extra_body
	lower := strings.ToLower(c.baseURL)
	if strings.Contains(lower, "dashscope") || strings.Contains(lower, "deepseek") {
		req.ExtraBody = map[string]any{"enable_thinking": c.enableThinking}
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("llm: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("llm: build request: %w", err)
	}
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("llm: http request: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("llm: read body: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("llm: provider returned %d: %s", resp.StatusCode, string(raw))
	}

	var cr chatResponse
	if err := json.Unmarshal(raw, &cr); err != nil {
		return nil, fmt.Errorf("llm: unmarshal response: %w", err)
	}
	if len(cr.Choices) == 0 {
		return nil, fmt.Errorf("llm: no choices in response")
	}

	return &Response{
		Content:         cr.Choices[0].Message.Content,
		ThinkingContent: cr.Choices[0].Message.ReasoningContent,
		InputTokens:     cr.Usage.PromptTokens,
		OutputTokens:    cr.Usage.CompletionTokens,
		ReasoningTokens: cr.Usage.CompletionTokensDetails.ReasoningTokens,
	}, nil
}

func (c *openAIClient) Chat(ctx context.Context, messages []Message) (*Response, error) {
	return c.chat(ctx, messages, false)
}

func (c *openAIClient) ChatJSON(ctx context.Context, messages []Message) (*Response, error) {
	return c.chat(ctx, messages, true)
}

// ---- Factory ----

type providerConfig struct {
	apiKeyEnv  string
	baseURLEnv string
	defaultURL string
	modelEnv   string
	defaultModel string
}

var providers = []providerConfig{
	{
		apiKeyEnv:    "DASHSCOPE_API_KEY",
		baseURLEnv:   "DASHSCOPE_BASE_URL",
		defaultURL:   "https://dashscope.aliyuncs.com/compatible-mode/v1",
		modelEnv:     "DASHSCOPE_MODEL",
		defaultModel: "qwen-plus",
	},
	{
		apiKeyEnv:    "DEEPSEEK_API_KEY",
		baseURLEnv:   "DEEPSEEK_BASE_URL",
		defaultURL:   "https://api.deepseek.com",
		defaultModel: "deepseek-chat",
	},
	{
		apiKeyEnv:    "OPENAI_API_KEY",
		baseURLEnv:   "OPENAI_BASE_URL",
		defaultURL:   "https://api.openai.com/v1",
		defaultModel: "gpt-4o-mini",
	},
}

// Config holds LLM client configuration.
type Config struct {
	Provider       string  // "dashscope" | "deepseek" | "openai" | "" (auto)
	Model          string  // override model name
	Temperature    float64
	TimeoutSeconds int
	EnableThinking bool
}

// NewClient returns a Client based on available environment variables.
// Provider priority when Config.Provider is empty: DashScope → DeepSeek → OpenAI.
func NewClient(cfg Config) (Client, error) {
	if cfg.TimeoutSeconds <= 0 {
		cfg.TimeoutSeconds = 60
	}

	requested := strings.ToLower(strings.TrimSpace(cfg.Provider))

	for _, p := range providers {
		// Skip providers that don't match the explicit request.
		if requested != "" {
			pName := strings.ToLower(strings.Split(p.apiKeyEnv, "_")[0])
			if !strings.HasPrefix(pName, requested) {
				continue
			}
		}

		apiKey := cleanEnv(p.apiKeyEnv)
		if apiKey == "" {
			if requested != "" {
				return nil, fmt.Errorf("llm: provider %q selected but %s is not set", requested, p.apiKeyEnv)
			}
			continue
		}

		baseURL := cleanEnv(p.baseURLEnv)
		if baseURL == "" {
			baseURL = p.defaultURL
		}

		model := cfg.Model
		if model == "" {
			model = cleanEnv(p.modelEnv)
		}
		if model == "" {
			model = p.defaultModel
		}

		slog.Info("llm: using provider", "base_url", baseURL, "model", model)
		return newOpenAIClient(baseURL, apiKey, model, cfg.Temperature, cfg.TimeoutSeconds, cfg.EnableThinking), nil
	}

	return nil, fmt.Errorf("llm: no API key found; set DASHSCOPE_API_KEY, DEEPSEEK_API_KEY, or OPENAI_API_KEY")
}

func cleanEnv(name string) string {
	return strings.Trim(strings.TrimSpace(os.Getenv(name)), `"'`)
}
