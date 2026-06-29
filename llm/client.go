// Package llm provides a unified LLM client interface and OpenAI-compatible
// implementations for DashScope (Qwen), DeepSeek, Tencent TokenPlan, and OpenAI.
//
// All providers share the same OpenAI Chat Completions wire format;
// only the base URL and API key differ.
//
// Streaming vs non-streaming
// --------------------------
// Each Client supports both modes. The active mode for a single call
// is resolved from (highest precedence first):
//
//  1. `WithStream(true|false)` option passed at the call site.
//  2. `Config.Stream` set when the Client was built with NewClient.
//  3. Default of `false` (a single non-streaming round-trip).
//
// For DashScope/DeepSeek/TokenPlan "thinking" models, streaming mode is the only
// safe choice — non-streaming requests sit on the connection until the
// model finishes thinking, often exceeding the HTTP timeout. Streaming
// mode runs a first-token watchdog that recognises the
// `reasoning_content` heartbeat as proof-of-life so reasoning bursts
// do not falsely time out.
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
	Content           string
	ThinkingContent   string // reasoning_content from Qwen thinking mode
	InputTokens       int
	OutputTokens      int
	ReasoningTokens   int
	ProviderRequestID string // provider HTTP response request id
}

// Client is the interface every LLM provider must satisfy.
//
// All three methods accept a variadic list of ChatOption so callers
// can override sampling parameters (temperature / top_p / seed /
// max_tokens), toggle streaming, or override the streaming first-token
// watchdog on a per-call basis. Options are source-compatible with
// pre-options call sites — passing zero options keeps the historical
// behaviour exactly.
type Client interface {
	// Chat sends messages and returns the model reply.
	Chat(ctx context.Context, messages []Message, opts ...ChatOption) (*Response, error)
	// ChatJSON is like Chat but requests JSON-mode output (json_object).
	ChatJSON(ctx context.Context, messages []Message, opts ...ChatOption) (*Response, error)
	// ChatSchema sends messages with a json_schema response_format, matching
	// Python's _invoke_structured_json_with_retry with json_schema enabled.
	// This is significantly faster than ChatJSON for DashScope/Qwen because
	// the model outputs tokens strictly within the schema without preamble.
	ChatSchema(ctx context.Context, messages []Message, schemaName string, schema map[string]any, opts ...ChatOption) (*Response, error)
}

// ---- OpenAI-compatible client ----

type openAIClient struct {
	baseURL        string
	apiKey         string
	model          string
	timeoutSeconds int

	// Defaults captured at construction. Per-call ChatOptions override
	// these. ``defaultTopP`` / ``defaultSeed`` / ``defaultMaxTokens``
	// are pointers so we can distinguish "Config did not set this"
	// from "Config explicitly set this to zero".
	defaultTemperature          float64
	defaultTopP                 *float64
	defaultSeed                 *int
	defaultMaxTokens            *int
	defaultEnableThinking       bool
	defaultThinkingBudget       *int
	defaultReasoningEffort      string
	defaultStream               bool
	defaultFirstTokenTimeout    time.Duration
	defaultFirstTokenMaxRetries int
	defaultChunkIdleTimeout     time.Duration

	httpClient *http.Client
}

// openAI wire types (minimal subset we need)
type chatRequest struct {
	Model          string          `json:"model"`
	Messages       []Message       `json:"messages"`
	Temperature    float64         `json:"temperature,omitempty"`
	TopP           *float64        `json:"top_p,omitempty"`
	Seed           *int            `json:"seed,omitempty"`
	MaxTokens      *int            `json:"max_tokens,omitempty"`
	ResponseFormat *responseFormat `json:"response_format,omitempty"`
	Stream         bool            `json:"stream"`
	StreamOptions  *streamOptions  `json:"stream_options,omitempty"`
	// EnableThinking is a DashScope/DeepSeek top-level extension for Qwen3 thinking models.
	// Python's openai SDK merges extra_body into the top-level request; Go must do the same.
	// Sending it nested under "extra_body" is silently ignored by DashScope, causing the model
	// to default to thinking=true and generate thousands of slow reasoning tokens.
	EnableThinking *bool `json:"enable_thinking,omitempty"`
	// ThinkingBudget is a DashScope/Ark extension consumed by Qwen thinking models.
	ThinkingBudget *int `json:"thinking_budget,omitempty"`
	// ReasoningEffort is a DashScope/DeepSeek top-level extension for DeepSeek thinking depth.
	ReasoningEffort *string `json:"reasoning_effort,omitempty"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type responseFormat struct {
	Type       string             `json:"type"`
	JSONSchema *jsonSchemaWrapper `json:"json_schema,omitempty"`
}

type jsonSchemaWrapper struct {
	Name   string         `json:"name"`
	Schema map[string]any `json:"schema"`
	Strict bool           `json:"strict,omitempty"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
		} `json:"message"`
	} `json:"choices"`
	Usage struct {
		PromptTokens            int `json:"prompt_tokens"`
		CompletionTokens        int `json:"completion_tokens"`
		CompletionTokensDetails struct {
			ReasoningTokens int `json:"reasoning_tokens"`
		} `json:"completion_tokens_details"`
	} `json:"usage"`
}

func providerRequestID(headers http.Header) string {
	for _, name := range []string{
		"x-request-id",
		"x-dashscope-request-id",
		"x-acs-request-id",
		"x-tc-requestid",
		"x-tc-request-id",
		"request-id",
	} {
		if value := strings.TrimSpace(headers.Get(name)); value != "" {
			return value
		}
	}
	return ""
}

// sharedTransport is a package-level HTTP transport shared across all LLM clients.
// This enables TCP+TLS connection reuse (keep-alive), matching the behaviour of
// Python's httpx connection pool.  Without sharing, each openAIClient would open
// its own fresh connections and trigger DashScope's per-IP new-connection rate limit
// under heavy concurrency.
//
// Goroutine safety: net/http.Transport is safe for concurrent use, so multiple
// concurrent Chat / ChatJSON / ChatSchema calls (LangGraph fan-out, Fanout, etc.)
// all share the same idle-connection pool without coordination.
var sharedTransport = &http.Transport{
	MaxIdleConns:        64,
	MaxIdleConnsPerHost: 16,
	IdleConnTimeout:     90 * time.Second,
}

func newOpenAIClient(baseURL, apiKey, model string, cfg Config) *openAIClient {
	c := &openAIClient{
		baseURL:                     strings.TrimRight(baseURL, "/"),
		apiKey:                      apiKey,
		model:                       model,
		timeoutSeconds:              cfg.TimeoutSeconds,
		defaultTemperature:          cfg.Temperature,
		defaultEnableThinking:       cfg.EnableThinking,
		defaultReasoningEffort:      cfg.ReasoningEffort,
		defaultStream:               cfg.Stream,
		defaultFirstTokenTimeout:    cfg.FirstTokenTimeout,
		defaultFirstTokenMaxRetries: cfg.FirstTokenMaxRetries,
		defaultChunkIdleTimeout:     cfg.ChunkIdleTimeout,
		// No Timeout on the http.Client itself: callers pass a context with deadline,
		// which is the correct Go idiom.  Setting Timeout here would race with the
		// context deadline and produce confusing error messages.
		httpClient: &http.Client{Transport: sharedTransport},
	}
	if cfg.TopP > 0 {
		v := cfg.TopP
		c.defaultTopP = &v
	}
	if cfg.SeedSet {
		v := cfg.Seed
		c.defaultSeed = &v
	}
	if cfg.MaxTokens > 0 {
		v := cfg.MaxTokens
		c.defaultMaxTokens = &v
	}
	if cfg.ThinkingBudget > 0 {
		v := cfg.ThinkingBudget
		c.defaultThinkingBudget = &v
	}
	return c
}

// buildRequest assembles the chatRequest body for one call by combining
// (in increasing precedence) the openAIClient defaults with the
// per-call ChatOption overrides. Caller decides whether to set
// ResponseFormat afterwards.
func (c *openAIClient) buildRequest(messages []Message, o chatOpts) chatRequest {
	req := chatRequest{
		Model:       c.model,
		Messages:    messages,
		Temperature: c.defaultTemperature,
		TopP:        c.defaultTopP,
		Seed:        c.defaultSeed,
		MaxTokens:   c.defaultMaxTokens,
		Stream:      c.defaultStream,
	}
	if o.temperature != nil {
		req.Temperature = *o.temperature
	}
	if o.topP != nil {
		v := *o.topP
		req.TopP = &v
	}
	if o.seed != nil {
		v := *o.seed
		req.Seed = &v
	}
	if o.maxTokens != nil {
		v := *o.maxTokens
		req.MaxTokens = &v
	}
	if o.stream != nil {
		req.Stream = *o.stream
	}
	if req.Stream {
		req.StreamOptions = &streamOptions{IncludeUsage: true}
	}

	// DashScope / DeepSeek / TokenPlan: send thinking controls as TOP-LEVEL fields.
	// Python's openai SDK extra_body merges into top-level — we must do the same.
	// Nesting it under "extra_body" key is silently ignored by DashScope, causing
	// qwen3.5-flash to default to thinking=true and generate ~5000 slow reasoning tokens.
	lower := strings.ToLower(c.baseURL)
	if isThinkingExtensionEndpoint(lower) {
		t := c.defaultEnableThinking
		if o.enableThinking != nil {
			t = *o.enableThinking
		}
		req.EnableThinking = &t
		if t {
			if budget := c.defaultThinkingBudget; budget != nil {
				req.ThinkingBudget = budget
			}
			if o.thinkingBudget != nil && *o.thinkingBudget > 0 {
				req.ThinkingBudget = o.thinkingBudget
			}
		}
		if effort := c.defaultReasoningEffort; effort != "" {
			req.ReasoningEffort = &effort
		}
		if o.reasoningEffort != nil {
			req.ReasoningEffort = o.reasoningEffort
		}
	} else if o.enableThinking != nil {
		// Caller explicitly opted in for a non-Aliyun endpoint; honour it
		// even though most providers will silently ignore the field.
		t := *o.enableThinking
		req.EnableThinking = &t
	}
	if !isThinkingExtensionEndpoint(lower) && o.reasoningEffort != nil {
		req.ReasoningEffort = o.reasoningEffort
	}
	return req
}

func isThinkingExtensionEndpoint(lowerBaseURL string) bool {
	return strings.Contains(lowerBaseURL, "dashscope") ||
		strings.Contains(lowerBaseURL, "deepseek") ||
		strings.Contains(lowerBaseURL, "tokenhub") ||
		strings.Contains(lowerBaseURL, "tencentmaas") ||
		strings.Contains(lowerBaseURL, "ark") ||
		strings.Contains(lowerBaseURL, "volces")
}

// resolveStreamWatchdog returns the effective first-token timeout +
// retry budget for a streaming call, applying per-call overrides on
// top of the client defaults.
func (c *openAIClient) resolveStreamWatchdog(o chatOpts) (timeout time.Duration, retries int, chunkIdleTimeout time.Duration) {
	timeout = c.defaultFirstTokenTimeout
	if o.firstTokenTimeout != nil {
		timeout = *o.firstTokenTimeout
	}
	retries = c.defaultFirstTokenMaxRetries
	if o.firstTokenMaxRetries != nil {
		retries = *o.firstTokenMaxRetries
	}
	if retries < 0 {
		retries = 0
	}
	chunkIdleTimeout = c.defaultChunkIdleTimeout
	if o.chunkIdleTimeout != nil {
		chunkIdleTimeout = *o.chunkIdleTimeout
	}
	return timeout, retries, chunkIdleTimeout
}

func (c *openAIClient) chat(ctx context.Context, messages []Message, jsonMode bool, opts []ChatOption) (*Response, error) {
	o := applyOptions(opts)
	req := c.buildRequest(messages, o)
	if jsonMode {
		req.ResponseFormat = &responseFormat{Type: "json_object"}
	}
	if req.Stream {
		ftt, ftMaxRetries, chunkIdleTimeout := c.resolveStreamWatchdog(o)
		return c.chatStream(ctx, req, ftt, ftMaxRetries, chunkIdleTimeout)
	}
	return c.chatNonStream(ctx, req)
}

// chatNonStream is the original blocking-POST path. Returned on
// non-streaming Config / call sites (default behaviour, backward
// compatible with pre-streaming langoraph).
func (c *openAIClient) chatNonStream(ctx context.Context, req chatRequest) (*Response, error) {
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
	defer func() { _ = resp.Body.Close() }()

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

	r := &Response{
		Content:           cr.Choices[0].Message.Content,
		ThinkingContent:   cr.Choices[0].Message.ReasoningContent,
		InputTokens:       cr.Usage.PromptTokens,
		OutputTokens:      cr.Usage.CompletionTokens,
		ReasoningTokens:   cr.Usage.CompletionTokensDetails.ReasoningTokens,
		ProviderRequestID: providerRequestID(resp.Header),
	}
	if r.ReasoningTokens > 0 && !req.streamingThinkingEnabled() {
		slog.Warn("llm: unexpected thinking tokens detected — enable_thinking may not be taking effect",
			"reasoning_tokens", r.ReasoningTokens, "model", c.model)
	}
	return r, nil
}

// streamingThinkingEnabled reports whether the request explicitly set
// enable_thinking=true. Used only for the "unexpected thinking tokens"
// warning path so we don't spam logs when thinking is intentional.
func (req chatRequest) streamingThinkingEnabled() bool {
	return req.EnableThinking != nil && *req.EnableThinking
}

func (c *openAIClient) Chat(ctx context.Context, messages []Message, opts ...ChatOption) (*Response, error) {
	return c.chat(ctx, messages, false, opts)
}

func (c *openAIClient) ChatJSON(ctx context.Context, messages []Message, opts ...ChatOption) (*Response, error) {
	return c.chat(ctx, messages, true, opts)
}

func (c *openAIClient) ChatSchema(ctx context.Context, messages []Message, schemaName string, schema map[string]any, opts ...ChatOption) (*Response, error) {
	o := applyOptions(opts)
	req := c.buildRequest(messages, o)
	req.ResponseFormat = &responseFormat{
		Type: "json_schema",
		JSONSchema: &jsonSchemaWrapper{
			Name:   schemaName,
			Schema: schema,
		},
	}
	if req.Stream {
		ftt, ftMaxRetries, chunkIdleTimeout := c.resolveStreamWatchdog(o)
		return c.chatStream(ctx, req, ftt, ftMaxRetries, chunkIdleTimeout)
	}
	return c.chatNonStream(ctx, req)
}

// ---- Factory ----

type providerConfig struct {
	name         string
	aliases      []string
	apiKeyEnvs   []string
	baseURLEnvs  []string
	defaultURL   string
	modelEnv     string
	defaultModel string
}

var providers = []providerConfig{
	{
		name:         "dashscope",
		aliases:      []string{"dashscope", "qwen"},
		apiKeyEnvs:   []string{"DASHSCOPE_API_KEY"},
		baseURLEnvs:  []string{"DASHSCOPE_BASE_URL"},
		defaultURL:   "https://dashscope.aliyuncs.com/compatible-mode/v1",
		modelEnv:     "DASHSCOPE_MODEL",
		defaultModel: "qwen-plus",
	},
	{
		name:         "deepseek",
		aliases:      []string{"deepseek"},
		apiKeyEnvs:   []string{"DASHSCOPE_API_KEY"},
		baseURLEnvs:  []string{"DASHSCOPE_BASE_URL"},
		defaultURL:   "https://dashscope.aliyuncs.com/compatible-mode/v1",
		defaultModel: "deepseek-chat",
	},
	{
		name:         "tokenplan",
		aliases:      []string{"tokenplan", "tencent-tokenplan", "tencent_tokenplan"},
		apiKeyEnvs:   []string{"TOKENPLAN_API_KEY", "TENCENT_TOKENPLAN_API_KEY"},
		baseURLEnvs:  []string{"TOKENPLAN_BASE_URL", "TENCENT_TOKENPLAN_BASE_URL"},
		defaultURL:   "https://tokenhub.tencentmaas.com/plan/v3",
		modelEnv:     "TOKENPLAN_MODEL",
		defaultModel: "deepseek-v4-flash-202605",
	},
	{
		name:         "openai",
		aliases:      []string{"openai", "gpt"},
		apiKeyEnvs:   []string{"OPENAI_API_KEY"},
		baseURLEnvs:  []string{"OPENAI_BASE_URL"},
		defaultURL:   "https://api.openai.com/v1",
		defaultModel: "gpt-4o-mini",
	},
	{
		name:         "ark",
		aliases:      []string{"ark", "doubao"},
		apiKeyEnvs:   []string{"ARK_API_KEY", "REVIEW_ARK_API_KEY"},
		baseURLEnvs:  []string{"ARK_BASE_URL"},
		defaultURL:   "https://ark.cn-beijing.volces.com/api/v3",
		modelEnv:     "ARK_MODEL",
		defaultModel: "doubao-seed-2-0-pro-260215",
	},
}

// Config holds LLM client configuration.
type Config struct {
	Provider       string // "dashscope" | "deepseek" | "tokenplan" | "openai" | "" (auto)
	BaseURL        string // explicit OpenAI-compatible base URL; falls back to provider env/default
	APIKey         string // explicit API key; falls back to provider env
	Model          string // override model name
	Temperature    float64
	TimeoutSeconds int
	EnableThinking bool
	// ThinkingBudget is sent as DashScope/Ark top-level thinking_budget when
	// EnableThinking is true and the value is positive.
	ThinkingBudget int
	// ReasoningEffort is sent as DashScope/DeepSeek top-level reasoning_effort
	// when non-empty, e.g. "low", "medium", or "high".
	ReasoningEffort string

	// --- Streaming defaults (per-call WithStream / WithFirstTokenTimeout
	//     / WithFirstTokenMaxRetries override these) ---
	//
	// Stream sets the default mode. When true, every call streams unless
	// a call-site WithStream(false) overrides. Required for DashScope
	// thinking models (qwen-deepseek, deepseek-v4-pro/flash) — the
	// non-stream path will deadline-exceed during the reasoning phase.
	Stream bool
	// FirstTokenTimeout caps how long the watchdog waits for the first
	// SSE delta carrying ``content`` (or ``reasoning_content`` heart-
	// beat) before declaring the upstream stalled and aborting the
	// stream. Zero or negative disables the watchdog.
	FirstTokenTimeout time.Duration
	// FirstTokenMaxRetries is the number of additional streaming
	// attempts on a first-token timeout before the timeout bubbles up
	// (matches Python's ``llm_first_token_max_retries``).
	FirstTokenMaxRetries int
	// ChunkIdleTimeout caps the allowed silence between successive SSE
	// chunks. This mirrors Python httpx read_timeout semantics.
	ChunkIdleTimeout time.Duration

	// --- Sampling defaults (per-call WithTopP / WithSeed /
	//     WithMaxTokens override these) ---
	//
	// TopP is omitted from the request when zero or unset.
	TopP float64
	// Seed is sent only when SeedSet is true (Go can't distinguish a
	// zero Seed from "unset" without an explicit flag).
	Seed    int
	SeedSet bool
	// MaxTokens is omitted from the request when zero or unset.
	MaxTokens int
}

// NewClient returns a Client based on explicit Config credentials or env vars.
// Provider priority when Config.Provider is empty: DashScope → DeepSeek → TokenPlan → OpenAI → Ark.
func NewClient(cfg Config) (Client, error) {
	if cfg.TimeoutSeconds <= 0 {
		cfg.TimeoutSeconds = 60
	}

	requested := strings.ToLower(strings.TrimSpace(cfg.Provider))
	explicitAPIKey := cleanString(cfg.APIKey)
	explicitBaseURL := cleanString(cfg.BaseURL)

	if explicitAPIKey != "" || explicitBaseURL != "" {
		p := providerForExplicitConfig(requested, explicitBaseURL)
		if p == nil {
			return nil, fmt.Errorf("llm: unsupported provider %q", requested)
		}
		apiKey := explicitAPIKey
		if apiKey == "" {
			apiKey = firstCleanEnv(p.apiKeyEnvs...)
		}
		if apiKey == "" {
			return nil, fmt.Errorf("llm: explicit base URL selected but no API key configured for provider %q", p.name)
		}
		baseURL := explicitBaseURL
		if baseURL == "" {
			baseURL = firstCleanEnv(p.baseURLEnvs...)
		}
		if baseURL == "" {
			baseURL = p.defaultURL
		}
		model := resolvedModel(cfg.Model, p)
		slog.Info("llm: using explicit provider config", "provider", p.name, "base_url", baseURL, "model", model, "stream", cfg.Stream)
		cfg.Model = model
		return newOpenAIClient(baseURL, apiKey, model, cfg), nil
	}

	for _, p := range providers {
		// Skip providers that don't match the explicit request.
		if requested != "" && !p.matches(requested) {
			continue
		}

		apiKey := firstCleanEnv(p.apiKeyEnvs...)
		if apiKey == "" {
			if requested != "" {
				return nil, fmt.Errorf("llm: provider %q selected but API key env is not set", requested)
			}
			continue
		}

		baseURL := firstCleanEnv(p.baseURLEnvs...)
		if baseURL == "" {
			baseURL = p.defaultURL
		}

		model := resolvedModel(cfg.Model, &p)

		slog.Info("llm: using provider", "provider", p.name, "base_url", baseURL, "model", model, "stream", cfg.Stream)
		// Capture the resolved model on the cfg copy so newOpenAIClient
		// records it alongside the user-supplied defaults.
		cfg.Model = model
		return newOpenAIClient(baseURL, apiKey, model, cfg), nil
	}

	return nil, fmt.Errorf("llm: no API key found; set DASHSCOPE_API_KEY, TOKENPLAN_API_KEY, OPENAI_API_KEY, or ARK_API_KEY")
}

func cleanEnv(name string) string {
	return cleanString(os.Getenv(name))
}

func cleanString(value string) string {
	return strings.Trim(strings.TrimSpace(value), `"'`)
}

func firstCleanEnv(names ...string) string {
	for _, name := range names {
		if value := cleanEnv(name); value != "" {
			return value
		}
	}
	return ""
}

func resolvedModel(override string, p *providerConfig) string {
	model := cleanString(override)
	if model != "" {
		return model
	}
	if p != nil && p.modelEnv != "" {
		model = cleanEnv(p.modelEnv)
	}
	if model != "" {
		return model
	}
	if p != nil {
		return p.defaultModel
	}
	return ""
}

func providerForExplicitConfig(requested, baseURL string) *providerConfig {
	if requested != "" {
		for i := range providers {
			if providers[i].matches(requested) {
				return &providers[i]
			}
		}
		return nil
	}
	baseURL = strings.ToLower(baseURL)
	for i := range providers {
		if providers[i].baseURLMatches(baseURL) {
			return &providers[i]
		}
	}
	return &providers[0]
}

func (p providerConfig) matches(requested string) bool {
	for _, alias := range p.aliases {
		if requested == alias {
			return true
		}
	}
	return false
}

func (p providerConfig) baseURLMatches(baseURL string) bool {
	if baseURL == "" {
		return false
	}
	switch p.name {
	case "dashscope", "deepseek":
		return strings.Contains(baseURL, "dashscope")
	case "tokenplan":
		return strings.Contains(baseURL, "tokenhub") || strings.Contains(baseURL, "tencentmaas")
	case "openai":
		return strings.Contains(baseURL, "openai")
	case "ark":
		return strings.Contains(baseURL, "ark") || strings.Contains(baseURL, "volces")
	default:
		return false
	}
}
