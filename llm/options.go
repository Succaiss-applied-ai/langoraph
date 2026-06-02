package llm

import "time"

// ChatOption tunes a single Chat / ChatJSON / ChatSchema call.
//
// Options are applied on top of the defaults captured in Config when
// the Client was constructed. An unset option leaves the default in
// place; a set option overrides it for that one call only. This is
// the same pattern Python's “chat_completion(**kwargs)“ uses, where
// “temperature=None“ falls back to “settings.llm_temperature“.
//
// Example:
//
//	client.Chat(ctx, msgs,
//	    llm.WithStream(true),
//	    llm.WithFirstTokenTimeout(8*time.Second),
//	    llm.WithMaxTokens(4096),
//	)
type ChatOption func(*chatOpts)

// chatOpts is the resolved per-call override set. All fields are
// pointers so the openAIClient.chat path can distinguish "explicitly
// set" from "leave defaults alone".
type chatOpts struct {
	stream               *bool
	temperature          *float64
	topP                 *float64
	seed                 *int
	maxTokens            *int
	enableThinking       *bool
	reasoningEffort      *string
	firstTokenTimeout    *time.Duration
	firstTokenMaxRetries *int
}

// applyOptions returns a chatOpts populated by every supplied option.
// nil options are skipped so callers can build option lists with
// conditional appends safely.
func applyOptions(opts []ChatOption) chatOpts {
	var o chatOpts
	for _, opt := range opts {
		if opt == nil {
			continue
		}
		opt(&o)
	}
	return o
}

// WithStream toggles streaming SSE for one call. When unset, the
// Config.Stream default wins.
//
// Streaming mode is required for "thinking" models (deepseek-v4-pro/
// flash, qwen-deepseek, …) whose first content token may arrive many
// seconds after the SSE connection opens; non-streaming requests would
// deadline-exceed during reasoning.
func WithStream(stream bool) ChatOption {
	return func(o *chatOpts) { o.stream = &stream }
}

// WithTemperature overrides the sampling temperature for one call.
// LangGraph parity: matches Python's per-call “temperature=...“.
func WithTemperature(t float64) ChatOption {
	return func(o *chatOpts) { o.temperature = &t }
}

// WithTopP overrides nucleus sampling for one call. When unset, the
// Config.TopP default wins; a 0 default means the provider's own
// default is used (the field is omitted from the request body).
func WithTopP(p float64) ChatOption {
	return func(o *chatOpts) { o.topP = &p }
}

// WithSeed overrides the determinism seed for one call. Providers
// treat this as best-effort.
func WithSeed(s int) ChatOption {
	return func(o *chatOpts) { o.seed = &s }
}

// WithMaxTokens overrides the per-call max output token budget.
func WithMaxTokens(n int) ChatOption {
	return func(o *chatOpts) { o.maxTokens = &n }
}

// WithEnableThinking overrides DashScope/DeepSeek's “enable_thinking“
// extension for one call. Useful for selectively turning thinking off
// on a per-call basis (e.g. a fast-path classifier inside an otherwise
// thinking-mode pipeline).
func WithEnableThinking(b bool) ChatOption {
	return func(o *chatOpts) { o.enableThinking = &b }
}

// WithReasoningEffort overrides DashScope/DeepSeek's “reasoning_effort“
// extension for one call. Valid provider values are model-dependent, but
// DashScope DeepSeek models accept "", "low", "medium", and "high".
func WithReasoningEffort(effort string) ChatOption {
	return func(o *chatOpts) { o.reasoningEffort = &effort }
}

// WithFirstTokenTimeout overrides how long the streaming layer will
// wait for the first “content“ (or “reasoning_content“ heartbeat)
// SSE delta before declaring the upstream stalled. Only meaningful in
// streaming mode.
//
// A non-positive value disables the watchdog (the connection's normal
// HTTP timeout is then the only ceiling).
func WithFirstTokenTimeout(d time.Duration) ChatOption {
	return func(o *chatOpts) { o.firstTokenTimeout = &d }
}

// WithFirstTokenMaxRetries overrides how many extra streaming attempts
// will be made on a first-token timeout before the timeout bubbles up.
// Each retry opens a fresh SSE connection; counters carry inside the
// llm package only — the structured-output retry loop and any caller
// retry logic are layered on top.
func WithFirstTokenMaxRetries(n int) ChatOption {
	return func(o *chatOpts) { o.firstTokenMaxRetries = &n }
}
