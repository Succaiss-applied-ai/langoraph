package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
)

// SchemaMode tags which response_format strictness tier carried a
// structured-output call, so callers can record observability traces.
//
// Mirrors Python's ``rf_mode`` from ``call_llm_json_with_validation``.
type SchemaMode string

const (
	SchemaModeJSONObject       SchemaMode = "json_object"
	SchemaModeJSONSchemaEnum   SchemaMode = "json_schema_enum"
	SchemaModeJSONSchemaFreer  SchemaMode = "json_schema_pattern"
	SchemaModeJSONSchemaCustom SchemaMode = "json_schema"
)

// ValidatorVerdict is what a Validator returns about one LLM payload.
//
// Field semantics — exactly mirrors Python's
// ``(is_valid, normalised_payload, feedback_for_next_turn_or_None)``:
//
//   - OK=true: the validator accepted Normalised. No retry occurs;
//     the helper returns Normalised immediately.
//   - OK=false, Feedback=="": the validator rejected the payload but
//     did not provide feedback. The helper still returns Normalised
//     (likely a partial/best-effort subset) without a retry.
//   - OK=false, Feedback!="": the validator rejected the payload and
//     supplied a user-facing feedback string. The helper appends the
//     last assistant output and the feedback to the message history,
//     then re-prompts up to MaxRetries more times.
type ValidatorVerdict[T any] struct {
	OK         bool
	Normalised T
	Feedback   string
}

// Validator inspects the parsed JSON object the LLM produced and
// returns a ValidatorVerdict. The raw map is the result of
// json.Unmarshal'ing the (post-extraction) LLM JSON; if the response
// could not be parsed at all, the validator is invoked with an empty
// map and free to decide whether that's salvageable or worth a retry.
type Validator[T any] func(payload map[string]any) ValidatorVerdict[T]

// ValidatedOutcome is the full result of ChatStructuredWithFeedback,
// carrying both the validated value and the observability bookkeeping
// callers (e.g. the KTree scope_trace) need.
type ValidatedOutcome[T any] struct {
	Result          T
	RetryCount      int
	SchemaMode      SchemaMode
	LastRawResponse string
}

// ChatStructuredWithFeedback drives a Python-style validation retry
// loop on top of the LLM client.
//
// Behaviour, mirroring ``call_llm_json_with_validation``:
//
//  1. Send ``messages`` once. If ``schema`` is non-nil, use ChatSchema
//     (json_schema response_format); otherwise use ChatJSON
//     (json_object). Streaming is honoured automatically when the
//     client (or per-call ChatOption) sets it.
//  2. Extract a JSON object from the response, json.Unmarshal it, then
//     hand the parsed map to ``validator``.
//  3. If the verdict is OK or the validator chose not to provide
//     feedback (Feedback==""), return immediately.
//  4. Otherwise append an ``assistant`` turn carrying the previous raw
//     output (truncated to 2000 runes to mirror Python) and a ``user``
//     turn carrying the validator's feedback, then re-prompt. Repeat
//     up to ``maxRetries`` times.
//
// On exhaustion the most recently normalised payload is returned with
// ``RetryCount`` reflecting how many feedback-retries actually fired.
//
// Goroutine safety: the helper holds no shared state — every call gets
// its own message buffer and option set, so multiple concurrent
// validation loops (e.g. one per L0 fan-out branch) do not interfere.
func ChatStructuredWithFeedback[T any](
	ctx context.Context,
	client Client,
	messages []Message,
	schemaName string,
	schema map[string]any,
	validator Validator[T],
	maxRetries int,
	opts ...ChatOption,
) (ValidatedOutcome[T], error) {
	if validator == nil {
		var zero T
		return ValidatedOutcome[T]{Result: zero}, fmt.Errorf("llm: validator must not be nil")
	}
	if maxRetries < 0 {
		maxRetries = 0
	}

	mode := SchemaModeJSONObject
	useSchema := schema != nil && schemaName != ""
	if useSchema {
		mode = SchemaModeJSONSchemaCustom
	}

	// Local copy so the caller's slice is untouched across feedback
	// rounds. The slice grows by 2 entries per retry.
	turns := make([]Message, len(messages))
	copy(turns, messages)

	var (
		outcome   ValidatedOutcome[T]
		lastRaw   string
		zero      T
		lastValid T
	)
	outcome.SchemaMode = mode

	for attempt := 0; attempt <= maxRetries; attempt++ {
		var (
			resp *Response
			err  error
		)
		if useSchema {
			resp, err = client.ChatSchema(ctx, turns, schemaName, schema, opts...)
		} else {
			resp, err = client.ChatJSON(ctx, turns, opts...)
		}
		if err != nil {
			outcome.Result = lastValid
			outcome.LastRawResponse = lastRaw
			return outcome, fmt.Errorf("llm: chat call failed on attempt %d: %w", attempt+1, err)
		}

		lastRaw = resp.Content
		raw, jerr := extractJSON(resp.Content)
		if jerr != nil {
			slog.Warn("llm: feedback retry — failed to extract JSON",
				"attempt", attempt+1, "err", jerr, "head", head(resp.Content, 200))
			// Treat as validator-rejected with synthetic feedback so we
			// keep retrying within the same budget.
			if attempt == maxRetries {
				outcome.Result = lastValid
				outcome.RetryCount = attempt
				outcome.LastRawResponse = lastRaw
				return outcome, fmt.Errorf("llm: failed to extract JSON after %d attempts: %w", attempt+1, jerr)
			}
			outcome.RetryCount = attempt + 1
			turns = appendFeedback(turns, resp.Content,
				"上次输出无法解析为合法 JSON，请严格返回一个 JSON 对象，不要附加任何解释或 Markdown 代码块。",
			)
			continue
		}

		var payload map[string]any
		if uerr := json.Unmarshal([]byte(raw), &payload); uerr != nil {
			slog.Warn("llm: feedback retry — JSON unmarshal failed",
				"attempt", attempt+1, "err", uerr)
			if attempt == maxRetries {
				outcome.Result = lastValid
				outcome.RetryCount = attempt
				outcome.LastRawResponse = lastRaw
				return outcome, fmt.Errorf("llm: failed to unmarshal JSON after %d attempts: %w", attempt+1, uerr)
			}
			outcome.RetryCount = attempt + 1
			turns = appendFeedback(turns, resp.Content,
				"上次 JSON 结构无法被解析为对象，请重新返回一个合法的 JSON 对象。",
			)
			continue
		}

		verdict := validator(payload)
		lastValid = verdict.Normalised
		if verdict.OK {
			outcome.Result = verdict.Normalised
			outcome.RetryCount = attempt
			outcome.LastRawResponse = lastRaw
			return outcome, nil
		}
		if verdict.Feedback == "" || attempt == maxRetries {
			// Validator chose not to retry, or we are out of budget.
			outcome.Result = verdict.Normalised
			outcome.RetryCount = attempt
			outcome.LastRawResponse = lastRaw
			return outcome, nil
		}

		outcome.RetryCount = attempt + 1
		turns = appendFeedback(turns, resp.Content, verdict.Feedback)
	}

	// Unreachable in practice — every loop iteration returns. Defensive
	// fallback so the type-checker is happy.
	outcome.Result = zero
	return outcome, fmt.Errorf("llm: exhausted feedback retries without a verdict (unreachable)")
}

// appendFeedback returns a new message slice with the previous raw
// LLM output (truncated to 2000 runes, matching Python) and a new
// user turn carrying the validator's feedback string.
func appendFeedback(turns []Message, lastAssistant string, feedback string) []Message {
	out := make([]Message, len(turns), len(turns)+2)
	copy(out, turns)
	out = append(out, Message{Role: "assistant", Content: truncateRunes(lastAssistant, 2000)})
	out = append(out, Message{Role: "user", Content: feedback})
	return out
}

func truncateRunes(s string, max int) string {
	if max <= 0 {
		return ""
	}
	r := []rune(s)
	if len(r) <= max {
		return s
	}
	return string(r[:max])
}
