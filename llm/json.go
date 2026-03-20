package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"regexp"
	"strings"
)

// ChatStructured sends a prompt and unmarshals the JSON response into dst.
// It mirrors Python's _invoke_structured_json_with_retry exactly:
//   - If schemaName+schema are provided: tries json_schema mode first (fastest),
//     then falls back to plain JSON mode on failure.
//   - If no schema provided: uses plain JSON mode (json_object).
//
// Call ChatStructuredWithSchema when you have a schema (most nodes).
// Call ChatStructured for backward compat / nodes without schema.
func ChatStructured(ctx context.Context, client Client, prompt string, dst any) error {
	return chatStructuredInner(ctx, client, prompt, "", nil, dst)
}

// ChatStructuredWithSchema is like ChatStructured but uses json_schema response_format
// (Structured Outputs) first, falling back to plain JSON mode on error.
// This matches Python when AGENTIC_RAG_ENABLE_JSON_SCHEMA=true, and is significantly
// faster for DashScope/Qwen because the model outputs tokens strictly within the schema.
func ChatStructuredWithSchema(ctx context.Context, client Client, prompt, schemaName string, schema map[string]any, dst any) error {
	return chatStructuredInner(ctx, client, prompt, schemaName, schema, dst)
}

func chatStructuredInner(ctx context.Context, client Client, prompt, schemaName string, schema map[string]any, dst any) error {
	msgs := []Message{{Role: "user", Content: prompt}}

	// Try json_schema mode first when schema is provided (mirrors Python schema_enabled path).
	if schemaName != "" && schema != nil {
		resp, err := client.ChatSchema(ctx, msgs, schemaName, schema)
		if err == nil {
			raw, jerr := extractJSON(resp.Content)
			if jerr == nil {
				if uerr := json.Unmarshal([]byte(raw), dst); uerr == nil {
					return nil
				}
			}
			// Schema mode returned something unparseable — fall through to plain JSON.
			slog.Warn("llm: json_schema parse failed, falling back to json_object", "head", head(resp.Content, 200))
		} else {
			slog.Warn("llm: json_schema call failed, falling back to json_object", "err", err)
		}
	}

	// Plain JSON mode with retries (mirrors Python plain / plain_retry path).
	var lastErr error
	for attempt := 0; attempt <= 2; attempt++ {
		resp, err := client.ChatJSON(ctx, msgs)
		if err != nil {
			return fmt.Errorf("llm: chat failed: %w", err)
		}
		raw, err := extractJSON(resp.Content)
		if err != nil {
			lastErr = err
			slog.Warn("llm: json extract failed, retrying",
				"attempt", attempt+1, "err", err, "head", head(resp.Content, 200))
			continue
		}
		if err := json.Unmarshal([]byte(raw), dst); err != nil {
			lastErr = err
			slog.Warn("llm: json unmarshal failed, retrying", "attempt", attempt+1, "err", err)
			continue
		}
		return nil
	}
	return fmt.Errorf("llm: structured output failed after 3 attempts: %w", lastErr)
}

// ExtractJSON pulls the first {...} block out of text, tolerating markdown fences
// and applying best-effort fixes for trailing commas and comments.
func ExtractJSON(text string) (string, error) {
	return extractJSON(text)
}

func extractJSON(text string) (string, error) {
	text = strings.TrimSpace(text)

	// Strip markdown code fences.
	if m := reFence.FindStringSubmatch(text); m != nil {
		text = strings.TrimSpace(m[1])
	}

	start := strings.Index(text, "{")
	end := strings.LastIndex(text, "}") + 1
	if start == -1 || end == 0 {
		return "", fmt.Errorf("no JSON object found in: %s", head(text, 200))
	}
	raw := text[start:end]

	// Try as-is first.
	if json.Valid([]byte(raw)) {
		return raw, nil
	}

	// Best-effort fix.
	fixed := fixJSON(raw)
	if json.Valid([]byte(fixed)) {
		return fixed, nil
	}

	return "", fmt.Errorf("invalid JSON after fix attempt: %s", head(raw, 200))
}

// FixJSON applies best-effort repairs to a JSON string (removes comments, trailing commas).
func FixJSON(s string) string {
	return fixJSON(s)
}

func fixJSON(s string) string {
	// Remove // line comments.
	s = reLineComment.ReplaceAllString(s, "")
	// Remove /* block comments */.
	s = reBlockComment.ReplaceAllString(s, "")
	// Remove trailing commas before } or ].
	s = reTrailingComma.ReplaceAllString(s, "$1")
	return s
}

var (
	reFence        = regexp.MustCompile("```(?:json)?\\s*([\\s\\S]*?)```")
	reLineComment  = regexp.MustCompile(`(?m)//.*?$`)
	reBlockComment = regexp.MustCompile(`(?s)/\*.*?\*/`)
	reTrailingComma = regexp.MustCompile(`,\s*([}\]])`)
)

func head(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
