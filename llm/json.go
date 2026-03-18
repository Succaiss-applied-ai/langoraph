package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"regexp"
	"strings"
)

const maxLLMRetries = 2

// ChatStructured sends a prompt, extracts the first JSON object from the
// response, and unmarshals it into dst. Retries up to maxLLMRetries times
// on JSON parse failure — mirroring the Python _invoke_structured_json_with_retry.
func ChatStructured(ctx context.Context, client Client, prompt string, dst any) error {
	var lastErr error
	for attempt := 0; attempt <= maxLLMRetries; attempt++ {
		resp, err := client.ChatJSON(ctx, []Message{{Role: "user", Content: prompt}})
		if err != nil {
			return fmt.Errorf("llm: chat failed: %w", err)
		}

		raw, err := extractJSON(resp.Content)
		if err != nil {
			lastErr = err
			slog.Warn("llm: json extract failed, retrying",
				"attempt", attempt+1, "max", maxLLMRetries+1,
				"err", err, "head", head(resp.Content, 200))
			continue
		}

		if err := json.Unmarshal([]byte(raw), dst); err != nil {
			lastErr = err
			slog.Warn("llm: json unmarshal failed, retrying",
				"attempt", attempt+1, "err", err)
			continue
		}
		return nil
	}
	return fmt.Errorf("llm: structured output failed after %d attempts: %w", maxLLMRetries+1, lastErr)
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
