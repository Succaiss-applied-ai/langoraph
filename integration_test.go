// Integration tests for Graph and Fanout with real LLM nodes.
// Run with: go test ./... -run Integration -v
// (auto-skipped if no API key is available)
package langoraph_test

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	langoraph "github.com/Succaiss-applied-ai/langoraph"
	"github.com/Succaiss-applied-ai/langoraph/llm"
)

// ---------------------------------------------------------------------------
// Shared setup
// ---------------------------------------------------------------------------

func loadEnvForIntegration(t *testing.T) {
	t.Helper()
	_, thisFile, _, _ := runtime.Caller(0)
	dir := filepath.Dir(thisFile)
	for i := 0; i < 5; i++ {
		data, err := os.ReadFile(filepath.Join(dir, ".env"))
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
			if len(parts) == 2 {
				key := strings.TrimSpace(parts[0])
				val := strings.Trim(strings.TrimSpace(parts[1]), `"'`)
				if os.Getenv(key) == "" {
					os.Setenv(key, val)
				}
			}
		}
		return
	}
}

func newIntegrationClient(t *testing.T) llm.Client {
	t.Helper()
	loadEnvForIntegration(t)
	client, err := llm.NewClient(llm.Config{Temperature: 0.1, TimeoutSeconds: 30})
	if err != nil {
		t.Skipf("no LLM API key, skipping: %v", err)
	}
	return client
}

// ---------------------------------------------------------------------------
// TestGraph_Integration_LLMNode
// A 3-node graph where node2 calls the real LLM to transform state.
// ---------------------------------------------------------------------------

type LLMGraphState struct {
	Input     string
	Summary   string
	WordCount int
	Errors    []string
}

func (s *LLMGraphState) RecordError(node string, err error) {
	s.Errors = append(s.Errors, fmt.Sprintf("[%s] %v", node, err))
}

func TestGraph_Integration_LLMNode(t *testing.T) {
	client := newIntegrationClient(t)

	g := langoraph.NewGraph[LLMGraphState]()

	g.AddNode("set_input", func(_ context.Context, s *LLMGraphState) error {
		s.Input = "Go 语言由 Google 开发，以简洁、高效著称，内置并发支持（goroutine），编译速度快，生态丰富。"
		return nil
	})

	g.AddNode("llm_summarise", func(ctx context.Context, s *LLMGraphState) error {
		prompt := fmt.Sprintf(
			`请用不超过10个字总结以下内容，只输出 JSON：{"summary":"..."}\n内容：%s`,
			s.Input,
		)
		var out struct {
			Summary string `json:"summary"`
		}
		if err := llm.ChatStructured(ctx, client, prompt, &out); err != nil {
			return fmt.Errorf("llm_summarise: %w", err)
		}
		s.Summary = out.Summary
		return nil
	})

	g.AddNode("count_words", func(_ context.Context, s *LLMGraphState) error {
		s.WordCount = len([]rune(s.Summary))
		return nil
	})

	g.AddEdge(langoraph.START, "set_input")
	g.AddEdge("set_input", "llm_summarise")
	g.AddEdge("llm_summarise", "count_words")
	g.AddEdge("count_words", langoraph.END)

	state := &LLMGraphState{}
	start := time.Now()
	if err := g.Run(context.Background(), state); err != nil {
		t.Fatalf("graph failed: %v", err)
	}
	t.Logf("graph completed in %v", time.Since(start))

	if len(state.Errors) > 0 {
		t.Errorf("node errors: %v", state.Errors)
	}
	if state.Summary == "" {
		t.Error("expected non-empty summary from LLM node")
	}
	if state.WordCount == 0 {
		t.Error("expected word count > 0")
	}
	t.Logf("summary=%q word_count=%d", state.Summary, state.WordCount)
}

// ---------------------------------------------------------------------------
// TestRunAll_Integration_Parallel
// Runs 3 independent graphs concurrently, each with an LLM call.
// ---------------------------------------------------------------------------

func TestRunAll_Integration_Parallel(t *testing.T) {
	client := newIntegrationClient(t)

	type QState struct {
		Role   string
		Answer string
	}

	roles := []string{"前端工程师", "后端工程师", "数据工程师"}
	states := make([]*QState, len(roles))
	for i, r := range roles {
		states[i] = &QState{Role: r}
	}

	var callCount int64

	g := langoraph.NewGraph[QState]()
	g.AddNode("ask_llm", func(ctx context.Context, s *QState) error {
		atomic.AddInt64(&callCount, 1)
		prompt := fmt.Sprintf(
			`请用一句话描述 %s 的核心职责，只输出 JSON：{"answer":"..."}`,
			s.Role,
		)
		var out struct {
			Answer string `json:"answer"`
		}
		if err := llm.ChatStructured(ctx, client, prompt, &out); err != nil {
			return fmt.Errorf("ask_llm(%s): %w", s.Role, err)
		}
		s.Answer = out.Answer
		return nil
	})
	g.AddEdge(langoraph.START, "ask_llm")
	g.AddEdge("ask_llm", langoraph.END)

	start := time.Now()
	if err := langoraph.RunAll(context.Background(), g, states); err != nil {
		t.Fatalf("RunAll failed: %v", err)
	}
	elapsed := time.Since(start)
	t.Logf("RunAll(%d states) completed in %v", len(states), elapsed)

	if callCount != int64(len(states)) {
		t.Errorf("expected %d LLM calls, got %d", len(states), callCount)
	}
	for i, s := range states {
		if s.Answer == "" {
			t.Errorf("state[%d] (%s): got empty answer", i, s.Role)
		}
		t.Logf("[%s] %s", s.Role, s.Answer)
	}
}

// ---------------------------------------------------------------------------
// TestFanout_Integration_LLM
// Fans out 3 items through a real LLM call, verifies order preservation.
// ---------------------------------------------------------------------------

func TestFanout_Integration_LLM(t *testing.T) {
	client := newIntegrationClient(t)

	questions := []string{
		"Go 中的 goroutine 是什么？",
		"什么是 channel？",
		"select 语句的作用是什么？",
	}

	type Answer struct {
		Q string
		A string
	}

	start := time.Now()
	results, err := langoraph.Fanout(
		context.Background(),
		questions,
		func(ctx context.Context, q string) (Answer, error) {
			prompt := fmt.Sprintf(
				`请用不超过15个字回答以下 Go 语言问题，只输出 JSON：{"answer":"..."}\n问题：%s`, q,
			)
			var out struct {
				Answer string `json:"answer"`
			}
			if err := llm.ChatStructured(ctx, client, prompt, &out); err != nil {
				return Answer{}, err
			}
			return Answer{Q: q, A: out.Answer}, nil
		},
	)
	if err != nil {
		t.Fatalf("Fanout failed: %v", err)
	}
	t.Logf("Fanout(%d items) completed in %v", len(questions), time.Since(start))

	if len(results) != len(questions) {
		t.Fatalf("expected %d results, got %d", len(questions), len(results))
	}
	for i, r := range results {
		if r.Q != questions[i] {
			t.Errorf("result[%d]: question mismatch, want %q got %q", i, questions[i], r.Q)
		}
		if r.A == "" {
			t.Errorf("result[%d]: empty answer", i)
		}
		t.Logf("[%d] Q: %s  A: %s", i, r.Q, r.A)
	}
}
