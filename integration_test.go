// Integration tests for Pipeline and Fanout with real LLM nodes.
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
// TestPipeline_Integration_LLMNode
// A 3-node pipeline where node2 calls the real LLM to transform state.
// Verifies that the pipeline correctly wires node outputs into subsequent nodes.
// ---------------------------------------------------------------------------

type LLMPipelineState struct {
	Input    string
	Summary  string // written by LLM node
	WordCount int   // written by next node
	Errors   []string
}

func (s *LLMPipelineState) RecordError(node string, err error) {
	s.Errors = append(s.Errors, fmt.Sprintf("[%s] %v", node, err))
}

func TestPipeline_Integration_LLMNode(t *testing.T) {
	client := newIntegrationClient(t)

	var p langoraph.Pipeline[LLMPipelineState]

	// Node 1: set the input text
	p.AddNode("set_input", func(_ context.Context, s *LLMPipelineState) error {
		s.Input = "Go 语言由 Google 开发，以简洁、高效著称，内置并发支持（goroutine），编译速度快，生态丰富。"
		return nil
	})

	// Node 2: call the LLM to summarise
	p.AddNode("llm_summarise", func(ctx context.Context, s *LLMPipelineState) error {
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

	// Node 3: count words in the summary (pure logic)
	p.AddNode("count_words", func(_ context.Context, s *LLMPipelineState) error {
		s.WordCount = len([]rune(s.Summary))
		return nil
	})

	state := &LLMPipelineState{}
	start := time.Now()
	if err := p.Run(context.Background(), state); err != nil {
		t.Fatalf("pipeline failed: %v", err)
	}
	t.Logf("pipeline completed in %v", time.Since(start))

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
// Runs 3 independent pipelines concurrently, each with an LLM call.
// Verifies that RunAll completes all of them and they run in parallel.
// ---------------------------------------------------------------------------

func TestRunAll_Integration_Parallel(t *testing.T) {
	client := newIntegrationClient(t)

	type QState struct {
		Role    string
		Answer  string
		Errors  []string
	}
	// QState doesn't implement ErrorRecorder on purpose — errors are fatal.

	roles := []string{"前端工程师", "后端工程师", "数据工程师"}
	states := make([]*QState, len(roles))
	for i, r := range roles {
		states[i] = &QState{Role: r}
	}

	var p langoraph.Pipeline[QState]
	var callCount int64

	p.AddNode("ask_llm", func(ctx context.Context, s *QState) error {
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

	start := time.Now()
	if err := langoraph.RunAll(context.Background(), &p, states); err != nil {
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
		// Order must match input order
		if r.Q != questions[i] {
			t.Errorf("result[%d]: question mismatch, want %q got %q", i, questions[i], r.Q)
		}
		if r.A == "" {
			t.Errorf("result[%d]: empty answer", i)
		}
		t.Logf("[%d] Q: %s  A: %s", i, r.Q, r.A)
	}
}
