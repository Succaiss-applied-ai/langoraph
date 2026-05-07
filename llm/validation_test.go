// Tests for ChatStructuredWithFeedback — the feedback-driven retry
// loop that mirrors Python's call_llm_json_with_validation.
package llm

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
)

// fakeClient is a stub Client that returns canned responses in order
// and records every messages slice it received.
type fakeClient struct {
	mu       sync.Mutex
	messages [][]Message
	jsonResp []string
	jsonErr  []error
	cursor   int
}

func (f *fakeClient) push(content string, err error) {
	f.jsonResp = append(f.jsonResp, content)
	f.jsonErr = append(f.jsonErr, err)
}

func (f *fakeClient) take(msgs []Message) (*Response, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	saved := make([]Message, len(msgs))
	copy(saved, msgs)
	f.messages = append(f.messages, saved)
	if f.cursor >= len(f.jsonResp) {
		return nil, fmt.Errorf("fakeClient: ran out of canned responses")
	}
	resp := f.jsonResp[f.cursor]
	err := f.jsonErr[f.cursor]
	f.cursor++
	return &Response{Content: resp}, err
}

func (f *fakeClient) Chat(_ context.Context, m []Message, _ ...ChatOption) (*Response, error) {
	return f.take(m)
}
func (f *fakeClient) ChatJSON(_ context.Context, m []Message, _ ...ChatOption) (*Response, error) {
	return f.take(m)
}
func (f *fakeClient) ChatSchema(_ context.Context, m []Message, _ string, _ map[string]any, _ ...ChatOption) (*Response, error) {
	return f.take(m)
}

// helper: build an "ids" validator like the L0/L1/L2 selectors do.
type idsResult struct {
	IDs []string
}

func idsValidator(allow map[string]bool) Validator[idsResult] {
	return func(payload map[string]any) ValidatorVerdict[idsResult] {
		raw, ok := payload["ids"].([]any)
		if !ok {
			return ValidatorVerdict[idsResult]{
				OK:       false,
				Feedback: "缺少 ids 字段或 ids 不是数组，请只输出 {\"ids\":[...]}。",
			}
		}
		var unknown []string
		var keep []string
		for _, v := range raw {
			s, _ := v.(string)
			if allow[s] {
				keep = append(keep, s)
			} else {
				unknown = append(unknown, s)
			}
		}
		if len(unknown) > 0 {
			return ValidatorVerdict[idsResult]{
				OK:         false,
				Normalised: idsResult{IDs: keep},
				Feedback:   fmt.Sprintf("以下 id 不在白名单内：%v，请只返回白名单中的 id。", unknown),
			}
		}
		return ValidatorVerdict[idsResult]{OK: true, Normalised: idsResult{IDs: keep}}
	}
}

// ----- TestValidation_Success_FirstTry -----

func TestValidation_Success_FirstTry(t *testing.T) {
	allow := map[string]bool{"L0_a": true, "L0_b": true}
	fc := &fakeClient{}
	fc.push(`{"ids":["L0_a","L0_b"]}`, nil)

	out, err := ChatStructuredWithFeedback(
		context.Background(), fc,
		[]Message{{Role: "user", Content: "pick"}},
		"", nil,
		idsValidator(allow), 3,
	)
	if err != nil {
		t.Fatalf("expected success, got %v", err)
	}
	if out.RetryCount != 0 {
		t.Errorf("expected RetryCount=0, got %d", out.RetryCount)
	}
	if len(out.Result.IDs) != 2 {
		t.Errorf("expected 2 ids, got %v", out.Result.IDs)
	}
}

// ----- TestValidation_RetryWithFeedback -----
// First call returns an out-of-whitelist id; validator rejects with
// feedback; second call succeeds.

func TestValidation_RetryWithFeedback(t *testing.T) {
	allow := map[string]bool{"L0_a": true, "L0_b": true}
	fc := &fakeClient{}
	fc.push(`{"ids":["L0_a","BOGUS"]}`, nil)
	fc.push(`{"ids":["L0_a","L0_b"]}`, nil)

	out, err := ChatStructuredWithFeedback(
		context.Background(), fc,
		[]Message{{Role: "user", Content: "pick"}},
		"", nil,
		idsValidator(allow), 3,
	)
	if err != nil {
		t.Fatalf("expected success after 1 retry, got %v", err)
	}
	if out.RetryCount != 1 {
		t.Errorf("expected RetryCount=1, got %d", out.RetryCount)
	}
	if len(out.Result.IDs) != 2 {
		t.Errorf("expected 2 ids, got %v", out.Result.IDs)
	}

	// Second invocation must have included the assistant + feedback turns.
	if len(fc.messages) != 2 {
		t.Fatalf("expected 2 LLM invocations, got %d", len(fc.messages))
	}
	round2 := fc.messages[1]
	if len(round2) != 3 {
		t.Fatalf("expected round2 messages of length 3 (user + assistant + feedback), got %d: %+v", len(round2), round2)
	}
	if round2[1].Role != "assistant" {
		t.Errorf("round2[1] should be assistant, got %q", round2[1].Role)
	}
	if round2[2].Role != "user" {
		t.Errorf("round2[2] should be user (feedback), got %q", round2[2].Role)
	}
	if !contains(round2[2].Content, "BOGUS") {
		t.Errorf("expected feedback to mention BOGUS, got %q", round2[2].Content)
	}
}

// ----- TestValidation_ExhaustionReturnsLastNormalised -----
// After maxRetries, helper returns the most recent Normalised payload.

func TestValidation_ExhaustionReturnsLastNormalised(t *testing.T) {
	allow := map[string]bool{"L0_a": true}
	fc := &fakeClient{}
	fc.push(`{"ids":["L0_a","X1"]}`, nil)
	fc.push(`{"ids":["L0_a","X2"]}`, nil)
	fc.push(`{"ids":["L0_a","X3"]}`, nil)

	out, err := ChatStructuredWithFeedback(
		context.Background(), fc,
		[]Message{{Role: "user", Content: "pick"}},
		"", nil,
		idsValidator(allow), 2, // 1 initial + 2 retries = 3 attempts
	)
	if err != nil {
		t.Fatalf("expected no error after exhaustion (returns last normalised), got %v", err)
	}
	if out.RetryCount != 2 {
		t.Errorf("expected RetryCount=2, got %d", out.RetryCount)
	}
	if len(out.Result.IDs) != 1 || out.Result.IDs[0] != "L0_a" {
		t.Errorf("expected fallback to whitelisted ids, got %v", out.Result.IDs)
	}
}

// ----- TestValidation_NoFeedbackStopsImmediately -----
// Validator returns OK=false with empty Feedback → helper accepts the
// normalised partial without retrying (mirrors Python's "validator
// chose to accept partial").

func TestValidation_NoFeedbackStopsImmediately(t *testing.T) {
	fc := &fakeClient{}
	fc.push(`{"ids":["only-this"]}`, nil)

	called := 0
	validator := func(payload map[string]any) ValidatorVerdict[idsResult] {
		called++
		raw, _ := payload["ids"].([]any)
		var ids []string
		for _, v := range raw {
			ids = append(ids, v.(string))
		}
		return ValidatorVerdict[idsResult]{OK: false, Normalised: idsResult{IDs: ids}, Feedback: ""}
	}

	out, err := ChatStructuredWithFeedback(
		context.Background(), fc,
		[]Message{{Role: "user", Content: "pick"}},
		"", nil,
		validator, 5,
	)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if called != 1 {
		t.Errorf("expected validator called exactly once (no feedback retry), got %d", called)
	}
	if out.RetryCount != 0 {
		t.Errorf("expected RetryCount=0, got %d", out.RetryCount)
	}
	if len(out.Result.IDs) != 1 {
		t.Errorf("expected 1 id, got %v", out.Result.IDs)
	}
}

// ----- TestValidation_BadJSONTriggersRetry -----
// LLM returns unparseable text → helper inserts synthetic feedback and
// retries.

func TestValidation_BadJSONTriggersRetry(t *testing.T) {
	allow := map[string]bool{"a": true}
	fc := &fakeClient{}
	fc.push("not json at all", nil)
	fc.push(`{"ids":["a"]}`, nil)

	out, err := ChatStructuredWithFeedback(
		context.Background(), fc,
		[]Message{{Role: "user", Content: "pick"}},
		"", nil,
		idsValidator(allow), 2,
	)
	if err != nil {
		t.Fatalf("expected success after extract retry, got %v", err)
	}
	if out.RetryCount != 1 {
		t.Errorf("expected RetryCount=1, got %d", out.RetryCount)
	}
	if len(out.Result.IDs) != 1 {
		t.Errorf("expected 1 id, got %v", out.Result.IDs)
	}
}

// ----- TestValidation_LLMErrorPropagates -----
// Transport-level error from the client must bubble up immediately
// (no synthetic retry).

func TestValidation_LLMErrorPropagates(t *testing.T) {
	fc := &fakeClient{}
	sentinel := errors.New("provider down")
	fc.push("", sentinel)

	_, err := ChatStructuredWithFeedback(
		context.Background(), fc,
		[]Message{{Role: "user", Content: "pick"}},
		"", nil,
		idsValidator(map[string]bool{}), 3,
	)
	if !errors.Is(err, sentinel) {
		t.Fatalf("expected sentinel error to propagate, got %v", err)
	}
}

// ----- TestValidation_NilValidator -----

func TestValidation_NilValidator(t *testing.T) {
	fc := &fakeClient{}
	_, err := ChatStructuredWithFeedback[idsResult](
		context.Background(), fc,
		[]Message{{Role: "user", Content: "pick"}},
		"", nil,
		nil, 3,
	)
	if err == nil {
		t.Fatal("expected error for nil validator")
	}
}

// ----- TestValidation_ConcurrentSafety -----
// Multiple goroutines drive their own ChatStructuredWithFeedback loops
// concurrently — independent message buffers, no shared mutation.

func TestValidation_ConcurrentSafety(t *testing.T) {
	const n = 8
	results := make([]idsResult, n)
	errs := make([]error, n)

	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			fc := &fakeClient{}
			fc.push(fmt.Sprintf(`{"ids":["item-%d"]}`, i), nil)
			validator := func(payload map[string]any) ValidatorVerdict[idsResult] {
				raw, _ := payload["ids"].([]any)
				var ids []string
				for _, v := range raw {
					ids = append(ids, v.(string))
				}
				return ValidatorVerdict[idsResult]{OK: true, Normalised: idsResult{IDs: ids}}
			}
			out, err := ChatStructuredWithFeedback(
				context.Background(), fc,
				[]Message{{Role: "user", Content: "x"}},
				"", nil, validator, 0,
			)
			results[i] = out.Result
			errs[i] = err
		}()
	}
	wg.Wait()

	for i := 0; i < n; i++ {
		if errs[i] != nil {
			t.Errorf("worker %d: %v", i, errs[i])
			continue
		}
		want := fmt.Sprintf("item-%d", i)
		if len(results[i].IDs) != 1 || results[i].IDs[0] != want {
			t.Errorf("worker %d: expected [%s], got %v", i, want, results[i].IDs)
		}
	}
}

// contains is provided by parallel_semantics_test.go but only in the
// langoraph_test package. Provide a local copy here for the llm pkg.
func contains(haystack, needle string) bool {
	for i := 0; i+len(needle) <= len(haystack); i++ {
		if haystack[i:i+len(needle)] == needle {
			return true
		}
	}
	return false
}
