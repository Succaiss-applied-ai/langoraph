package langoraph_test

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"testing"

	langoraph "github.com/zhaocun/langoraph"
)

// ----- mock State types -----

// State that implements ErrorRecorder — matches generator AgentState pattern.
type RecordingState struct {
	Value  int
	Steps  []string
	Errors []string
}

func (s *RecordingState) RecordError(nodeName string, err error) {
	s.Errors = append(s.Errors, fmt.Sprintf("[%s] %v", nodeName, err))
}

// State that does NOT implement ErrorRecorder — errors are fatal.
type StrictState struct {
	Value int
}

// ----- TestPipeline_OrderedExecution -----
// Verifies nodes run in registration order and each sees the state left by the previous node.

func TestPipeline_OrderedExecution(t *testing.T) {
	var p langoraph.Pipeline[RecordingState]
	p.AddNode("step1", func(_ context.Context, s *RecordingState) error {
		s.Value += 1
		s.Steps = append(s.Steps, "step1")
		return nil
	})
	p.AddNode("step2", func(_ context.Context, s *RecordingState) error {
		s.Value *= 10
		s.Steps = append(s.Steps, "step2")
		return nil
	})
	p.AddNode("step3", func(_ context.Context, s *RecordingState) error {
		s.Value += 5
		s.Steps = append(s.Steps, "step3")
		return nil
	})

	state := &RecordingState{Value: 0}
	if err := p.Run(context.Background(), state); err != nil {
		t.Fatal(err)
	}

	// Expected: ((0+1)*10)+5 = 15
	if state.Value != 15 {
		t.Errorf("expected Value=15, got %d", state.Value)
	}
	if len(state.Steps) != 3 {
		t.Errorf("expected 3 steps, got %v", state.Steps)
	}
	if state.Steps[0] != "step1" || state.Steps[1] != "step2" || state.Steps[2] != "step3" {
		t.Errorf("unexpected step order: %v", state.Steps)
	}
}

// ----- TestPipeline_ErrorRecorder_ContinuesAfterError -----
// If state implements ErrorRecorder, a failing node records the error and execution continues.

func TestPipeline_ErrorRecorder_ContinuesAfterError(t *testing.T) {
	var p langoraph.Pipeline[RecordingState]
	p.AddNode("ok_before", func(_ context.Context, s *RecordingState) error {
		s.Steps = append(s.Steps, "ok_before")
		return nil
	})
	p.AddNode("fail", func(_ context.Context, s *RecordingState) error {
		return errors.New("simulated failure")
	})
	p.AddNode("ok_after", func(_ context.Context, s *RecordingState) error {
		s.Steps = append(s.Steps, "ok_after")
		return nil
	})

	state := &RecordingState{}
	if err := p.Run(context.Background(), state); err != nil {
		t.Fatalf("expected no returned error for ErrorRecorder state, got: %v", err)
	}

	if len(state.Errors) != 1 {
		t.Errorf("expected 1 recorded error, got %v", state.Errors)
	}
	if len(state.Steps) != 2 {
		t.Errorf("expected ok_before and ok_after to run, got steps: %v", state.Steps)
	}
}

// ----- TestPipeline_StrictState_StopsOnError -----
// If state does NOT implement ErrorRecorder, the first node error stops execution.

func TestPipeline_StrictState_StopsOnError(t *testing.T) {
	var p langoraph.Pipeline[StrictState]
	executed := 0
	p.AddNode("fail", func(_ context.Context, s *StrictState) error {
		executed++
		return errors.New("fatal error")
	})
	p.AddNode("should_not_run", func(_ context.Context, s *StrictState) error {
		executed++
		return nil
	})

	state := &StrictState{}
	err := p.Run(context.Background(), state)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if executed != 1 {
		t.Errorf("expected only 1 node to execute, got %d", executed)
	}
}

// ----- TestRunAll_ConcurrentIsolation -----
// RunAll runs N states concurrently; each state's mutations stay isolated.

func TestRunAll_ConcurrentIsolation(t *testing.T) {
	var p langoraph.Pipeline[RecordingState]
	p.AddNode("increment", func(_ context.Context, s *RecordingState) error {
		s.Value++
		s.Steps = append(s.Steps, fmt.Sprintf("val=%d", s.Value))
		return nil
	})

	const n = 10
	states := make([]*RecordingState, n)
	for i := range states {
		states[i] = &RecordingState{Value: i * 100}
	}

	if err := langoraph.RunAll(context.Background(), &p, states); err != nil {
		t.Fatal(err)
	}

	for i, s := range states {
		expected := i*100 + 1
		if s.Value != expected {
			t.Errorf("state[%d]: expected Value=%d, got %d", i, expected, s.Value)
		}
	}
}

// ----- TestRunAll_ActuallyParallel -----
// Verifies that RunAll runs goroutines concurrently, not serially.

func TestRunAll_ActuallyParallel(t *testing.T) {
	var p langoraph.Pipeline[RecordingState]

	var active int64
	var maxActive int64

	p.AddNode("concurrent_check", func(ctx context.Context, s *RecordingState) error {
		cur := atomic.AddInt64(&active, 1)
		for {
			old := atomic.LoadInt64(&maxActive)
			if cur <= old || atomic.CompareAndSwapInt64(&maxActive, old, cur) {
				break
			}
		}
		done := make(chan struct{})
		go func() { close(done) }()
		<-done
		atomic.AddInt64(&active, -1)
		return nil
	})

	const n = 20
	states := make([]*RecordingState, n)
	for i := range states {
		states[i] = &RecordingState{}
	}

	if err := langoraph.RunAll(context.Background(), &p, states); err != nil {
		t.Fatal(err)
	}

	if maxActive < 2 {
		t.Errorf("expected concurrent execution (maxActive>=2), got %d", maxActive)
	}
}
