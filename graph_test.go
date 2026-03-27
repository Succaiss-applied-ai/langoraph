package langoraph_test

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"

	langoraph "github.com/Succaiss-applied-ai/langoraph"
)

// ----- TestGraph_LinearExecution -----
// A graph with only fixed edges behaves like a linear pipeline.

func TestGraph_LinearExecution(t *testing.T) {
	g := langoraph.NewGraph[RecordingState]()

	g.AddNode("a", func(_ context.Context, s *RecordingState) error {
		s.Value += 1
		s.Steps = append(s.Steps, "a")
		return nil
	})
	g.AddNode("b", func(_ context.Context, s *RecordingState) error {
		s.Value *= 10
		s.Steps = append(s.Steps, "b")
		return nil
	})
	g.AddNode("c", func(_ context.Context, s *RecordingState) error {
		s.Value += 5
		s.Steps = append(s.Steps, "c")
		return nil
	})

	g.AddEdge(langoraph.START, "a")
	g.AddEdge("a", "b")
	g.AddEdge("b", "c")
	g.AddEdge("c", langoraph.END)

	state := &RecordingState{}
	if err := g.Run(context.Background(), state); err != nil {
		t.Fatal(err)
	}

	if state.Value != 15 {
		t.Errorf("expected Value=15, got %d", state.Value)
	}
	want := []string{"a", "b", "c"}
	if len(state.Steps) != len(want) {
		t.Fatalf("expected steps %v, got %v", want, state.Steps)
	}
	for i, s := range want {
		if state.Steps[i] != s {
			t.Errorf("step[%d]: expected %q, got %q", i, s, state.Steps[i])
		}
	}
}

// ----- TestGraph_ConditionalEdge -----
// Routes to different nodes based on state.

func TestGraph_ConditionalEdge(t *testing.T) {
	g := langoraph.NewGraph[RecordingState]()

	g.AddNode("classify", func(_ context.Context, s *RecordingState) error {
		s.Steps = append(s.Steps, "classify")
		return nil
	})
	g.AddNode("positive", func(_ context.Context, s *RecordingState) error {
		s.Steps = append(s.Steps, "positive")
		return nil
	})
	g.AddNode("negative", func(_ context.Context, s *RecordingState) error {
		s.Steps = append(s.Steps, "negative")
		return nil
	})

	g.AddEdge(langoraph.START, "classify")
	g.AddConditionalEdge("classify",
		func(_ context.Context, s *RecordingState) string {
			if s.Value >= 0 {
				return "pos"
			}
			return "neg"
		},
		map[string]string{
			"pos": "positive",
			"neg": "negative",
		},
	)
	g.AddEdge("positive", langoraph.END)
	g.AddEdge("negative", langoraph.END)

	t.Run("positive_path", func(t *testing.T) {
		state := &RecordingState{Value: 42}
		if err := g.Run(context.Background(), state); err != nil {
			t.Fatal(err)
		}
		if len(state.Steps) != 2 || state.Steps[1] != "positive" {
			t.Errorf("expected [classify positive], got %v", state.Steps)
		}
	})

	t.Run("negative_path", func(t *testing.T) {
		state := &RecordingState{Value: -1}
		if err := g.Run(context.Background(), state); err != nil {
			t.Fatal(err)
		}
		if len(state.Steps) != 2 || state.Steps[1] != "negative" {
			t.Errorf("expected [classify negative], got %v", state.Steps)
		}
	})
}

// ----- TestGraph_ConditionalEdge_Passthrough -----
// When the router returns a key not in the mapping, the key itself is used
// as the target node name.

func TestGraph_ConditionalEdge_Passthrough(t *testing.T) {
	g := langoraph.NewGraph[RecordingState]()

	g.AddNode("router_node", func(_ context.Context, s *RecordingState) error {
		s.Steps = append(s.Steps, "router_node")
		return nil
	})
	g.AddNode("target_a", func(_ context.Context, s *RecordingState) error {
		s.Steps = append(s.Steps, "target_a")
		return nil
	})

	g.AddEdge(langoraph.START, "router_node")
	g.AddConditionalEdge("router_node",
		func(_ context.Context, s *RecordingState) string {
			return "target_a" // not in mapping, used directly as node name
		},
		map[string]string{},
	)
	g.AddEdge("target_a", langoraph.END)

	state := &RecordingState{}
	if err := g.Run(context.Background(), state); err != nil {
		t.Fatal(err)
	}
	if len(state.Steps) != 2 || state.Steps[1] != "target_a" {
		t.Errorf("expected passthrough to target_a, got %v", state.Steps)
	}
}

// ----- TestGraph_CycleWithExit -----
// A graph with a cycle that eventually exits via a conditional edge.

func TestGraph_CycleWithExit(t *testing.T) {
	g := langoraph.NewGraph[RecordingState]()

	g.AddNode("increment", func(_ context.Context, s *RecordingState) error {
		s.Value++
		s.Steps = append(s.Steps, "increment")
		return nil
	})

	g.AddEdge(langoraph.START, "increment")
	g.AddConditionalEdge("increment",
		func(_ context.Context, s *RecordingState) string {
			if s.Value >= 3 {
				return "done"
			}
			return "again"
		},
		map[string]string{
			"done":  langoraph.END,
			"again": "increment",
		},
	)

	state := &RecordingState{}
	if err := g.Run(context.Background(), state); err != nil {
		t.Fatal(err)
	}
	if state.Value != 3 {
		t.Errorf("expected Value=3, got %d", state.Value)
	}
	if len(state.Steps) != 3 {
		t.Errorf("expected 3 iterations, got %d steps: %v", len(state.Steps), state.Steps)
	}
}

// ----- TestGraph_ErrorRecorder -----
// Graph.Run records errors and continues when state implements ErrorRecorder.

func TestGraph_ErrorRecorder(t *testing.T) {
	g := langoraph.NewGraph[RecordingState]()

	g.AddNode("ok1", func(_ context.Context, s *RecordingState) error {
		s.Steps = append(s.Steps, "ok1")
		return nil
	})
	g.AddNode("fail", func(_ context.Context, s *RecordingState) error {
		return errors.New("boom")
	})
	g.AddNode("ok2", func(_ context.Context, s *RecordingState) error {
		s.Steps = append(s.Steps, "ok2")
		return nil
	})

	g.AddEdge(langoraph.START, "ok1")
	g.AddEdge("ok1", "fail")
	g.AddEdge("fail", "ok2")
	g.AddEdge("ok2", langoraph.END)

	state := &RecordingState{}
	if err := g.Run(context.Background(), state); err != nil {
		t.Fatalf("expected no error with ErrorRecorder, got: %v", err)
	}

	if len(state.Errors) != 1 {
		t.Errorf("expected 1 recorded error, got %v", state.Errors)
	}
	if len(state.Steps) != 2 || state.Steps[0] != "ok1" || state.Steps[1] != "ok2" {
		t.Errorf("expected [ok1 ok2], got %v", state.Steps)
	}
}

// ----- TestGraph_StrictState_StopsOnError -----
// Without ErrorRecorder, the first error halts execution.

func TestGraph_StrictState_StopsOnError(t *testing.T) {
	g := langoraph.NewGraph[StrictState]()

	executed := 0
	g.AddNode("fail", func(_ context.Context, s *StrictState) error {
		executed++
		return errors.New("fatal")
	})
	g.AddNode("after", func(_ context.Context, s *StrictState) error {
		executed++
		return nil
	})

	g.AddEdge(langoraph.START, "fail")
	g.AddEdge("fail", "after")
	g.AddEdge("after", langoraph.END)

	err := g.Run(context.Background(), &StrictState{})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if executed != 1 {
		t.Errorf("expected 1 execution, got %d", executed)
	}
}

// ----- TestGraph_Validate_NoStartEdge -----

func TestGraph_Validate_NoStartEdge(t *testing.T) {
	g := langoraph.NewGraph[StrictState]()
	g.AddNode("a", func(_ context.Context, s *StrictState) error { return nil })

	err := g.Run(context.Background(), &StrictState{})
	if err == nil {
		t.Fatal("expected validation error for missing START edge")
	}
}

// ----- TestGraph_Validate_UnregisteredTarget -----

func TestGraph_Validate_UnregisteredTarget(t *testing.T) {
	g := langoraph.NewGraph[StrictState]()
	g.AddNode("a", func(_ context.Context, s *StrictState) error { return nil })
	g.AddEdge(langoraph.START, "a")
	g.AddEdge("a", "nonexistent") // nonexistent node

	err := g.Run(context.Background(), &StrictState{})
	if err == nil {
		t.Fatal("expected validation error for unregistered target")
	}
}

// ----- TestGraph_Validate_EndWithOutgoing -----

func TestGraph_Validate_EndWithOutgoing(t *testing.T) {
	g := langoraph.NewGraph[StrictState]()
	g.AddNode("a", func(_ context.Context, s *StrictState) error { return nil })
	g.AddEdge(langoraph.START, "a")
	g.AddEdge("a", langoraph.END)
	g.AddEdge(langoraph.END, "a") // END cannot have outgoing

	err := g.Run(context.Background(), &StrictState{})
	if err == nil {
		t.Fatal("expected validation error for END with outgoing edge")
	}
}

// ----- TestGraph_MaxSteps -----
// Infinite cycle hits the step limit.

func TestGraph_MaxSteps(t *testing.T) {
	g := langoraph.NewGraph[StrictState]()
	g.SetMaxSteps(5)

	g.AddNode("loop", func(_ context.Context, s *StrictState) error {
		s.Value++
		return nil
	})
	g.AddEdge(langoraph.START, "loop")
	g.AddEdge("loop", "loop") // infinite cycle

	state := &StrictState{}
	err := g.Run(context.Background(), state)
	if err == nil {
		t.Fatal("expected max-steps error")
	}
	if state.Value != 5 {
		t.Errorf("expected exactly 5 iterations, got %d", state.Value)
	}
}

// ----- TestGraph_ContextCancellation -----

func TestGraph_ContextCancellation(t *testing.T) {
	g := langoraph.NewGraph[StrictState]()

	executed := 0
	g.AddNode("a", func(ctx context.Context, s *StrictState) error {
		executed++
		return ctx.Err()
	})
	g.AddNode("b", func(ctx context.Context, s *StrictState) error {
		executed++
		return nil
	})

	g.AddEdge(langoraph.START, "a")
	g.AddEdge("a", "b")
	g.AddEdge("b", langoraph.END)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := g.Run(ctx, &StrictState{})
	if err == nil {
		t.Fatal("expected context cancellation error")
	}
	if executed != 1 {
		t.Errorf("expected 1 execution, got %d", executed)
	}
}

// ----- TestGraph_ConditionalEdgeFromSTART -----
// START itself can have a conditional edge.

func TestGraph_ConditionalEdgeFromSTART(t *testing.T) {
	g := langoraph.NewGraph[RecordingState]()

	g.AddNode("fast", func(_ context.Context, s *RecordingState) error {
		s.Steps = append(s.Steps, "fast")
		return nil
	})
	g.AddNode("slow", func(_ context.Context, s *RecordingState) error {
		s.Steps = append(s.Steps, "slow")
		return nil
	})

	g.AddConditionalEdge(langoraph.START,
		func(_ context.Context, s *RecordingState) string {
			if s.Value > 100 {
				return "big"
			}
			return "small"
		},
		map[string]string{
			"big":   "slow",
			"small": "fast",
		},
	)
	g.AddEdge("fast", langoraph.END)
	g.AddEdge("slow", langoraph.END)

	t.Run("small", func(t *testing.T) {
		state := &RecordingState{Value: 5}
		if err := g.Run(context.Background(), state); err != nil {
			t.Fatal(err)
		}
		if len(state.Steps) != 1 || state.Steps[0] != "fast" {
			t.Errorf("expected [fast], got %v", state.Steps)
		}
	})

	t.Run("big", func(t *testing.T) {
		state := &RecordingState{Value: 999}
		if err := g.Run(context.Background(), state); err != nil {
			t.Fatal(err)
		}
		if len(state.Steps) != 1 || state.Steps[0] != "slow" {
			t.Errorf("expected [slow], got %v", state.Steps)
		}
	})
}

// ----- TestGraph_NoOutgoingEdge -----
// A node with no outgoing edge produces a runtime error.

func TestGraph_NoOutgoingEdge(t *testing.T) {
	g := langoraph.NewGraph[StrictState]()
	g.AddNode("a", func(_ context.Context, s *StrictState) error { return nil })
	g.AddNode("b", func(_ context.Context, s *StrictState) error { return nil })
	g.AddEdge(langoraph.START, "a")
	g.AddEdge("a", "b")
	// "b" has no outgoing edge

	err := g.Run(context.Background(), &StrictState{})
	if err == nil {
		t.Fatal("expected error for missing outgoing edge from b")
	}
}

// ----- TestGraph_AddNode_ReservedName_Panics -----

func TestGraph_AddNode_ReservedName_Panics(t *testing.T) {
	for _, name := range []string{langoraph.START, langoraph.END} {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Error("expected panic for reserved node name")
				}
			}()
			g := langoraph.NewGraph[StrictState]()
			g.AddNode(name, func(_ context.Context, s *StrictState) error { return nil })
		})
	}
}

// ===================================================================
// Parallel edge tests
// ===================================================================

// parallelState has per-branch fields to avoid data races.
type parallelState struct {
	BranchA int
	BranchB int
	BranchC int
	Result  int
	Steps   []string // only used in single-goroutine contexts
	Errors  []string
}

func (s *parallelState) RecordError(nodeName string, err error) {
	s.Errors = append(s.Errors, nodeName+": "+err.Error())
}

// ----- TestGraph_ParallelEdge_BasicFanOut -----
// Three branches run after "prepare", then "aggregate" combines results.

func TestGraph_ParallelEdge_BasicFanOut(t *testing.T) {
	g := langoraph.NewGraph[parallelState]()

	g.AddNode("prepare", func(_ context.Context, s *parallelState) error {
		s.BranchA = 1
		s.BranchB = 1
		s.BranchC = 1
		return nil
	})
	g.AddNode("double_a", func(_ context.Context, s *parallelState) error {
		s.BranchA *= 2
		return nil
	})
	g.AddNode("triple_b", func(_ context.Context, s *parallelState) error {
		s.BranchB *= 3
		return nil
	})
	g.AddNode("quad_c", func(_ context.Context, s *parallelState) error {
		s.BranchC *= 4
		return nil
	})
	g.AddNode("aggregate", func(_ context.Context, s *parallelState) error {
		s.Result = s.BranchA + s.BranchB + s.BranchC
		return nil
	})

	g.AddEdge(langoraph.START, "prepare")
	g.AddParallelEdge("prepare", []string{"double_a", "triple_b", "quad_c"}, "aggregate")
	g.AddEdge("aggregate", langoraph.END)

	state := &parallelState{}
	if err := g.Run(context.Background(), state); err != nil {
		t.Fatal(err)
	}

	if state.BranchA != 2 {
		t.Errorf("BranchA: expected 2, got %d", state.BranchA)
	}
	if state.BranchB != 3 {
		t.Errorf("BranchB: expected 3, got %d", state.BranchB)
	}
	if state.BranchC != 4 {
		t.Errorf("BranchC: expected 4, got %d", state.BranchC)
	}
	if state.Result != 9 {
		t.Errorf("Result: expected 9 (2+3+4), got %d", state.Result)
	}
}

// ----- TestGraph_ParallelEdge_ActuallyConcurrent -----
// Verifies branches run concurrently, not serially.

func TestGraph_ParallelEdge_ActuallyConcurrent(t *testing.T) {
	g := langoraph.NewGraph[StrictState]()

	var active int64
	var maxActive int64

	makeBranch := func() langoraph.NodeFunc[StrictState] {
		return func(ctx context.Context, s *StrictState) error {
			cur := atomic.AddInt64(&active, 1)
			for {
				old := atomic.LoadInt64(&maxActive)
				if cur <= old || atomic.CompareAndSwapInt64(&maxActive, old, cur) {
					break
				}
			}
			// yield to scheduler to give other goroutines a chance
			done := make(chan struct{})
			go func() { close(done) }()
			<-done
			atomic.AddInt64(&active, -1)
			return nil
		}
	}

	g.AddNode("entry", func(_ context.Context, s *StrictState) error { return nil })
	g.AddNode("b1", makeBranch())
	g.AddNode("b2", makeBranch())
	g.AddNode("b3", makeBranch())
	g.AddNode("done", func(_ context.Context, s *StrictState) error { return nil })

	g.AddEdge(langoraph.START, "entry")
	g.AddParallelEdge("entry", []string{"b1", "b2", "b3"}, "done")
	g.AddEdge("done", langoraph.END)

	if err := g.Run(context.Background(), &StrictState{}); err != nil {
		t.Fatal(err)
	}

	if maxActive < 2 {
		t.Errorf("expected concurrent execution (maxActive>=2), got %d", maxActive)
	}
}

// ----- TestGraph_ParallelEdge_ErrorRecorder -----
// With ErrorRecorder, all branches run and errors are recorded.

func TestGraph_ParallelEdge_ErrorRecorder(t *testing.T) {
	g := langoraph.NewGraph[parallelState]()

	g.AddNode("start_node", func(_ context.Context, s *parallelState) error { return nil })
	g.AddNode("ok_branch", func(_ context.Context, s *parallelState) error {
		s.BranchA = 42
		return nil
	})
	g.AddNode("fail_branch", func(_ context.Context, s *parallelState) error {
		return errors.New("branch failed")
	})
	g.AddNode("after", func(_ context.Context, s *parallelState) error {
		s.Result = 1
		return nil
	})

	g.AddEdge(langoraph.START, "start_node")
	g.AddParallelEdge("start_node", []string{"ok_branch", "fail_branch"}, "after")
	g.AddEdge("after", langoraph.END)

	state := &parallelState{}
	if err := g.Run(context.Background(), state); err != nil {
		t.Fatalf("expected no returned error with ErrorRecorder, got: %v", err)
	}

	if state.BranchA != 42 {
		t.Errorf("ok_branch should have set BranchA=42, got %d", state.BranchA)
	}
	if len(state.Errors) != 1 {
		t.Fatalf("expected 1 recorded error, got %d: %v", len(state.Errors), state.Errors)
	}
	if state.Result != 1 {
		t.Errorf("after node should have run, Result expected 1, got %d", state.Result)
	}
}

// ----- TestGraph_ParallelEdge_StrictError -----
// Without ErrorRecorder, first branch error stops and is returned.

func TestGraph_ParallelEdge_StrictError(t *testing.T) {
	g := langoraph.NewGraph[StrictState]()

	g.AddNode("entry", func(_ context.Context, s *StrictState) error { return nil })
	g.AddNode("ok", func(_ context.Context, s *StrictState) error { return nil })
	g.AddNode("fail", func(_ context.Context, s *StrictState) error {
		return errors.New("branch boom")
	})
	g.AddNode("after", func(_ context.Context, s *StrictState) error {
		s.Value = 999
		return nil
	})

	g.AddEdge(langoraph.START, "entry")
	g.AddParallelEdge("entry", []string{"ok", "fail"}, "after")
	g.AddEdge("after", langoraph.END)

	state := &StrictState{}
	err := g.Run(context.Background(), state)
	if err == nil {
		t.Fatal("expected error from failing branch")
	}
	if state.Value == 999 {
		t.Error("after node should not have run")
	}
}

// ----- TestGraph_ParallelEdge_Validate_EmptyBranches -----

func TestGraph_ParallelEdge_Validate_EmptyBranches(t *testing.T) {
	g := langoraph.NewGraph[StrictState]()
	g.AddNode("a", func(_ context.Context, s *StrictState) error { return nil })
	g.AddEdge(langoraph.START, "a")
	g.AddParallelEdge("a", []string{}, "a") // empty branches

	err := g.Run(context.Background(), &StrictState{})
	if err == nil {
		t.Fatal("expected validation error for empty branches")
	}
}

// ----- TestGraph_ParallelEdge_Validate_UnregisteredBranch -----

func TestGraph_ParallelEdge_Validate_UnregisteredBranch(t *testing.T) {
	g := langoraph.NewGraph[StrictState]()
	g.AddNode("a", func(_ context.Context, s *StrictState) error { return nil })
	g.AddEdge(langoraph.START, "a")
	g.AddParallelEdge("a", []string{"nonexistent"}, langoraph.END)

	err := g.Run(context.Background(), &StrictState{})
	if err == nil {
		t.Fatal("expected validation error for unregistered branch")
	}
}

// ----- TestGraph_ParallelEdge_Validate_UnregisteredThen -----

func TestGraph_ParallelEdge_Validate_UnregisteredThen(t *testing.T) {
	g := langoraph.NewGraph[StrictState]()
	g.AddNode("a", func(_ context.Context, s *StrictState) error { return nil })
	g.AddNode("b", func(_ context.Context, s *StrictState) error { return nil })
	g.AddEdge(langoraph.START, "a")
	g.AddParallelEdge("a", []string{"b"}, "nonexistent")

	err := g.Run(context.Background(), &StrictState{})
	if err == nil {
		t.Fatal("expected validation error for unregistered then target")
	}
}

// ----- TestGraph_ParallelEdge_ThenIsEND -----
// Fan-out branches, then terminate directly.

func TestGraph_ParallelEdge_ThenIsEND(t *testing.T) {
	g := langoraph.NewGraph[parallelState]()

	g.AddNode("entry", func(_ context.Context, s *parallelState) error { return nil })
	g.AddNode("b1", func(_ context.Context, s *parallelState) error {
		s.BranchA = 10
		return nil
	})
	g.AddNode("b2", func(_ context.Context, s *parallelState) error {
		s.BranchB = 20
		return nil
	})

	g.AddEdge(langoraph.START, "entry")
	g.AddParallelEdge("entry", []string{"b1", "b2"}, langoraph.END)

	state := &parallelState{}
	if err := g.Run(context.Background(), state); err != nil {
		t.Fatal(err)
	}
	if state.BranchA != 10 || state.BranchB != 20 {
		t.Errorf("expected BranchA=10, BranchB=20, got %d, %d", state.BranchA, state.BranchB)
	}
}

// ----- TestGraph_ParallelEdge_WithConditionalAfter -----
// Fan-out followed by a conditional edge.

func TestGraph_ParallelEdge_WithConditionalAfter(t *testing.T) {
	g := langoraph.NewGraph[parallelState]()

	g.AddNode("init", func(_ context.Context, s *parallelState) error { return nil })
	g.AddNode("set_a", func(_ context.Context, s *parallelState) error {
		s.BranchA = 5
		return nil
	})
	g.AddNode("set_b", func(_ context.Context, s *parallelState) error {
		s.BranchB = 15
		return nil
	})
	g.AddNode("check", func(_ context.Context, s *parallelState) error {
		s.Result = s.BranchA + s.BranchB
		return nil
	})
	g.AddNode("high", func(_ context.Context, s *parallelState) error {
		s.Steps = append(s.Steps, "high")
		return nil
	})
	g.AddNode("low", func(_ context.Context, s *parallelState) error {
		s.Steps = append(s.Steps, "low")
		return nil
	})

	g.AddEdge(langoraph.START, "init")
	g.AddParallelEdge("init", []string{"set_a", "set_b"}, "check")
	g.AddConditionalEdge("check",
		func(_ context.Context, s *parallelState) string {
			if s.Result >= 10 {
				return "high"
			}
			return "low"
		},
		map[string]string{"high": "high", "low": "low"},
	)
	g.AddEdge("high", langoraph.END)
	g.AddEdge("low", langoraph.END)

	state := &parallelState{}
	if err := g.Run(context.Background(), state); err != nil {
		t.Fatal(err)
	}
	if state.Result != 20 {
		t.Errorf("expected Result=20, got %d", state.Result)
	}
	if len(state.Steps) != 1 || state.Steps[0] != "high" {
		t.Errorf("expected [high], got %v", state.Steps)
	}
}
