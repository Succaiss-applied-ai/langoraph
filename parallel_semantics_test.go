// Tests pinning langoraph's parallel semantics to LangGraph parity:
//
//   1. All siblings run to completion even when a peer errors
//      (no errgroup-style early cancellation).
//   2. The first error in input/branch order wins, deterministically.
//   3. ErrorRecorder.RecordError is called under an internal mutex so
//      user recorders do not need their own locking.
//   4. RunAll obeys the same wait-all + deterministic-first-error
//      contract as Fanout / Graph.runParallel.
//
// These tests are the load-bearing contract for the "all parallelism
// matches LangGraph" guarantee documented in graph.go and fanout.go.
package langoraph_test

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	langoraph "github.com/Succaiss-applied-ai/langoraph"
)

// ---------------------------------------------------------------------------
// Fanout: wait-all on first error
// ---------------------------------------------------------------------------

func TestFanout_WaitsAllSiblingsOnError(t *testing.T) {
	const n = 6
	items := make([]int, n)
	for i := range items {
		items[i] = i
	}

	var completed int64
	sentinel := errors.New("item 2 boom")

	_, err := langoraph.Fanout(
		context.Background(),
		items,
		func(ctx context.Context, item int) (int, error) {
			if item == 2 {
				return 0, sentinel
			}
			// Simulate work that takes a real beat. Without wait-all
			// guarantees, a context-cancelling Fanout would short-circuit
			// these and `completed` would not reach n-1.
			select {
			case <-time.After(50 * time.Millisecond):
			case <-ctx.Done():
				return 0, ctx.Err()
			}
			atomic.AddInt64(&completed, 1)
			return item * item, nil
		},
	)

	if !errors.Is(err, sentinel) {
		t.Fatalf("expected sentinel error, got: %v", err)
	}
	if got := atomic.LoadInt64(&completed); got != int64(n-1) {
		t.Errorf("expected %d siblings to complete, got %d", n-1, got)
	}
}

func TestFanout_DeterministicFirstError(t *testing.T) {
	items := []string{"a", "b", "c", "d", "e"}

	for trial := 0; trial < 20; trial++ {
		_, err := langoraph.Fanout(
			context.Background(),
			items,
			func(_ context.Context, item string) (int, error) {
				if item == "b" {
					return 0, errors.New("b failed")
				}
				if item == "d" {
					return 0, errors.New("d failed")
				}
				return 0, nil
			},
		)
		if err == nil || err.Error() != "b failed" {
			t.Fatalf("trial %d: expected 'b failed' (first in input order), got %v", trial, err)
		}
	}
}

func TestFanout_PartialResultsOnError(t *testing.T) {
	items := []int{1, 2, 3}

	results, err := langoraph.Fanout(
		context.Background(),
		items,
		func(_ context.Context, item int) (int, error) {
			if item == 2 {
				return 0, errors.New("boom")
			}
			return item * 10, nil
		},
	)
	if err == nil {
		t.Fatal("expected error")
	}
	if results[0] != 10 {
		t.Errorf("results[0]: expected 10, got %d", results[0])
	}
	if results[2] != 30 {
		t.Errorf("results[2]: expected 30, got %d", results[2])
	}
}

// ---------------------------------------------------------------------------
// Graph.runParallel: wait-all on first error (strict mode)
// ---------------------------------------------------------------------------

type pState struct {
	completed int64
	errors    []string
}

func (s *pState) RecordError(name string, err error) {
	s.errors = append(s.errors, fmt.Sprintf("%s: %v", name, err))
}

func TestGraph_ParallelEdge_WaitsAllSiblings_Strict(t *testing.T) {
	// Strict (no recorder) state — int with no methods.
	type strictState struct{ Done int64 }

	g := langoraph.NewGraph[strictState]()
	g.AddNode("entry", func(_ context.Context, _ *strictState) error { return nil })
	g.AddNode("ok_a", func(ctx context.Context, s *strictState) error {
		select {
		case <-time.After(40 * time.Millisecond):
		case <-ctx.Done():
			return ctx.Err()
		}
		atomic.AddInt64(&s.Done, 1)
		return nil
	})
	g.AddNode("fail", func(_ context.Context, _ *strictState) error {
		return errors.New("fail boom")
	})
	g.AddNode("ok_b", func(ctx context.Context, s *strictState) error {
		select {
		case <-time.After(40 * time.Millisecond):
		case <-ctx.Done():
			return ctx.Err()
		}
		atomic.AddInt64(&s.Done, 1)
		return nil
	})
	g.AddNode("after", func(_ context.Context, _ *strictState) error { return nil })

	g.AddEdge(langoraph.START, "entry")
	g.AddParallelEdge("entry", []string{"ok_a", "fail", "ok_b"}, "after")
	g.AddEdge("after", langoraph.END)

	state := &strictState{}
	err := g.Run(context.Background(), state)
	if err == nil {
		t.Fatal("expected error from failing branch")
	}
	if got := atomic.LoadInt64(&state.Done); got != 2 {
		t.Errorf("expected both ok_* branches to complete (Done=2), got %d", got)
	}
}

func TestGraph_ParallelEdge_DeterministicFirstError(t *testing.T) {
	type strictState struct{}

	g := langoraph.NewGraph[strictState]()
	g.AddNode("entry", func(_ context.Context, _ *strictState) error { return nil })
	g.AddNode("ok", func(_ context.Context, _ *strictState) error { return nil })
	g.AddNode("err_b", func(_ context.Context, _ *strictState) error { return errors.New("b") })
	g.AddNode("err_d", func(_ context.Context, _ *strictState) error { return errors.New("d") })
	g.AddNode("after", func(_ context.Context, _ *strictState) error { return nil })

	g.AddEdge(langoraph.START, "entry")
	g.AddParallelEdge("entry", []string{"ok", "err_b", "ok", "err_d"}, "after")
	g.AddEdge("after", langoraph.END)

	for trial := 0; trial < 20; trial++ {
		err := g.Run(context.Background(), &strictState{})
		if err == nil || !contains(err.Error(), `node "err_b"`) {
			t.Fatalf("trial %d: expected first error from err_b, got %v", trial, err)
		}
	}
}

// ---------------------------------------------------------------------------
// Graph.runParallel: ErrorRecorder is serialised
// ---------------------------------------------------------------------------

func TestGraph_ParallelEdge_ErrorRecorder_Serialised(t *testing.T) {
	g := langoraph.NewGraph[pState]()

	// 8 branches all error simultaneously — without internal locking, the
	// pState.errors slice append would race under -race.
	g.AddNode("entry", func(_ context.Context, _ *pState) error { return nil })
	branches := []string{}
	for i := 0; i < 8; i++ {
		name := fmt.Sprintf("fail_%d", i)
		branches = append(branches, name)
		g.AddNode(name, func(_ context.Context, _ *pState) error {
			return fmt.Errorf("err from %s", name)
		})
	}
	g.AddNode("after", func(_ context.Context, s *pState) error {
		atomic.AddInt64(&s.completed, 1)
		return nil
	})

	g.AddEdge(langoraph.START, "entry")
	g.AddParallelEdge("entry", branches, "after")
	g.AddEdge("after", langoraph.END)

	state := &pState{}
	if err := g.Run(context.Background(), state); err != nil {
		t.Fatalf("expected no returned error with ErrorRecorder, got: %v", err)
	}
	if len(state.errors) != 8 {
		t.Errorf("expected 8 recorded errors, got %d: %v", len(state.errors), state.errors)
	}
	if state.completed != 1 {
		t.Errorf("expected after node to run exactly once, got %d", state.completed)
	}
}

// ---------------------------------------------------------------------------
// RunAll: wait-all + deterministic first error
// ---------------------------------------------------------------------------

func TestRunAll_WaitsAllStatesOnError(t *testing.T) {
	type qState struct {
		Tag  string
		Done bool
	}

	g := langoraph.NewGraph[qState]()
	g.AddNode("work", func(ctx context.Context, s *qState) error {
		if s.Tag == "boom" {
			return errors.New("intentional fail")
		}
		select {
		case <-time.After(40 * time.Millisecond):
		case <-ctx.Done():
			return ctx.Err()
		}
		s.Done = true
		return nil
	})
	g.AddEdge(langoraph.START, "work")
	g.AddEdge("work", langoraph.END)

	states := []*qState{
		{Tag: "ok_1"},
		{Tag: "boom"},
		{Tag: "ok_2"},
		{Tag: "ok_3"},
	}
	err := langoraph.RunAll(context.Background(), g, states)
	if err == nil {
		t.Fatal("expected error from boom state")
	}
	if !states[0].Done || !states[2].Done || !states[3].Done {
		t.Errorf("expected all non-failing states to complete, got: %+v", states)
	}
}

func TestRunAll_DeterministicFirstError(t *testing.T) {
	type qState struct{ Tag string }

	g := langoraph.NewGraph[qState]()
	g.AddNode("work", func(_ context.Context, s *qState) error {
		if s.Tag == "x" || s.Tag == "y" {
			return errors.New(s.Tag)
		}
		return nil
	})
	g.AddEdge(langoraph.START, "work")
	g.AddEdge("work", langoraph.END)

	for trial := 0; trial < 20; trial++ {
		states := []*qState{{Tag: "a"}, {Tag: "x"}, {Tag: "b"}, {Tag: "y"}}
		err := langoraph.RunAll(context.Background(), g, states)
		if err == nil || !contains(err.Error(), "x") {
			t.Fatalf("trial %d: expected first error to come from state 'x', got: %v", trial, err)
		}
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func contains(haystack, needle string) bool {
	for i := 0; i+len(needle) <= len(haystack); i++ {
		if haystack[i:i+len(needle)] == needle {
			return true
		}
	}
	return false
}
