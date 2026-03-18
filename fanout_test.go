package langoraph_test

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"

	langoraph "github.com/zhaocun/langoraph"
)

// ----- TestFanout_ResultsInOrder -----
// Output slice must match input order, regardless of goroutine scheduling.

func TestFanout_ResultsInOrder(t *testing.T) {
	items := []int{3, 1, 4, 1, 5, 9, 2, 6}

	results, err := langoraph.Fanout(
		context.Background(),
		items,
		func(_ context.Context, n int) (int, error) {
			return n * n, nil
		},
	)
	if err != nil {
		t.Fatal(err)
	}

	if len(results) != len(items) {
		t.Fatalf("expected %d results, got %d", len(items), len(results))
	}
	for i, item := range items {
		expected := item * item
		if results[i] != expected {
			t.Errorf("results[%d]: expected %d, got %d", i, expected, results[i])
		}
	}
}

// ----- TestFanout_ActuallyParallel -----
// Verifies that Fanout runs items concurrently.

func TestFanout_ActuallyParallel(t *testing.T) {
	var active int64
	var maxActive int64

	items := make([]int, 20)
	for i := range items {
		items[i] = i
	}

	_, err := langoraph.Fanout(
		context.Background(),
		items,
		func(ctx context.Context, n int) (int, error) {
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
			return n, nil
		},
	)
	if err != nil {
		t.Fatal(err)
	}

	if maxActive < 2 {
		t.Errorf("expected concurrent execution (maxActive>=2), got %d", maxActive)
	}
}

// ----- TestFanout_ErrorStopsAndPropagates -----
// If any item returns an error, Fanout returns that error.

func TestFanout_ErrorStopsAndPropagates(t *testing.T) {
	items := []int{1, 2, 3, 4, 5}
	sentinel := errors.New("item error")

	_, err := langoraph.Fanout(
		context.Background(),
		items,
		func(_ context.Context, n int) (int, error) {
			if n == 3 {
				return 0, sentinel
			}
			return n, nil
		},
	)

	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !errors.Is(err, sentinel) {
		t.Errorf("expected sentinel error, got: %v", err)
	}
}

// ----- TestFanout_EmptyInput -----
// Fanout with zero items returns an empty slice, not nil.

func TestFanout_EmptyInput(t *testing.T) {
	results, err := langoraph.Fanout(
		context.Background(),
		[]string{},
		func(_ context.Context, s string) (string, error) {
			return s, nil
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if results == nil {
		t.Error("expected non-nil empty slice, got nil")
	}
	if len(results) != 0 {
		t.Errorf("expected 0 results, got %d", len(results))
	}
}

// ----- TestFanout_StructItems -----
// Fanout works with struct input/output types, matching the review pattern.

type ReviewItem struct {
	Index    int
	Question string
}

type ReviewOutput struct {
	Index  int
	Passed bool
	Score  float64
}

func TestFanout_StructItems(t *testing.T) {
	items := []ReviewItem{
		{Index: 0, Question: "What is a goroutine?"},
		{Index: 1, Question: "Explain REST."},
		{Index: 2, Question: "What is a mutex?"},
	}

	results, err := langoraph.Fanout(
		context.Background(),
		items,
		func(_ context.Context, item ReviewItem) (ReviewOutput, error) {
			return ReviewOutput{
				Index:  item.Index,
				Passed: true,
				Score:  0.85,
			}, nil
		},
	)
	if err != nil {
		t.Fatal(err)
	}

	for i, r := range results {
		if r.Index != i {
			t.Errorf("results[%d].Index = %d, want %d", i, r.Index, i)
		}
		if !r.Passed {
			t.Errorf("results[%d].Passed = false", i)
		}
	}
}
