package langoraph

import (
	"context"
	"sync"
)

// ItemFunc is the function signature for a single fan-out task.
// It receives one item and returns one output.
type ItemFunc[Item, Output any] func(ctx context.Context, item Item) (Output, error)

// Fanout concurrently processes each item using fn, then collects results in
// the original input order.
//
// This is the Go equivalent of:
//   - LangGraph Send() fan-out from parse_input
//   - operator.add reducer that accumulates item_results
//
// Parallel semantics — DELIBERATELY MATCHES LangGraph
// ---------------------------------------------------
// Every goroutine runs to its natural completion regardless of sibling
// failures. The shared ``ctx`` is **never** cancelled by Fanout itself
// (callers can still cancel it externally). After all goroutines have
// returned, Fanout returns the first non-nil error encountered, in the
// order items appear in the input slice — making the returned error
// deterministic across runs.
//
// This mirrors Python's ``asyncio.gather(*coros, return_exceptions=False)``
// behaviour at the time the awaiter raises: the gather call propagates
// the first exception, but **does not** cancel sibling tasks. We extend
// that contract slightly by waiting for siblings to finish before
// returning, so the caller never observes leaked in-flight goroutines.
//
// Successfully completed results are written to the output slice
// regardless of whether any sibling errored, so callers may inspect
// the partial output alongside the returned error.
func Fanout[Item, Output any](ctx context.Context, items []Item, fn ItemFunc[Item, Output]) ([]Output, error) {
	results := make([]Output, len(items))
	errs := make([]error, len(items))

	var wg sync.WaitGroup
	for i, item := range items {
		i, item := i, item
		wg.Add(1)
		go func() {
			defer wg.Done()
			out, err := fn(ctx, item)
			if err != nil {
				errs[i] = err
				return
			}
			results[i] = out
		}()
	}
	wg.Wait()

	for _, err := range errs {
		if err != nil {
			return results, err
		}
	}
	return results, nil
}
