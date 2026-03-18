package langoraph

import (
	"context"

	"golang.org/x/sync/errgroup"
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
// If any item returns an error, Fanout waits for all in-flight items to finish
// and returns the first error. Successfully completed results up to that point
// are still written to the output slice.
func Fanout[Item, Output any](ctx context.Context, items []Item, fn ItemFunc[Item, Output]) ([]Output, error) {
	results := make([]Output, len(items))
	g, ctx := errgroup.WithContext(ctx)
	for i, item := range items {
		i, item := i, item
		g.Go(func() error {
			out, err := fn(ctx, item)
			if err != nil {
				return err
			}
			results[i] = out
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return nil, err
	}
	return results, nil
}
