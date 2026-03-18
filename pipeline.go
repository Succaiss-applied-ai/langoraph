package langoraph

import (
	"context"
	"fmt"

	"golang.org/x/sync/errgroup"
)

// NodeFunc is the function signature for a pipeline node.
// It receives a pointer to the shared State and mutates it in place.
type NodeFunc[S any] func(ctx context.Context, state *S) error

// ErrorRecorder is an optional interface that State types can implement
// to allow the pipeline to record per-node errors without interrupting execution.
// If State does not implement this interface, a node error will stop the pipeline.
type ErrorRecorder interface {
	RecordError(nodeName string, err error)
}

type node[S any] struct {
	name string
	fn   NodeFunc[S]
}

// Pipeline executes a fixed sequence of nodes against a single State.
// It is the Go equivalent of a linear LangGraph StateGraph with add_edge only.
type Pipeline[S any] struct {
	nodes []node[S]
}

// AddNode registers a node at the end of the pipeline.
func (p *Pipeline[S]) AddNode(name string, fn NodeFunc[S]) {
	p.nodes = append(p.nodes, node[S]{name: name, fn: fn})
}

// Run executes all nodes in registration order against state.
// If state implements ErrorRecorder, node errors are recorded and execution continues.
// Otherwise, the first node error stops execution and is returned.
func (p *Pipeline[S]) Run(ctx context.Context, state *S) error {
	recorder, canRecord := any(state).(ErrorRecorder)
	for _, n := range p.nodes {
		if err := n.fn(ctx, state); err != nil {
			if canRecord {
				recorder.RecordError(n.name, err)
			} else {
				return fmt.Errorf("node %q: %w", n.name, err)
			}
		}
	}
	return nil
}

// RunAll concurrently runs the pipeline against each state in states.
// Each state is independent; mutations do not affect other states.
// Corresponds to: asyncio.gather(*[to_thread(app.invoke, s) for s in states])
func RunAll[S any](ctx context.Context, p *Pipeline[S], states []*S) error {
	g, ctx := errgroup.WithContext(ctx)
	for _, s := range states {
		s := s
		g.Go(func() error {
			return p.Run(ctx, s)
		})
	}
	return g.Wait()
}
