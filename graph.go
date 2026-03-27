package langoraph

import (
	"context"
	"fmt"
	"sync"

	"golang.org/x/sync/errgroup"
)

const (
	START = "__start__"
	END   = "__end__"

	defaultMaxSteps = 1000
)

// RouterFunc examines state and returns a routing key that determines the
// next node to execute. The returned string is looked up in the mapping
// supplied to AddConditionalEdge; if not found, it is used directly as a
// node name.
type RouterFunc[S any] func(ctx context.Context, state *S) string

type conditionalEdge[S any] struct {
	router  RouterFunc[S]
	mapping map[string]string
}

type parallelEdge struct {
	branches []string
	then     string
}

// Graph executes nodes connected by directed edges, supporting both fixed
// and conditional routing. It is the Go equivalent of a LangGraph StateGraph
// with add_edge and add_conditional_edges.
type Graph[S any] struct {
	nodes         map[string]NodeFunc[S]
	edges         map[string]string
	condEdges     map[string]conditionalEdge[S]
	parallelEdges map[string]parallelEdge
	maxSteps      int
}

// NewGraph creates an empty Graph ready for node and edge registration.
func NewGraph[S any]() *Graph[S] {
	return &Graph[S]{
		nodes:         make(map[string]NodeFunc[S]),
		edges:         make(map[string]string),
		condEdges:     make(map[string]conditionalEdge[S]),
		parallelEdges: make(map[string]parallelEdge),
		maxSteps:      defaultMaxSteps,
	}
}

// SetMaxSteps overrides the default cycle-protection limit (1000).
// Run returns an error if execution exceeds this many node invocations.
func (g *Graph[S]) SetMaxSteps(n int) {
	g.maxSteps = n
}

// AddNode registers a named node. START and END are reserved and cannot be
// used as node names.
func (g *Graph[S]) AddNode(name string, fn NodeFunc[S]) {
	if name == START || name == END {
		panic(fmt.Sprintf("langoraph: %q is a reserved name", name))
	}
	g.nodes[name] = fn
}

// AddEdge adds a fixed directed edge from one node to another.
// Use START as from to set the entry point; use END as to to mark a terminal.
func (g *Graph[S]) AddEdge(from, to string) {
	g.edges[from] = to
}

// AddConditionalEdge registers a conditional edge from the source node.
// After the source executes, router is called with the current state.
// The returned key is looked up in mapping to find the next node name.
// If the key is absent from mapping, it is used directly as the node name
// (matching LangGraph Python's passthrough behavior).
func (g *Graph[S]) AddConditionalEdge(from string, router RouterFunc[S], mapping map[string]string) {
	g.condEdges[from] = conditionalEdge[S]{router: router, mapping: mapping}
}

// AddParallelEdge adds a fan-out edge: after from executes, all branches run
// concurrently on the shared state. Once every branch completes, execution
// continues at the then node.
//
// Branches share the same *S pointer — callers must ensure concurrent branch
// functions do not race on overlapping state fields.
func (g *Graph[S]) AddParallelEdge(from string, branches []string, then string) {
	g.parallelEdges[from] = parallelEdge{branches: branches, then: then}
}

// Run executes the graph starting from the START edge, following fixed and
// conditional edges until END is reached.
//
// If state implements ErrorRecorder, node errors are recorded and execution
// continues along the outgoing edge. Otherwise the first node error halts
// execution.
func (g *Graph[S]) Run(ctx context.Context, state *S) error {
	if err := g.validate(); err != nil {
		return err
	}

	recorder, canRecord := any(state).(ErrorRecorder)

	var current string
	if pe, ok := g.parallelEdges[START]; ok {
		if err := g.runParallel(ctx, state, pe, recorder, canRecord); err != nil {
			return err
		}
		current = pe.then
	} else {
		current = g.resolveNext(START, ctx, state)
	}

	for step := 0; current != END; step++ {
		if step >= g.maxSteps {
			return fmt.Errorf("langoraph: exceeded %d steps (possible cycle)", g.maxSteps)
		}

		fn, ok := g.nodes[current]
		if !ok {
			return fmt.Errorf("langoraph: node %q not registered", current)
		}

		if err := fn(ctx, state); err != nil {
			if canRecord {
				recorder.RecordError(current, err)
			} else {
				return fmt.Errorf("node %q: %w", current, err)
			}
		}

		if pe, ok := g.parallelEdges[current]; ok {
			if err := g.runParallel(ctx, state, pe, recorder, canRecord); err != nil {
				return err
			}
			step += len(pe.branches)
			current = pe.then
			continue
		}

		next := g.resolveNext(current, ctx, state)
		if next == "" {
			return fmt.Errorf("langoraph: node %q has no outgoing edge", current)
		}
		current = next
	}
	return nil
}

// runParallel executes all branch nodes concurrently.
// With ErrorRecorder: all branches run to completion, errors are recorded.
// Without ErrorRecorder: first error cancels remaining branches.
func (g *Graph[S]) runParallel(ctx context.Context, state *S, pe parallelEdge, recorder ErrorRecorder, canRecord bool) error {
	if canRecord {
		var wg sync.WaitGroup
		for _, b := range pe.branches {
			wg.Add(1)
			go func(name string) {
				defer wg.Done()
				if err := g.nodes[name](ctx, state); err != nil {
					recorder.RecordError(name, err)
				}
			}(b)
		}
		wg.Wait()
		return nil
	}

	eg, branchCtx := errgroup.WithContext(ctx)
	for _, b := range pe.branches {
		name := b
		eg.Go(func() error {
			if err := g.nodes[name](branchCtx, state); err != nil {
				return fmt.Errorf("node %q: %w", name, err)
			}
			return nil
		})
	}
	return eg.Wait()
}

// resolveNext returns the next node name after from. It checks fixed edges
// first, then conditional edges. Returns "" if no edge is found.
func (g *Graph[S]) resolveNext(from string, ctx context.Context, state *S) string {
	if target, ok := g.edges[from]; ok {
		return target
	}
	if ce, ok := g.condEdges[from]; ok {
		key := ce.router(ctx, state)
		if mapped, ok := ce.mapping[key]; ok {
			return mapped
		}
		return key
	}
	return ""
}

// validate checks the graph structure before execution.
func (g *Graph[S]) validate() error {
	if _, hasStart := g.edges[START]; !hasStart {
		if _, hasCondStart := g.condEdges[START]; !hasCondStart {
			if _, hasParStart := g.parallelEdges[START]; !hasParStart {
				return fmt.Errorf("langoraph: no edge from START; call AddEdge(START, ...), AddConditionalEdge(START, ...), or AddParallelEdge(START, ...)")
			}
		}
	}

	check := func(target, context string) error {
		if target == START {
			return fmt.Errorf("langoraph: edge from %s points to START (invalid)", context)
		}
		if target == END {
			return nil
		}
		if _, ok := g.nodes[target]; !ok {
			return fmt.Errorf("langoraph: edge from %s targets unregistered node %q", context, target)
		}
		return nil
	}

	for from, to := range g.edges {
		if from == END {
			return fmt.Errorf("langoraph: END cannot have outgoing edges")
		}
		if err := check(to, fmt.Sprintf("%q", from)); err != nil {
			return err
		}
	}

	for from, ce := range g.condEdges {
		if from == END {
			return fmt.Errorf("langoraph: END cannot have outgoing edges")
		}
		for key, to := range ce.mapping {
			if err := check(to, fmt.Sprintf("%q (key %q)", from, key)); err != nil {
				return err
			}
		}
	}

	for from, pe := range g.parallelEdges {
		if from == END {
			return fmt.Errorf("langoraph: END cannot have outgoing edges")
		}
		if len(pe.branches) == 0 {
			return fmt.Errorf("langoraph: parallel edge from %q has no branches", from)
		}
		for _, b := range pe.branches {
			if err := check(b, fmt.Sprintf("%q (parallel branch)", from)); err != nil {
				return err
			}
		}
		if err := check(pe.then, fmt.Sprintf("%q (parallel then)", from)); err != nil {
			return err
		}
	}

	return nil
}
