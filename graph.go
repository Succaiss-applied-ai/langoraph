package langoraph

import (
	"context"
	"fmt"
	"sync"
)

// Parallel semantics — DELIBERATELY MATCHES LangGraph
// ---------------------------------------------------
// When ``AddParallelEdge(from, branches, then)`` fires, every branch
// goroutine runs to its natural completion regardless of sibling
// failures. The shared ``ctx`` is **never** cancelled by Graph itself
// (callers can still cancel it externally). After all branches have
// returned, behaviour splits on whether ``State`` implements
// ``ErrorRecorder``:
//
//   * Recorder mode: every branch error is funnelled through
//     ``RecordError(name, err)`` (calls are serialised by an internal
//     mutex so user-supplied recorders need not be thread-safe), and
//     execution continues at ``then``.
//   * Strict mode: the first non-nil branch error in the original
//     ``branches`` slice order is returned and execution halts before
//     ``then`` runs.
//
// This mirrors LangGraph's Pregel runtime, which uses
// ``asyncio.gather(*coros, return_exceptions=True)``-style scheduling
// on its parallel super-step: every parallel task always runs to
// completion, the runtime then aggregates errors deterministically.
// Branches share ``*S`` directly, so callers must still ensure
// concurrent branch functions do not race on overlapping fields — the
// safe pattern is for each branch to mutate disjoint state fields, or
// to write into per-key buckets that an aggregator node merges later
// (see the README for the canonical "per-L0 fan-out" pattern).

const (
	START = "__start__"
	END   = "__end__"

	defaultMaxSteps = 1000
)

// NodeFunc is the function signature for a graph node.
// It receives a pointer to the shared State and mutates it in place.
type NodeFunc[S any] func(ctx context.Context, state *S) error

// ErrorRecorder is an optional interface that State types can implement
// to allow the graph to record per-node errors without interrupting execution.
// If State does not implement this interface, a node error will stop the graph.
type ErrorRecorder interface {
	RecordError(nodeName string, err error)
}

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

// runParallel executes every branch node concurrently and waits for
// all of them to complete before returning, deliberately matching
// LangGraph's Pregel super-step semantics (see the package-level
// "Parallel semantics" doc-block above).
//
//   - With ErrorRecorder: all branches run; errors are funnelled
//     through ``RecordError`` under an internal mutex so user
//     recorders can stay racy-but-simple (e.g. a slice append).
//     ``runParallel`` always returns nil so execution flows to ``then``.
//   - Without ErrorRecorder: all branches still run to completion;
//     ``runParallel`` then returns the first non-nil error in the
//     original ``branches`` slice order, making the returned error
//     deterministic across runs.
//
// The shared ``ctx`` is never cancelled by ``runParallel`` itself —
// sibling branches keep running even after a peer errors. This matches
// LangGraph and avoids leaking in-flight LLM calls when a single sub-
// task fails fast.
func (g *Graph[S]) runParallel(ctx context.Context, state *S, pe parallelEdge, recorder ErrorRecorder, canRecord bool) error {
	branchErrs := make([]error, len(pe.branches))

	var wg sync.WaitGroup
	var recordMu sync.Mutex // serialises recorder.RecordError calls

	for i, b := range pe.branches {
		i, name := i, b
		wg.Add(1)
		go func() {
			defer wg.Done()
			err := g.nodes[name](ctx, state)
			if err == nil {
				return
			}
			if canRecord {
				recordMu.Lock()
				recorder.RecordError(name, err)
				recordMu.Unlock()
				return
			}
			branchErrs[i] = fmt.Errorf("node %q: %w", name, err)
		}()
	}
	wg.Wait()

	if canRecord {
		return nil
	}
	for _, err := range branchErrs {
		if err != nil {
			return err
		}
	}
	return nil
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

// RunAll concurrently runs the graph against each state in states.
// Each state is independent; mutations do not affect other states.
//
// Parallel semantics — DELIBERATELY MATCHES LangGraph
// ---------------------------------------------------
// Every state's graph runs to its natural completion regardless of
// sibling failures. The shared ``ctx`` is **never** cancelled by
// RunAll itself; an external cancellation of ``ctx`` still aborts every
// in-flight graph. After all graphs return, RunAll returns the first
// non-nil error in the order states appear in the input slice.
//
// This matches the "wait-all, deterministic-first-error" contract used
// by ``Fanout`` and ``Graph.runParallel``, so callers get the same
// behaviour whether they fan out items, parallel-edge branches, or
// whole graphs.
func RunAll[S any](ctx context.Context, g *Graph[S], states []*S) error {
	errs := make([]error, len(states))

	var wg sync.WaitGroup
	for i, s := range states {
		i, s := i, s
		wg.Add(1)
		go func() {
			defer wg.Done()
			errs[i] = g.Run(ctx, s)
		}()
	}
	wg.Wait()

	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	return nil
}
