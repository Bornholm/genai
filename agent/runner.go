package agent

import (
	"context"
)

// Runner is a synchronous agent runner that executes a handler with middleware
type Runner struct {
	handler     Handler
	middlewares []Middleware
}

// NewRunner creates a new Runner with the given handler and optional middlewares
func NewRunner(handler Handler, middlewares ...Middleware) *Runner {
	return &Runner{
		handler:     handler,
		middlewares: middlewares,
	}
}

// Run executes the handler with the given input and emit function.
// It chains the middlewares and calls Handle synchronously.
// Cancellation is through the context.
func (r *Runner) Run(ctx context.Context, input Input, emit EmitFunc) error {
	// Chain middlewares
	handler := chain(r.handler, r.middlewares)

	// Execute the handler
	return handler.Handle(ctx, input, emit)
}
