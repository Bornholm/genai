package agent

import (
	"context"
)

// EmitFunc is the function type for emitting events during handler execution
type EmitFunc func(Event) error

// Handler is the interface for handling agent inputs
type Handler interface {
	Handle(ctx context.Context, input Input, emit EmitFunc) error
}

// HandlerFunc is an adapter to allow the use of ordinary functions as Handlers
type HandlerFunc func(ctx context.Context, input Input, emit EmitFunc) error

// Handle implements Handler
func (fn HandlerFunc) Handle(ctx context.Context, input Input, emit EmitFunc) error {
	return fn(ctx, input, emit)
}
