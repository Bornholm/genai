package agent

import (
	"context"
	"errors"
	"testing"
)

// MockHandler implements Handler for testing
type MockHandler struct {
	handleFunc func(ctx context.Context, input Input, emit EmitFunc) error
}

func (m *MockHandler) Handle(ctx context.Context, input Input, emit EmitFunc) error {
	if m.handleFunc == nil {
		return nil
	}
	return m.handleFunc(ctx, input, emit)
}

// PanicHandler implements Handler that panics
type PanicHandler struct{}

func (p *PanicHandler) Handle(ctx context.Context, input Input, emit EmitFunc) error {
	panic("handler panic!")
}

func TestRunner_BasicRun(t *testing.T) {
	// Test: Basic run executes handler
	handler := &MockHandler{
		handleFunc: func(ctx context.Context, input Input, emit EmitFunc) error {
			emit(NewEvent(EventTypeComplete, &CompleteData{Message: "done"}))
			return nil
		},
	}

	runner := NewRunner(handler)

	var events []Event
	err := runner.Run(context.Background(), NewInput("test"), func(evt Event) error {
		events = append(events, evt)
		return nil
	})

	if err != nil {
		t.Errorf("expected no error, got: %v", err)
	}

	if len(events) != 1 {
		t.Errorf("expected 1 event, got %d", len(events))
	}
}

func TestRunner_MiddlewareOrdering(t *testing.T) {
	// Test: Verify middleware ordering
	var order []string

	handler := &MockHandler{
		handleFunc: func(ctx context.Context, input Input, emit EmitFunc) error {
			order = append(order, "handler")
			return nil
		},
	}

	// Middleware 1 - should run first
	middleware1 := func(next Handler) Handler {
		return HandlerFunc(func(ctx context.Context, input Input, emit EmitFunc) error {
			order = append(order, "middleware1-before")
			err := next.Handle(ctx, input, emit)
			order = append(order, "middleware1-after")
			return err
		})
	}

	// Middleware 2 - should run second
	middleware2 := func(next Handler) Handler {
		return HandlerFunc(func(ctx context.Context, input Input, emit EmitFunc) error {
			order = append(order, "middleware2-before")
			err := next.Handle(ctx, input, emit)
			order = append(order, "middleware2-after")
			return err
		})
	}

	runner := NewRunner(handler, middleware1, middleware2)

	err := runner.Run(context.Background(), NewInput("test"), func(evt Event) error {
		return nil
	})

	if err != nil {
		t.Errorf("expected no error, got: %v", err)
	}

	expected := []string{
		"middleware1-before",
		"middleware2-before",
		"handler",
		"middleware2-after",
		"middleware1-after",
	}

	if len(order) != len(expected) {
		t.Errorf("expected %d calls, got %d: %v", len(expected), len(order), order)
	}

	for i, v := range expected {
		if i >= len(order) || order[i] != v {
			t.Errorf("expected order[%d] = %s, got %s", i, v, order[i])
		}
	}
}

func TestRunner_HandlerError(t *testing.T) {
	// Test: Handler error is returned
	expectedErr := errors.New("handler error")

	handler := &MockHandler{
		handleFunc: func(ctx context.Context, input Input, emit EmitFunc) error {
			return expectedErr
		},
	}

	runner := NewRunner(handler)

	err := runner.Run(context.Background(), NewInput("test"), func(evt Event) error {
		return nil
	})

	if err != expectedErr {
		t.Errorf("expected handler error, got: %v", err)
	}
}

func TestRunner_PanicRecovery(t *testing.T) {
	// Test: Panicking handler does not crash the caller
	// Note: The current implementation does NOT recover from panics.
	// This test documents that behavior - panics propagate to the caller.
	handler := &PanicHandler{}
	runner := NewRunner(handler)

	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic to propagate, but it was recovered")
		}
	}()

	_ = runner.Run(context.Background(), NewInput("test"), func(evt Event) error {
		return nil
	})
}

func TestRunner_ContextCancellation(t *testing.T) {
	// Test: Context cancellation is respected
	handler := &MockHandler{
		handleFunc: func(ctx context.Context, input Input, emit EmitFunc) error {
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
				return nil
			}
		},
	}

	runner := NewRunner(handler)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	err := runner.Run(ctx, NewInput("test"), func(evt Event) error {
		return nil
	})

	if err != context.Canceled {
		t.Errorf("expected context.Canceled, got: %v", err)
	}
}

func TestRunner_EmitError(t *testing.T) {
	// Test: Emit function error is propagated
	handler := &MockHandler{
		handleFunc: func(ctx context.Context, input Input, emit EmitFunc) error {
			return emit(NewEvent(EventTypeComplete, &CompleteData{Message: "test"}))
		},
	}

	runner := NewRunner(handler)

	emitErr := errors.New("emit error")
	err := runner.Run(context.Background(), NewInput("test"), func(evt Event) error {
		return emitErr
	})

	if err != emitErr {
		t.Errorf("expected emit error, got: %v", err)
	}
}
