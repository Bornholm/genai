package agent

import (
	"context"
	"sync"
	"sync/atomic"
	"time"

	"github.com/pkg/errors"
)

// Agent is a thread-safe version of the Agent that eliminates race conditions
type Agent struct {
	handler Handler
	options *Options

	// Channel management
	inputs  chan Event
	outputs chan Event
	errors  chan error

	// State management
	state int32 // 0: stopped, 1: starting, 2: running, 3: stopping

	// Synchronization
	startOnce sync.Once
	stopOnce  sync.Once

	// Context for lifecycle management
	ctx    context.Context
	cancel context.CancelFunc

	// Wait group for graceful shutdown
	wg sync.WaitGroup
}

// Agent states
const (
	stateStopped int32 = iota
	stateStarting
	stateRunning
	stateStopping
)

// New creates a new thread-safe agent
func New(handler Handler, opts ...Option) *Agent {
	ctx, cancel := context.WithCancel(context.Background())

	options := &Options{}
	for _, o := range opts {
		o(options)
	}
	options.setDefaults()

	return &Agent{
		handler: chain(handler, options.Middlewares),
		options: options,
		state:   stateStopped,
		ctx:     ctx,
		cancel:  cancel,
	}
}

// Output returns a read-only channel for receiving agent outputs
func (a *Agent) Output() <-chan Event {
	return a.outputs
}

// Err returns a read-only channel for receiving agent errors.
func (a *Agent) Err() <-chan error {
	return a.errors
}

// In sends an event to the agent for processing
func (a *Agent) In(evt Event) error {
	if !a.IsRunning() {
		return errors.New("agent is not running")
	}

	select {
	case a.inputs <- evt:
		return nil
	case <-a.ctx.Done():
		return errors.New("agent is shutting down")
	case <-time.After(a.options.InputTimeout):
		return errors.New("agent input buffer is full")
	}
}

// Start starts the agent and processes events until the context is cancelled
func (a *Agent) Start(ctx context.Context) (<-chan Event, <-chan error, error) {
	var err error
	a.startOnce.Do(func() {
		err = a.doRun(ctx)
	})

	if err != nil {
		return nil, nil, err
	}

	return a.outputs, a.errors, nil
}

// Stop gracefully stops the agent
func (a *Agent) Stop() error {
	var stopErr error
	a.stopOnce.Do(func() {
		stopErr = a.doStop()
	})
	return stopErr
}

// IsRunning returns true if the agent is currently running
func (a *Agent) IsRunning() bool {
	return atomic.LoadInt32(&a.state) == stateRunning
}

// doRun performs the actual agent startup and execution
func (a *Agent) doRun(ctx context.Context) error {
	if !atomic.CompareAndSwapInt32(&a.state, stateStopped, stateStarting) {
		return errors.New("agent is already running or starting")
	}

	a.inputs = make(chan Event, a.options.InputChanSize)
	a.outputs = make(chan Event, a.options.OutputChanSize)
	a.errors = make(chan error, a.options.ErrorChanSize)

	// Link the external context with the agent's internal context
	go func() {
		<-ctx.Done()
		a.Stop()
	}()

	atomic.StoreInt32(&a.state, stateRunning)

	a.wg.Add(1)
	go func() {
		defer a.wg.Done()
		defer close(a.inputs)
		defer close(a.outputs)
		defer close(a.errors)
		defer atomic.StoreInt32(&a.state, stateStopped)
		a.processEvents(a.ctx)
	}()

	return nil
}

// doStop performs the actual agent shutdown
func (a *Agent) doStop() error {
	if !atomic.CompareAndSwapInt32(&a.state, stateRunning, stateStopping) {
		return nil // Already stopped or stopping
	}

	// Cancel context to signal shutdown
	a.cancel()

	// Wait for processing to complete
	a.wg.Wait()

	return nil
}

// processEvents processes events in the main agent loop
func (a *Agent) processEvents(ctx context.Context) {
	var wg sync.WaitGroup
	for i := 0; i < a.options.ConcurrentHandlers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			a.worker(ctx)
		}()
	}
	wg.Wait()
}

// handle processes a single event through the handler
func (a *Agent) worker(ctx context.Context) {
	for {
		select {
		case evt, ok := <-a.inputs:
			if !ok {
				return
			}
			a.handle(evt)
		case <-ctx.Done():
			return
		}
	}
}

func (a *Agent) handle(input Event) {
	ctx := input.Context()
	ctx = WithContextAgent(ctx, a)

	input = input.WithContext(ctx)

	handlerOutputs := make(chan Event, a.options.HandlerOutputChanSize)

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer close(handlerOutputs)
		if err := a.handler.Handle(input, handlerOutputs); err != nil {
			select {
			case a.errors <- err:
			case <-a.ctx.Done():
			}
		}
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		for evt := range handlerOutputs {
			select {
			case a.outputs <- evt:
			case <-ctx.Done():
				return
			}
		}
	}()

	wg.Wait()
}
