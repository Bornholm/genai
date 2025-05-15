package agent

import (
	"context"
	"sync"

	"github.com/pkg/errors"
)

type Handler interface {
	Handle(ctx context.Context, input Event, outputs chan Event) error
}

type HandlerFunc func(ctx context.Context, input Event, outputs chan Event) error

func (fn HandlerFunc) Handle(ctx context.Context, input Event, outputs chan Event) error {
	return fn(ctx, input, outputs)
}

type Agent struct {
	handler Handler

	inputs  chan Event
	outputs chan Event

	mutex *sync.Mutex
	cond  *sync.Cond
}

func (a *Agent) Output() chan Event {
	a.mutex.Lock()
	if a.outputs == nil {
		a.cond.Wait()
	}
	a.mutex.Unlock()

	return a.outputs
}

func (a *Agent) In(evt Event) error {
	a.mutex.Lock()
	if a.inputs == nil {
		a.cond.Wait()
	}
	a.mutex.Unlock()

	a.inputs <- evt

	return nil
}

func (a *Agent) Run(ctx context.Context) error {
	a.mutex.Lock()
	a.inputs = make(chan Event)
	a.outputs = make(chan Event)
	a.cond.Broadcast()
	a.mutex.Unlock()

	defer close(a.inputs)
	defer close(a.outputs)

	for {
		select {
		case evt, ok := <-a.inputs:
			if !ok {
				return nil
			}

			if err := a.handle(ctx, evt); err != nil {
				return errors.WithStack(err)
			}

		case <-ctx.Done():
			return errors.WithStack(ctx.Err())
		}
	}
}

func (a *Agent) handle(ctx context.Context, input Event) error {
	handlerCtx, cancel := context.WithCancel(ctx)

	handlerCtx = WithContextAgent(handlerCtx, a)

	handlerOutputs := make(chan Event)
	go func() {
		defer cancel()

		for evt := range handlerOutputs {
			a.outputs <- evt
		}
	}()

	if err := a.handler.Handle(handlerCtx, input, handlerOutputs); err != nil {
		return errors.WithStack(err)
	}

	return nil
}

func New(handler Handler, middlewares ...Middleware) *Agent {
	var mutex sync.Mutex

	return &Agent{
		handler: chain(handler, middlewares),
		outputs: make(chan Event),
		mutex:   &mutex,
		cond:    sync.NewCond(&mutex),
	}
}
