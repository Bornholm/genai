package simple

import (
	"context"

	"github.com/bornholm/genai/agent"
)

type ResponseEvent interface {
	agent.MessageEvent
	Origin() agent.MessageEvent
}

type BaseResponseEvent struct {
	id      agent.EventID
	ctx     context.Context
	message string
	origin  agent.MessageEvent
}

// ID implements ResponseEvent.
func (e *BaseResponseEvent) ID() agent.EventID {
	return e.id
}

// WithContext implements ResponseEvent.
func (e *BaseResponseEvent) WithContext(ctx context.Context) agent.Event {
	return &BaseResponseEvent{
		id:      e.id,
		ctx:     ctx,
		message: e.message,
		origin:  e.origin,
	}
}

// Context implements ResponseEvent.
func (e *BaseResponseEvent) Context() context.Context {
	return e.ctx
}

// Origin implements ResponseEvent.
func (e *BaseResponseEvent) Origin() agent.MessageEvent {
	return e.origin
}

// Event implements MessageEvent.
func (e *BaseResponseEvent) Event() {}

// Message implements MessageEvent.
func (e *BaseResponseEvent) Message() string {
	return e.message
}

var _ ResponseEvent = &BaseResponseEvent{}

func NewResponseEvent(ctx context.Context, message string, origin agent.MessageEvent) *BaseResponseEvent {
	return &BaseResponseEvent{
		id:      agent.NewEventID(),
		ctx:     ctx,
		message: message,
		origin:  origin,
	}
}
