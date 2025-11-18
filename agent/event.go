package agent

import (
	"context"

	"github.com/rs/xid"
)

type Event interface {
	ID() EventID
	Context() context.Context
	WithContext(ctx context.Context) Event
}

// MessageEvent represents an event that carries a string message.
type MessageEvent interface {
	Event
	Message() string
}

type BaseMessageEvent struct {
	id      EventID
	ctx     context.Context
	message string
}

// ID implements MessageEvent.
func (e *BaseMessageEvent) ID() EventID {
	return e.id
}

// Context implements MessageEvent.
func (e *BaseMessageEvent) Context() context.Context {
	return e.ctx
}

// WithContext implements MessageEvent.
func (e *BaseMessageEvent) WithContext(ctx context.Context) Event {
	return &BaseMessageEvent{
		id:      e.id,
		ctx:     ctx,
		message: e.message,
	}
}

// Message implements the MessageEvent interface.
func (e *BaseMessageEvent) Message() string {
	return e.message
}

var _ MessageEvent = &BaseMessageEvent{}

// NewMessageEvent creates a new BaseMessageEvent.
func NewMessageEvent(ctx context.Context, message string) *BaseMessageEvent {
	return &BaseMessageEvent{
		ctx:     ctx,
		message: message,
	}
}

type EventID string

func NewEventID() EventID {
	return EventID(xid.New().String())
}
