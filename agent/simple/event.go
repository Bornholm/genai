package simple

import (
	"github.com/bornholm/genai/agent"
)

type ResponseEvent interface {
	agent.MessageEvent
	Origin() agent.MessageEvent
}

type BaseResponseEvent struct {
	message string
	origin  agent.MessageEvent
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

func NewResponseEvent(message string, origin agent.MessageEvent) *BaseResponseEvent {
	return &BaseResponseEvent{
		message: message,
		origin:  origin,
	}
}
