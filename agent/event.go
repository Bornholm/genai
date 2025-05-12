package agent

type Event interface {
	Event()
}

type MessageEvent interface {
	Event
	Message() string
}

type BaseMessageEvent struct {
	message string
}

// Event implements MessageEvent.
func (e *BaseMessageEvent) Event() {}

// Message implements MessageEvent.
func (e *BaseMessageEvent) Message() string {
	return e.message
}

var _ MessageEvent = &BaseMessageEvent{}

func NewMessageEvent(message string) *BaseMessageEvent {
	return &BaseMessageEvent{
		message: message,
	}
}
