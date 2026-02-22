package agent

// EventType represents the type of event emitted during agent execution
type EventType string

const (
	EventTypeTextDelta     EventType = "text_delta"
	EventTypeToolCallStart EventType = "tool_call_start"
	EventTypeToolCallDone  EventType = "tool_call_done"
	EventTypeTodoUpdated   EventType = "todo_updated"
	EventTypeError         EventType = "error"
	EventTypeComplete      EventType = "complete"
)

// Event represents an event emitted during agent execution.
// Events are output-only — the caller observes them but does not respond through them.
type Event interface {
	Type() EventType
	Data() any
}

// BaseEvent is a concrete implementation of Event
type BaseEvent struct {
	eventType EventType
	data      any
}

// Type implements Event
func (e *BaseEvent) Type() EventType {
	return e.eventType
}

// Data implements Event
func (e *BaseEvent) Data() any {
	return e.data
}

var _ Event = &BaseEvent{}

// NewEvent creates a new Event with the given type and data
func NewEvent(eventType EventType, data any) Event {
	return &BaseEvent{
		eventType: eventType,
		data:      data,
	}
}

// TextDeltaData represents the data for a EventTypeTextDelta event
type TextDeltaData struct {
	Delta string
}

// ToolCallStartData represents the data for a EventTypeToolCallStart event
type ToolCallStartData struct {
	ID         string
	Name       string
	Parameters any
}

// ToolCallDoneData represents the data for a EventTypeToolCallDone event
type ToolCallDoneData struct {
	ID     string
	Name   string
	Result string
}

// TodoUpdatedData represents the data for a EventTypeTodoUpdated event
type TodoUpdatedData struct {
	Items []TodoItem
}

// TodoItem represents a single todo item
type TodoItem struct {
	ID      string
	Content string
	Status  TodoStatus
}

// TodoStatus represents the status of a todo item
type TodoStatus string

const (
	TodoStatusPending    TodoStatus = "pending"
	TodoStatusInProgress TodoStatus = "in_progress"
	TodoStatusDone       TodoStatus = "done"
)

// ErrorData represents the data for a EventTypeError event
type ErrorData struct {
	Message string
}

// CompleteData represents the data for a EventTypeComplete event
type CompleteData struct {
	Message string
}
