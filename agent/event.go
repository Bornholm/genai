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
	// EventTypeReasoning is emitted when the model returns reasoning tokens.
	// It is fired once per LLM turn, immediately after the response is received
	// and before any tool calls are dispatched.
	EventTypeReasoning EventType = "reasoning"
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

// ReasoningData represents the data for a EventTypeReasoning event.
// Reasoning is only present when the underlying model returns thinking tokens
// (e.g. Claude with extended thinking, GPT-5 with reasoning effort > none).
type ReasoningData struct {
	// Reasoning is the plain-text reasoning string returned by the model.
	Reasoning string
	// ReasoningDetails contains the structured reasoning blocks returned by the model.
	// These are needed for multi-turn preservation with models that use encrypted or
	// summarised reasoning blocks (pass them back unmodified in assistant messages).
	ReasoningDetails []ReasoningDetail
}

// ReasoningDetail mirrors llm.ReasoningDetail but lives in the agent package
// so callers of the agent API do not need to import the llm package just for events.
type ReasoningDetail struct {
	ID        string
	Type      string
	Text      string
	Summary   string
	Data      string
	Format    string
	Index     int
	Signature string
}
