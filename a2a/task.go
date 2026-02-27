package a2a

import "time"

// TaskState represents the current state of a task
type TaskState string

const (
	TaskStateSubmitted   TaskState = "submitted"
	TaskStateWorking     TaskState = "working"
	TaskStateInputNeeded TaskState = "input-needed"
	TaskStateCompleted   TaskState = "completed"
	TaskStateCanceled    TaskState = "canceled"
	TaskStateFailed      TaskState = "failed"
)

// Task represents a unit of work processed by an agent
type Task struct {
	ID        string         `json:"id"`
	SessionID string         `json:"sessionId,omitempty"`
	Status    TaskStatus     `json:"status"`
	History   []Message      `json:"history,omitempty"`
	Artifacts []Artifact     `json:"artifacts,omitempty"`
	Metadata  map[string]any `json:"metadata,omitempty"`
}

// TaskStatus represents the status of a task
type TaskStatus struct {
	State     TaskState `json:"state"`
	Timestamp time.Time `json:"timestamp"`
	Message   *Message  `json:"message,omitempty"`
}

// Message represents a message in the task history
type Message struct {
	Role  string `json:"role"` // "user" or "agent"
	Parts []Part `json:"parts"`
}

// Part is a union type — exactly one field is non-nil.
type Part struct {
	Type     string         `json:"type"` // "text", "file", "data"
	Text     string         `json:"text,omitempty"`
	File     *FilePart      `json:"file,omitempty"`
	Data     map[string]any `json:"data,omitempty"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

// FilePart represents a file in a message part
type FilePart struct {
	Name     string `json:"name,omitempty"`
	MimeType string `json:"mimeType,omitempty"`
	Bytes    string `json:"bytes,omitempty"` // base64
	URI      string `json:"uri,omitempty"`
}

// Artifact represents an output artifact produced by the agent
type Artifact struct {
	Name        string         `json:"name,omitempty"`
	Description string         `json:"description,omitempty"`
	Parts       []Part         `json:"parts"`
	Index       int            `json:"index"`
	Metadata    map[string]any `json:"metadata,omitempty"`
}

// NewTextPart creates a new text part
func NewTextPart(text string) Part {
	return Part{
		Type: "text",
		Text: text,
	}
}

// NewFilePart creates a new file part
func NewFilePart(file *FilePart) Part {
	return Part{
		Type: "file",
		File: file,
	}
}

// NewDataPart creates a new data part
func NewDataPart(data map[string]any) Part {
	return Part{
		Type: "data",
		Data: data,
	}
}
