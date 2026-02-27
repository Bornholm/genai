package a2a

import "encoding/json"

// JSON-RPC 2.0 primitives

// JSONRPCRequest represents a JSON-RPC 2.0 request
type JSONRPCRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      any             `json:"id,omitempty"` // string or int
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

// JSONRPCResponse represents a JSON-RPC 2.0 response
type JSONRPCResponse struct {
	JSONRPC string        `json:"jsonrpc"`
	ID      any           `json:"id,omitempty"`
	Result  any           `json:"result,omitempty"`
	Error   *JSONRPCError `json:"error,omitempty"`
}

// JSONRPCError represents a JSON-RPC 2.0 error
type JSONRPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Data    any    `json:"data,omitempty"`
}

// A2A method names
const (
	MethodTasksSend          = "tasks/send"
	MethodTasksSendSubscribe = "tasks/sendSubscribe"
	MethodTasksGet           = "tasks/get"
	MethodTasksCancel        = "tasks/cancel"
)

// TaskSendParams represents the parameters for tasks/send and tasks/sendSubscribe
type TaskSendParams struct {
	ID        string         `json:"id"`
	SessionID string         `json:"sessionId,omitempty"`
	Message   Message        `json:"message"`
	Metadata  map[string]any `json:"metadata,omitempty"`
}

// TaskQueryParams represents the parameters for tasks/get and tasks/cancel
type TaskQueryParams struct {
	ID            string `json:"id"`
	HistoryLength *int   `json:"historyLength,omitempty"`
}

// TaskStatusUpdateEvent represents a streaming event for task status updates (for SSE)
type TaskStatusUpdateEvent struct {
	ID     string     `json:"id"`
	Status TaskStatus `json:"status"`
	Final  bool       `json:"final"`
}

// TaskArtifactUpdateEvent represents a streaming event for artifact updates (for SSE)
type TaskArtifactUpdateEvent struct {
	ID       string   `json:"id"`
	Artifact Artifact `json:"artifact"`
}
