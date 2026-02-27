package a2a

// A2A protocol error codes (built on JSON-RPC 2.0)
const (
	// Standard JSON-RPC error codes
	ErrCodeParseError     = -32700
	ErrCodeInvalidRequest = -32600
	ErrCodeMethodNotFound = -32601
	ErrCodeInvalidParams  = -32602
	ErrCodeInternal       = -32603

	// A2A-specific error codes
	ErrCodeTaskNotFound      = -32001
	ErrCodeTaskNotCancelable = -32002
	ErrCodeUnsupported       = -32003
)

// NewTaskNotFoundError creates a task not found error
func NewTaskNotFoundError(taskID string) *JSONRPCError {
	return &JSONRPCError{
		Code:    ErrCodeTaskNotFound,
		Message: "task not found",
		Data:    map[string]string{"taskId": taskID},
	}
}

// NewMethodNotFoundError creates a method not found error
func NewMethodNotFoundError(method string) *JSONRPCError {
	return &JSONRPCError{
		Code:    ErrCodeMethodNotFound,
		Message: "method not found: " + method,
	}
}

// NewInvalidParamsError creates an invalid params error
func NewInvalidParamsError(message string) *JSONRPCError {
	return &JSONRPCError{
		Code:    ErrCodeInvalidParams,
		Message: message,
	}
}

// NewInternalError creates an internal error
func NewInternalError(message string) *JSONRPCError {
	return &JSONRPCError{
		Code:    ErrCodeInternal,
		Message: message,
	}
}

// NewParseError creates a parse error
func NewParseError(message string) *JSONRPCError {
	return &JSONRPCError{
		Code:    ErrCodeParseError,
		Message: message,
	}
}

// NewInvalidRequestError creates an invalid request error
func NewInvalidRequestError(message string) *JSONRPCError {
	return &JSONRPCError{
		Code:    ErrCodeInvalidRequest,
		Message: message,
	}
}

// NewTaskNotCancelableError creates a task not cancelable error
func NewTaskNotCancelableError(taskID string) *JSONRPCError {
	return &JSONRPCError{
		Code:    ErrCodeTaskNotCancelable,
		Message: "task not cancelable",
		Data:    map[string]string{"taskId": taskID},
	}
}

// NewUnsupportedError creates an unsupported operation error
func NewUnsupportedError(message string) *JSONRPCError {
	return &JSONRPCError{
		Code:    ErrCodeUnsupported,
		Message: message,
	}
}
