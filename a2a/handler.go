package a2a

import "context"

// TaskHandler is the interface that the agent loop must satisfy
// to be used as an A2A agent backend.
type TaskHandler interface {
	// HandleTask processes a task/send request synchronously.
	// It receives the full task params, executes the agent loop,
	// and returns the completed task.
	HandleTask(ctx context.Context, params TaskSendParams) (*Task, error)

	// HandleTaskSubscribe processes a task/sendSubscribe request.
	// It streams status/artifact updates through the provided channel.
	// The channel is closed by the handler when processing completes.
	HandleTaskSubscribe(ctx context.Context, params TaskSendParams, events chan<- any) error

	// GetTask retrieves a task by ID.
	GetTask(ctx context.Context, params TaskQueryParams) (*Task, error)

	// CancelTask cancels a running task.
	CancelTask(ctx context.Context, params TaskQueryParams) (*Task, error)
}
