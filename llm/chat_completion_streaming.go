package llm

import (
	"context"
)

// ChatCompletionStreamingClient defines the interface for streaming chat completions
type ChatCompletionStreamingClient interface {
	ChatCompletionStream(ctx context.Context, funcs ...ChatCompletionOptionFunc) (<-chan StreamChunk, error)
}

// StreamChunkType represents the type of streaming chunk
type StreamChunkType string

const (
	StreamChunkTypeDelta    StreamChunkType = "delta"
	StreamChunkTypeUsage    StreamChunkType = "usage"
	StreamChunkTypeError    StreamChunkType = "error"
	StreamChunkTypeComplete StreamChunkType = "complete"
)

// StreamChunk represents a single streaming response chunk
type StreamChunk interface {
	Type() StreamChunkType
	Delta() StreamDelta
	Usage() ChatCompletionUsage
	Error() error
	IsComplete() bool
}

// StreamDelta represents incremental content changes in a streaming response
type StreamDelta interface {
	Role() Role
	Content() string
	ToolCalls() []ToolCallDelta
}

// ToolCallDelta represents incremental tool call data in streaming
type ToolCallDelta interface {
	Index() int
	ID() string
	Name() string
	ParametersDelta() string
}

// BaseStreamChunk provides a base implementation of StreamChunk
type BaseStreamChunk struct {
	chunkType StreamChunkType
	delta     StreamDelta
	usage     ChatCompletionUsage
	err       error
	complete  bool
}

// Type implements StreamChunk
func (c *BaseStreamChunk) Type() StreamChunkType {
	return c.chunkType
}

// Delta implements StreamChunk
func (c *BaseStreamChunk) Delta() StreamDelta {
	return c.delta
}

// Usage implements StreamChunk
func (c *BaseStreamChunk) Usage() ChatCompletionUsage {
	return c.usage
}

// Error implements StreamChunk
func (c *BaseStreamChunk) Error() error {
	return c.err
}

// IsComplete implements StreamChunk
func (c *BaseStreamChunk) IsComplete() bool {
	return c.complete
}

var _ StreamChunk = &BaseStreamChunk{}

// BaseStreamDelta provides a base implementation of StreamDelta
type BaseStreamDelta struct {
	role      Role
	content   string
	toolCalls []ToolCallDelta
}

// Role implements StreamDelta
func (d *BaseStreamDelta) Role() Role {
	return d.role
}

// Content implements StreamDelta
func (d *BaseStreamDelta) Content() string {
	return d.content
}

// ToolCalls implements StreamDelta
func (d *BaseStreamDelta) ToolCalls() []ToolCallDelta {
	return d.toolCalls
}

var _ StreamDelta = &BaseStreamDelta{}

// BaseToolCallDelta provides a base implementation of ToolCallDelta
type BaseToolCallDelta struct {
	index           int
	id              string
	name            string
	parametersDelta string
}

// Index implements ToolCallDelta
func (t *BaseToolCallDelta) Index() int {
	return t.index
}

// ID implements ToolCallDelta
func (t *BaseToolCallDelta) ID() string {
	return t.id
}

// Name implements ToolCallDelta
func (t *BaseToolCallDelta) Name() string {
	return t.name
}

// ParametersDelta implements ToolCallDelta
func (t *BaseToolCallDelta) ParametersDelta() string {
	return t.parametersDelta
}

var _ ToolCallDelta = &BaseToolCallDelta{}

// NewStreamChunk creates a new streaming chunk with delta content
func NewStreamChunk(delta StreamDelta) *BaseStreamChunk {
	return &BaseStreamChunk{
		chunkType: StreamChunkTypeDelta,
		delta:     delta,
		complete:  false,
	}
}

// NewCompleteStreamChunk creates a final streaming chunk with usage information
func NewCompleteStreamChunk(usage ChatCompletionUsage) *BaseStreamChunk {
	return &BaseStreamChunk{
		chunkType: StreamChunkTypeComplete,
		usage:     usage,
		complete:  true,
	}
}

// NewErrorStreamChunk creates an error streaming chunk
func NewErrorStreamChunk(err error) *BaseStreamChunk {
	return &BaseStreamChunk{
		chunkType: StreamChunkTypeError,
		err:       err,
		complete:  false,
	}
}

// NewStreamDelta creates a new stream delta
func NewStreamDelta(role Role, content string, toolCalls ...ToolCallDelta) *BaseStreamDelta {
	return &BaseStreamDelta{
		role:      role,
		content:   content,
		toolCalls: toolCalls,
	}
}

// NewToolCallDelta creates a new tool call delta
func NewToolCallDelta(index int, id, name, parametersDelta string) *BaseToolCallDelta {
	return &BaseToolCallDelta{
		index:           index,
		id:              id,
		name:            name,
		parametersDelta: parametersDelta,
	}
}

// StreamingUsageTracker tracks token usage across streaming chunks
type StreamingUsageTracker struct {
	promptTokens     int64
	completionTokens int64
	totalTokens      int64
}

// Update updates the usage tracker with data from a streaming chunk
func (t *StreamingUsageTracker) Update(chunk StreamChunk) {
	if usage := chunk.Usage(); usage != nil {
		t.promptTokens = usage.PromptTokens()
		t.completionTokens = usage.CompletionTokens()
		t.totalTokens = usage.TotalTokens()
	}
}

// Usage returns the current usage as a ChatCompletionUsage
func (t *StreamingUsageTracker) Usage() ChatCompletionUsage {
	return NewChatCompletionUsage(t.promptTokens, t.completionTokens, t.totalTokens)
}

// NewStreamingUsageTracker creates a new streaming usage tracker
func NewStreamingUsageTracker() *StreamingUsageTracker {
	return &StreamingUsageTracker{}
}
