package llm

import (
	"context"
	"testing"
	"time"
)

func TestStreamChunk(t *testing.T) {
	// Test creating different types of stream chunks
	delta := NewStreamDelta(RoleAssistant, "Hello", nil)
	chunk := NewStreamChunk(delta)

	if chunk.Type() != StreamChunkTypeDelta {
		t.Errorf("expected chunk type %s, got %s", StreamChunkTypeDelta, chunk.Type())
	}

	if chunk.Delta().Content() != "Hello" {
		t.Errorf("expected content 'Hello', got '%s'", chunk.Delta().Content())
	}

	if chunk.IsComplete() {
		t.Error("expected chunk to not be complete")
	}

	if chunk.Error() != nil {
		t.Errorf("expected no error, got %v", chunk.Error())
	}
}

func TestCompleteStreamChunk(t *testing.T) {
	usage := NewChatCompletionUsage(10, 20, 30)
	chunk := NewCompleteStreamChunk(usage)

	if chunk.Type() != StreamChunkTypeComplete {
		t.Errorf("expected chunk type %s, got %s", StreamChunkTypeComplete, chunk.Type())
	}

	if !chunk.IsComplete() {
		t.Error("expected chunk to be complete")
	}

	if chunk.Usage().TotalTokens() != 30 {
		t.Errorf("expected total tokens 30, got %d", chunk.Usage().TotalTokens())
	}
}

func TestErrorStreamChunk(t *testing.T) {
	testErr := NewError("test error")
	chunk := NewErrorStreamChunk(testErr)

	if chunk.Type() != StreamChunkTypeError {
		t.Errorf("expected chunk type %s, got %s", StreamChunkTypeError, chunk.Type())
	}

	if chunk.Error() == nil {
		t.Error("expected error, got nil")
	}

	if chunk.Error().Error() != "test error" {
		t.Errorf("expected error 'test error', got '%s'", chunk.Error().Error())
	}
}

func TestStreamDelta(t *testing.T) {
	toolCallDelta := NewToolCallDelta(0, "call_123", "get_weather", `{"location": "Paris"}`)
	delta := NewStreamDelta(RoleAssistant, "The weather in", toolCallDelta)

	if delta.Role() != RoleAssistant {
		t.Errorf("expected role %s, got %s", RoleAssistant, delta.Role())
	}

	if delta.Content() != "The weather in" {
		t.Errorf("expected content 'The weather in', got '%s'", delta.Content())
	}

	toolCalls := delta.ToolCalls()
	if len(toolCalls) != 1 {
		t.Errorf("expected 1 tool call, got %d", len(toolCalls))
	}

	if toolCalls[0].ID() != "call_123" {
		t.Errorf("expected tool call ID 'call_123', got '%s'", toolCalls[0].ID())
	}
}

func TestToolCallDelta(t *testing.T) {
	delta := NewToolCallDelta(0, "call_123", "get_weather", `{"location": "Paris"}`)

	if delta.Index() != 0 {
		t.Errorf("expected index 0, got %d", delta.Index())
	}

	if delta.ID() != "call_123" {
		t.Errorf("expected ID 'call_123', got '%s'", delta.ID())
	}

	if delta.Name() != "get_weather" {
		t.Errorf("expected name 'get_weather', got '%s'", delta.Name())
	}

	if delta.ParametersDelta() != `{"location": "Paris"}` {
		t.Errorf("expected parameters delta '{\"location\": \"Paris\"}', got '%s'", delta.ParametersDelta())
	}
}

func TestStreamingUsageTracker(t *testing.T) {
	tracker := NewStreamingUsageTracker()

	// Test initial state
	usage := tracker.Usage()
	if usage.TotalTokens() != 0 {
		t.Errorf("expected initial total tokens 0, got %d", usage.TotalTokens())
	}

	// Test updating with usage chunk
	usageData := NewChatCompletionUsage(10, 20, 30)
	usageChunk := NewCompleteStreamChunk(usageData)
	tracker.Update(usageChunk)

	updatedUsage := tracker.Usage()
	if updatedUsage.TotalTokens() != 30 {
		t.Errorf("expected total tokens 30, got %d", updatedUsage.TotalTokens())
	}

	if updatedUsage.PromptTokens() != 10 {
		t.Errorf("expected prompt tokens 10, got %d", updatedUsage.PromptTokens())
	}

	if updatedUsage.CompletionTokens() != 20 {
		t.Errorf("expected completion tokens 20, got %d", updatedUsage.CompletionTokens())
	}
}

// MockStreamingClient for testing
type MockStreamingClient struct {
	chunks []StreamChunk
}

func (m *MockStreamingClient) ChatCompletionStream(ctx context.Context, funcs ...ChatCompletionOptionFunc) (<-chan StreamChunk, error) {
	ch := make(chan StreamChunk, len(m.chunks))

	go func() {
		defer close(ch)
		for _, chunk := range m.chunks {
			select {
			case <-ctx.Done():
				return
			case ch <- chunk:
			}
		}
	}()

	return ch, nil
}

func TestMockStreamingClient(t *testing.T) {
	// Create mock client with test chunks
	chunks := []StreamChunk{
		NewStreamChunk(NewStreamDelta(RoleAssistant, "Hello", nil)),
		NewStreamChunk(NewStreamDelta(RoleAssistant, " world", nil)),
		NewCompleteStreamChunk(NewChatCompletionUsage(5, 10, 15)),
	}

	client := &MockStreamingClient{chunks: chunks}

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	stream, err := client.ChatCompletionStream(ctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var receivedChunks []StreamChunk
	for chunk := range stream {
		receivedChunks = append(receivedChunks, chunk)
	}

	if len(receivedChunks) != 3 {
		t.Errorf("expected 3 chunks, got %d", len(receivedChunks))
	}

	// Test first chunk
	if receivedChunks[0].Delta().Content() != "Hello" {
		t.Errorf("expected first chunk content 'Hello', got '%s'", receivedChunks[0].Delta().Content())
	}

	// Test second chunk
	if receivedChunks[1].Delta().Content() != " world" {
		t.Errorf("expected second chunk content ' world', got '%s'", receivedChunks[1].Delta().Content())
	}

	// Test completion chunk
	if !receivedChunks[2].IsComplete() {
		t.Error("expected third chunk to be complete")
	}

	if receivedChunks[2].Usage().TotalTokens() != 15 {
		t.Errorf("expected total tokens 15, got %d", receivedChunks[2].Usage().TotalTokens())
	}
}
