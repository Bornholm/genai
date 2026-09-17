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

func TestStreamingUsageTracker_KeepsCacheCreationWithCost(t *testing.T) {
	tracker := NewStreamingUsageTracker()
	usage := NewChatCompletionUsageWithCost(100, 10, 110, 40, 0.5, "USD")
	usage.cacheCreationTokens = 25
	tracker.Update(NewCompleteStreamChunk(usage))

	got := tracker.Usage()
	if cost, _, ok := got.(CostReportingUsage).Cost(); !ok || cost != 0.5 {
		t.Errorf("cost lost: %v", got)
	}
	if got.(CacheCreationReportingUsage).CacheCreationTokens() != 25 {
		t.Errorf("cache creation tokens lost when a cost is reported: %v", got)
	}
}

// TestSendTerminalChunkSurvivesCancellation asserts that the chunk ending a
// stream is delivered even when the context is already canceled — which is
// usually what that chunk is there to report. A plain select would have both
// its cases ready and drop it about half the time.
func TestSendTerminalChunkSurvivesCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	for i := 0; i < 200; i++ {
		chunks := make(chan StreamChunk, 10)
		if !SendTerminalChunk(ctx, chunks, NewErrorStreamChunk(context.Canceled)) {
			t.Fatalf("terminal chunk dropped on attempt %d although the channel had room", i)
		}
		if len(chunks) != 1 {
			t.Fatalf("channel holds %d chunks, want 1", len(chunks))
		}
	}
}

// TestSendChunkGivesUpOnCancellation asserts the other half of the contract: an
// ordinary delta is not forced on a consumer that walked away.
func TestSendChunkGivesUpOnCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	// Unbuffered: the send can only succeed if someone is receiving, and nobody
	// is, so the canceled context must win rather than block forever.
	if SendChunk(ctx, make(chan StreamChunk), NewStreamChunk(NewStreamDelta(RoleAssistant, "tok"))) {
		t.Error("SendChunk reported a delivery on a channel nobody reads")
	}
}

// TestStreamingUsageTrackerIgnoresZeroedUsage asserts that an all-zero usage is
// not counted as a report. Providers synthesize one to carry a termination
// signal; treating it as published counters would make Reported() claim numbers
// nobody produced, which is the confusion the flag exists to prevent.
func TestStreamingUsageTrackerIgnoresZeroedUsage(t *testing.T) {
	tracker := NewStreamingUsageTracker()
	tracker.Update(NewCompleteStreamChunk(NewChatCompletionUsage(0, 0, 0)))
	if tracker.Reported() {
		t.Error("Reported() is true after an all-zero usage: zero must read as unknown")
	}

	tracker.Update(NewCompleteStreamChunk(NewChatCompletionUsage(5, 3, 8)))
	if !tracker.Reported() {
		t.Error("Reported() is false after real counters were published")
	}
}

// TestDrainStreamGivesUpOnDeadline asserts that draining an abandoned stream is
// bounded. Without the deadline a client that watches neither its context nor
// its consumer would hold the draining goroutine for the life of the process.
func TestDrainStreamGivesUpOnDeadline(t *testing.T) {
	// A producer that never closes its channel and never stops sending: what an
	// implementation ignoring its context looks like.
	stream := make(chan StreamChunk)
	done := make(chan struct{})
	go func() {
		defer close(done)
		for {
			select {
			case stream <- NewStreamChunk(NewStreamDelta(RoleAssistant, "tok")):
			case <-done:
				return
			}
		}
	}()

	if DrainStream(stream, 50*time.Millisecond) {
		t.Error("DrainStream reported a stream that ended on its own, but it never closes")
	}
}

// TestDrainStreamReportsACleanEnd asserts the ordinary case: a stream that ends
// on its own is drained to completion and says so, which is what happens with
// every provider that honours its context.
func TestDrainStreamReportsACleanEnd(t *testing.T) {
	stream := make(chan StreamChunk, 3)
	stream <- NewStreamChunk(NewStreamDelta(RoleAssistant, "tok"))
	stream <- NewCompleteStreamChunk(NewChatCompletionUsage(5, 3, 8))
	close(stream)

	if !DrainStream(stream, time.Second) {
		t.Error("DrainStream gave up on a stream that had already ended")
	}
}

// TestDrainStreamWithoutDeadline asserts that a zero timeout means no limit
// rather than an immediate give-up, which would bring back the leak.
func TestDrainStreamWithoutDeadline(t *testing.T) {
	stream := make(chan StreamChunk, 1)
	stream <- NewStreamChunk(NewStreamDelta(RoleAssistant, "tok"))
	go func() {
		time.Sleep(20 * time.Millisecond)
		close(stream)
	}()

	if !DrainStream(stream, 0) {
		t.Error("DrainStream gave up although no deadline was asked for")
	}
}

// TestUsagePublishesCounters covers the one definition of "the provider
// reported something", on which PartialUsage, the wire format and the reference
// accounting all agree.
func TestUsagePublishesCounters(t *testing.T) {
	cost := 0.02
	for _, tc := range []struct {
		name  string
		usage ChatCompletionUsage
		want  bool
	}{
		{"nil", nil, false},
		{"all zero", NewChatCompletionUsage(0, 0, 0), false},
		{"prompt tokens", NewChatCompletionUsage(5, 0, 5), true},
		{"completion tokens", NewChatCompletionUsage(0, 3, 3), true},
		{"cached tokens only", NewChatCompletionUsageWithCache(0, 0, 0, 100), true},
		{"cache creation only", NewChatCompletionUsageWithCacheCreation(0, 0, 0, 0, 50), true},
		{"cost only", NewChatCompletionUsageWithCost(0, 0, 0, 0, cost, "USD"), true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := UsagePublishesCounters(tc.usage); got != tc.want {
				t.Errorf("UsagePublishesCounters() = %v, want %v", got, tc.want)
			}
		})
	}
}
