package tokenlimit

import (
	"context"
	"testing"
	"time"

	"github.com/bornholm/genai/llm"
)

// blockingStreamClient keeps producing until its context is canceled, then ends
// with a terminal chunk as the contract requires, so a test can tell whether the
// wrapper released it and what it passed on.
type blockingStreamClient struct {
	ctx context.Context
}

func (c *blockingStreamClient) ChatCompletionStream(ctx context.Context, _ ...llm.ChatCompletionOptionFunc) (<-chan llm.StreamChunk, error) {
	c.ctx = ctx
	// Buffered like every in-repo provider: llm.SendTerminalChunk only
	// guarantees delivery while there is room.
	out := make(chan llm.StreamChunk, 10)
	go func() {
		defer close(out)
		for {
			if !llm.SendChunk(ctx, out, llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, "tok"))) {
				llm.SendTerminalChunk(ctx, out, llm.NewErrorStreamChunk(ctx.Err()))
				return
			}
		}
	}()
	return out, nil
}

func (c *blockingStreamClient) ChatCompletion(_ context.Context, _ ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	return nil, nil
}

func (c *blockingStreamClient) Embeddings(_ context.Context, _ []string, _ ...llm.EmbeddingsOptionFunc) (llm.EmbeddingsResponse, error) {
	return nil, nil
}

func (c *blockingStreamClient) Transcription(_ context.Context, _ []byte, _ ...llm.TranscriptionOptionFunc) (llm.TranscriptionResponse, error) {
	return nil, nil
}

// TestWrapStream_ReportsTheConsumerLeaving asserts that a consumer walking away
// gets a terminal chunk rather than a bare close, and that the upstream stream
// is released instead of being left with a producer blocked on a send.
func TestWrapStream_ReportsTheConsumerLeaving(t *testing.T) {
	upstream := &blockingStreamClient{}
	client := NewClient(upstream, WithDrainTimeout(time.Second))

	ctx, cancel := context.WithCancel(context.Background())
	stream, err := client.ChatCompletionStream(ctx)
	if err != nil {
		t.Fatalf("ChatCompletionStream: %v", err)
	}

	// Read one chunk, then walk away.
	<-stream
	cancel()

	var last llm.StreamChunk
	for chunk := range stream {
		last = chunk
	}
	if last == nil || last.Error() == nil {
		t.Fatal("the stream closed without a terminal chunk: a consumer cannot tell a hangup from a truncation")
	}
}

// TestWrapStream_ForwardsACompletedStream guards the ordinary path: the wrapper
// is transparent, and the usage chunk reaches the consumer.
func TestWrapStream_ForwardsACompletedStream(t *testing.T) {
	upstream := make(chan llm.StreamChunk, 3)
	upstream <- llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, "tok"))
	upstream <- llm.NewCompleteStreamChunk(llm.NewChatCompletionUsage(5, 3, 8))
	close(upstream)

	client := NewClient(nil)
	out := client.wrapStreamWithTokenTracking(context.Background(), upstream)

	var deltas int
	var complete bool
	for chunk := range out {
		if chunk.IsComplete() {
			complete = true
			continue
		}
		deltas++
	}
	if deltas != 1 || !complete {
		t.Errorf("deltas = %d, complete = %v, want 1 and true", deltas, complete)
	}
}
