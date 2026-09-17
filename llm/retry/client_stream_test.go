package retry

import (
	"context"
	"errors"
	"net/http"
	"testing"
	"time"

	"github.com/bornholm/genai/llm"
)

// scriptedStreamClient serves one stream per attempt, from the scripts it is
// given, and records the context each attempt ran with so a test can assert
// that an abandoned attempt was actually told to stop.
type scriptedStreamClient struct {
	scripts  [][]llm.StreamChunk
	attempts int
	ctxs     []context.Context
	// block, when set on an attempt index, makes that attempt keep sending
	// forever instead of ending, the way a provider does while nobody hangs up.
	block map[int]bool
}

func (c *scriptedStreamClient) ChatCompletionStream(ctx context.Context, _ ...llm.ChatCompletionOptionFunc) (<-chan llm.StreamChunk, error) {
	attempt := c.attempts
	c.attempts++
	c.ctxs = append(c.ctxs, ctx)

	out := make(chan llm.StreamChunk, 10)
	var script []llm.StreamChunk
	if attempt < len(c.scripts) {
		script = c.scripts[attempt]
	}

	go func() {
		defer close(out)
		for _, chunk := range script {
			if !llm.SendChunk(ctx, out, chunk) {
				llm.SendTerminalChunk(ctx, out, llm.NewErrorStreamChunk(ctx.Err()))
				return
			}
		}
		if !c.block[attempt] {
			return
		}
		// Keep producing until this attempt's context is canceled: without the
		// wrapper cancelling it, this goroutine would live on.
		for {
			// Ending with a terminal chunk is what the contract asks of an
			// implementation; the wrapper is what this exercises, not the fake.
			if !llm.SendChunk(ctx, out, llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, "tok"))) {
				llm.SendTerminalChunk(ctx, out, llm.NewErrorStreamChunk(ctx.Err()))
				return
			}
		}
	}()

	return out, nil
}

func (c *scriptedStreamClient) ChatCompletion(_ context.Context, _ ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	return nil, nil
}

func (c *scriptedStreamClient) Embeddings(_ context.Context, _ []string, _ ...llm.EmbeddingsOptionFunc) (llm.EmbeddingsResponse, error) {
	return nil, nil
}

func (c *scriptedStreamClient) Transcription(_ context.Context, _ []byte, _ ...llm.TranscriptionOptionFunc) (llm.TranscriptionResponse, error) {
	return nil, nil
}

func retryable() error {
	return llm.RateLimitError(http.StatusTooManyRequests, "slow down")
}

// TestChatCompletionStream_AbandonsTheRetriedAttempt asserts that a retried
// attempt is cancelled and drained. Left alone, its provider goroutine stays
// blocked on a send nobody reads and holds its upstream response open — and
// billed — for the life of the process.
func TestChatCompletionStream_AbandonsTheRetriedAttempt(t *testing.T) {
	client := &scriptedStreamClient{
		scripts: [][]llm.StreamChunk{
			{llm.NewErrorStreamChunk(retryable())},
			{llm.NewCompleteStreamChunk(llm.NewChatCompletionUsage(5, 3, 8))},
		},
		block: map[int]bool{0: true},
	}

	stream, err := NewClient(client, time.Millisecond, 2).ChatCompletionStream(context.Background())
	if err != nil {
		t.Fatalf("ChatCompletionStream: %v", err)
	}

	var complete bool
	for chunk := range stream {
		if chunk.Error() != nil {
			t.Fatalf("the retry was not transparent: %v", chunk.Error())
		}
		complete = complete || chunk.IsComplete()
	}
	if !complete {
		t.Fatal("no completion chunk: the retried call never delivered its stream")
	}
	if client.attempts != 2 {
		t.Fatalf("attempts = %d, want 2", client.attempts)
	}

	// The abandoned attempt must have been told to stop. That the one which
	// replaced it was not cut short is what the completion chunk above proves;
	// its context is released too, once its stream has ended.
	if err := client.ctxs[0].Err(); err == nil {
		t.Error("the abandoned attempt was never cancelled: its provider keeps producing into a channel nobody reads")
	}
}

// TestChatCompletionStream_ReportsTheConsumerLeaving asserts that giving up on
// the wrapper ends its stream with an error chunk rather than a bare close. A
// channel closing on nothing reads as a truncation — an upstream incident —
// instead of the ordinary hangup it is.
func TestChatCompletionStream_ReportsTheConsumerLeaving(t *testing.T) {
	client := &scriptedStreamClient{
		scripts: [][]llm.StreamChunk{{llm.NewStreamChunk(llm.NewStreamDelta(llm.RoleAssistant, "tok"))}},
		block:   map[int]bool{0: true},
	}

	ctx, cancel := context.WithCancel(context.Background())
	stream, err := NewClient(client, time.Millisecond, 2).ChatCompletionStream(ctx)
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
	if !errors.Is(last.Error(), context.Canceled) {
		t.Errorf("terminal error = %v, want a cancellation", last.Error())
	}
}

// TestChatCompletionStream_ForwardsANonRetryableError asserts the ordinary
// failure path still ends with the provider's own error chunk.
func TestChatCompletionStream_ForwardsANonRetryableError(t *testing.T) {
	boom := errors.New("upstream exploded")
	client := &scriptedStreamClient{
		scripts: [][]llm.StreamChunk{{llm.NewErrorStreamChunk(boom)}},
	}

	stream, err := NewClient(client, time.Millisecond, 2).ChatCompletionStream(context.Background())
	if err != nil {
		t.Fatalf("ChatCompletionStream: %v", err)
	}

	var last llm.StreamChunk
	for chunk := range stream {
		last = chunk
	}
	if last == nil || !errors.Is(last.Error(), boom) {
		t.Fatalf("terminal chunk = %v, want the provider error", last)
	}
	if client.attempts != 1 {
		t.Errorf("attempts = %d, want 1: a non-retryable error must not be retried", client.attempts)
	}
}
