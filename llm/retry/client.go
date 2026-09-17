package retry

import (
	"context"
	"log/slog"
	"time"

	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
)

type Client struct {
	baseDelay  time.Duration
	maxRetries int
	client     llm.Client
}

// Embeddings implements llm.Client.
func (c *Client) Embeddings(ctx context.Context, inputs []string, funcs ...llm.EmbeddingsOptionFunc) (llm.EmbeddingsResponse, error) {
	backoff := c.baseDelay
	maxRetries := c.maxRetries
	retries := 0

	for {
		res, err := c.client.Embeddings(ctx, inputs, funcs...)
		if err != nil {
			if retries >= maxRetries {
				return nil, errors.WithStack(err)
			}

			if llm.IsRetryable(err) {
				slog.DebugContext(ctx, "request failed, will retry", slog.Int("retries", retries), slog.Duration("backoff", backoff), slog.Any("error", errors.WithStack(err)))

				retries++
				time.Sleep(backoff)
				backoff *= 2
				continue
			}

			return nil, errors.WithStack(err)
		}

		return res, nil
	}
}

// Transcription implements llm.Client.
func (c *Client) Transcription(ctx context.Context, audio []byte, funcs ...llm.TranscriptionOptionFunc) (llm.TranscriptionResponse, error) {
	backoff := c.baseDelay
	maxRetries := c.maxRetries
	retries := 0

	for {
		res, err := c.client.Transcription(ctx, audio, funcs...)
		if err != nil {
			if retries >= maxRetries {
				return nil, errors.WithStack(err)
			}

			if llm.IsRetryable(err) {
				slog.DebugContext(ctx, "request failed, will retry", slog.Int("retries", retries), slog.Duration("backoff", backoff), slog.Any("error", errors.WithStack(err)))

				retries++
				time.Sleep(backoff)
				backoff *= 2
				continue
			}

			return nil, errors.WithStack(err)
		}

		return res, nil
	}
}

// ChatCompletion implements llm.ChatCompletionClient.
func (c *Client) ChatCompletion(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	backoff := c.baseDelay
	maxRetries := c.maxRetries
	retries := 0

	for {
		res, err := c.client.ChatCompletion(ctx, funcs...)
		if err != nil {
			if retries >= maxRetries {
				return nil, errors.WithStack(err)
			}

			if llm.IsRetryable(err) {
				slog.DebugContext(ctx, "request failed, will retry", slog.Int("retries", retries), slog.Duration("backoff", backoff), slog.Any("error", errors.WithStack(err)))

				retries++
				time.Sleep(backoff)
				backoff *= 2
				continue
			}

			return nil, errors.WithStack(err)
		}

		return res, nil
	}
}

// ChatCompletionStream implements llm.Client.
// Stream errors that are retryable (e.g. 429) trigger a full retry of the call.
// All retries and stream reading happen inside a goroutine; the returned channel
// carries both data chunks and any eventual non-retryable error chunk.
func (c *Client) ChatCompletionStream(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (<-chan llm.StreamChunk, error) {
	outCh := make(chan llm.StreamChunk, 10)

	// See llm.SendChunk: forwarding with a bare channel write would strand this
	// goroutine, and the stream it wraps, as soon as a consumer stops reading.
	// The chunk that ends the stream goes through llm.SendTerminalChunk, which
	// survives the cancellation it usually reports.
	send := func(chunk llm.StreamChunk) bool { return llm.SendChunk(ctx, outCh, chunk) }
	sendTerminal := func(chunk llm.StreamChunk) { llm.SendTerminalChunk(ctx, outCh, chunk) }

	go func() {
		defer close(outCh)

		backoff := c.baseDelay
		retries := 0

		// Each attempt gets its own context so that the stream of an abandoned
		// one can be told to stop. Without it the provider goroutine of a
		// retried attempt stays blocked on a send nobody reads — ctx is still
		// alive — holding its upstream response open, and billed, for good.
		var cancelAttempt context.CancelFunc
		abandonAttempt := func(stream <-chan llm.StreamChunk) {
			if cancelAttempt == nil {
				return
			}
			cancelAttempt()
			cancelAttempt = nil
			go func() {
				for range stream { //nolint:revive // draining, the values are of no use
				}
			}()
		}

		for {
			attemptCtx, cancel := context.WithCancel(ctx)
			cancelAttempt = cancel
			stream, err := c.client.ChatCompletionStream(attemptCtx, funcs...)
			if err != nil {
				// Nothing was opened, but the attempt context still has to be
				// released before the next one replaces it.
				cancelAttempt()
				cancelAttempt = nil
				if retries < c.maxRetries && llm.IsRetryable(err) {
					slog.DebugContext(ctx, "stream open failed, will retry", slog.Int("retries", retries), slog.Duration("backoff", backoff), slog.Any("error", err))
					retries++
					select {
					case <-time.After(backoff):
						backoff *= 2
					case <-ctx.Done():
						sendTerminal(llm.NewErrorStreamChunk(errors.WithStack(ctx.Err())))
						return
					}
					continue
				}
				sendTerminal(llm.NewErrorStreamChunk(errors.WithStack(err)))
				return
			}

			// Read the stream; retry the whole call on retryable chunk errors.
			// Use select to respect ctx cancellation — a stalled TCP connection
			// would otherwise block this goroutine indefinitely.
			var retryCall bool
		streamLoop:
			for {
				select {
				case chunk, ok := <-stream:
					if !ok {
						break streamLoop
					}
					if chunkErr := chunk.Error(); chunkErr != nil {
						if retries < c.maxRetries && llm.IsRetryable(chunkErr) {
							slog.DebugContext(ctx, "stream chunk error, will retry", slog.Int("retries", retries), slog.Duration("backoff", backoff), slog.Any("error", chunkErr))
							retries++
							retryCall = true
							break streamLoop
						}
						abandonAttempt(stream) // non-retryable error — forward and stop
						sendTerminal(chunk)
						return
					}
					if !send(chunk) {
						abandonAttempt(stream)
						return
					}
				case <-ctx.Done():
					abandonAttempt(stream)
					sendTerminal(llm.NewErrorStreamChunk(errors.WithStack(ctx.Err())))
					return
				}
			}

			if !retryCall {
				// The provider closed its channel: nothing left to abandon,
				// only the attempt context to release.
				cancelAttempt()
				return // stream ended normally
			}
			abandonAttempt(stream)
			// retryCall == true: wait (respecting ctx) then open a fresh stream
			select {
			case <-time.After(backoff):
				backoff *= 2
			case <-ctx.Done():
				sendTerminal(llm.NewErrorStreamChunk(errors.WithStack(ctx.Err())))
				return
			}
		}
	}()

	return outCh, nil
}

func NewClient(client llm.Client, baseDelay time.Duration, maxRetries int) *Client {
	return &Client{
		baseDelay:  baseDelay,
		maxRetries: maxRetries,
		client:     client,
	}
}

var _ llm.Client = &Client{}
