package tokenlimit

import (
	"context"

	"github.com/bornholm/genai/llm"
	"github.com/pkg/errors"
	"golang.org/x/time/rate"
)

type Client struct {
	chatCompletionLimiter *rate.Limiter
	embeddingsLimiter     *rate.Limiter
	client                llm.Client
}

// ChatCompletion implements llm.Client.
//
// NOTE: Rate limiting is applied after the request completes, because the actual
// token count is only known from the response. This means the first requests in a
// burst will go through immediately; subsequent requests will be delayed to respect
// the configured token-per-interval rate.
func (c *Client) ChatCompletion(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	response, err := c.client.ChatCompletion(ctx, funcs...)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	if c.chatCompletionLimiter != nil && response.Usage() != nil && response.Usage().TotalTokens() > 0 {
		if err := waitN(ctx, c.chatCompletionLimiter, int(response.Usage().TotalTokens())); err != nil {
			return nil, errors.WithStack(err)
		}
	}

	return response, nil
}

// ChatCompletionStream implements llm.Client.
func (c *Client) ChatCompletionStream(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (<-chan llm.StreamChunk, error) {
	stream, err := c.client.ChatCompletionStream(ctx, funcs...)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	return c.wrapStreamWithTokenTracking(ctx, stream), nil
}

func (c *Client) wrapStreamWithTokenTracking(ctx context.Context, stream <-chan llm.StreamChunk) <-chan llm.StreamChunk {
	outputChan := make(chan llm.StreamChunk)

	go func() {
		defer close(outputChan)
		tracker := llm.NewStreamingUsageTracker()
		var lastCompletionTokens int64

		for chunk := range stream {
			tracker.Update(chunk)

			// Forward the chunk to the consumer first so it is never lost.
			select {
			case <-ctx.Done():
				return
			case outputChan <- chunk:
			}

			// Then apply rate limiting based on the token delta.
			if c.chatCompletionLimiter != nil && tracker.Usage() != nil {
				currentTokens := tracker.Usage().CompletionTokens()
				if currentTokens > lastCompletionTokens {
					delta := int(currentTokens - lastCompletionTokens)
					if err := waitN(ctx, c.chatCompletionLimiter, delta); err != nil {
						outputChan <- llm.NewErrorStreamChunk(errors.WithStack(err))
						return
					}
					lastCompletionTokens = currentTokens
				}
			}
		}
	}()

	return outputChan
}

// Embeddings implements llm.Client.
//
// NOTE: Same post-request rate limiting approach as ChatCompletion.
func (c *Client) Embeddings(ctx context.Context, inputs []string, funcs ...llm.EmbeddingsOptionFunc) (llm.EmbeddingsResponse, error) {
	response, err := c.client.Embeddings(ctx, inputs, funcs...)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	if c.embeddingsLimiter != nil && response.Usage() != nil && response.Usage().TotalTokens() > 0 {
		if err := waitN(ctx, c.embeddingsLimiter, int(response.Usage().TotalTokens())); err != nil {
			return nil, errors.WithStack(err)
		}
	}

	return response, nil
}

func NewClient(client llm.Client, funcs ...OptionFunc) *Client {
	opts := NewOptions(funcs...)
	return &Client{
		chatCompletionLimiter: opts.ChatCompletionLimiter,
		embeddingsLimiter:     opts.EmbeddingsLimiter,
		client:                client,
	}
}

var _ llm.Client = &Client{}

// waitN splits the wait into chunks no larger than the limiter's burst size
// to avoid "WaitN(n) exceeds limiter's burst" errors.
func waitN(ctx context.Context, limiter *rate.Limiter, n int) error {
	burst := limiter.Burst()
	for n > 0 {
		take := min(n, burst)
		if err := limiter.WaitN(ctx, take); err != nil {
			return err
		}
		n -= take
	}
	return nil
}
