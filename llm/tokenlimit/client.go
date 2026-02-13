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
func (c *Client) ChatCompletion(ctx context.Context, funcs ...llm.ChatCompletionOptionFunc) (llm.ChatCompletionResponse, error) {
	response, err := c.client.ChatCompletion(ctx, funcs...)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	if response.Usage() != nil {
		if c.chatCompletionLimiter != nil && response.Usage().TotalTokens() > 0 {
			if err := c.chatCompletionLimiter.WaitN(ctx, int(response.Usage().TotalTokens())); err != nil {
				return nil, errors.WithStack(err)
			}
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

			if c.chatCompletionLimiter != nil && tracker.Usage() != nil {
				currentTokens := tracker.Usage().CompletionTokens()
				if currentTokens > lastCompletionTokens {
					delta := currentTokens - lastCompletionTokens
					if delta > 0 {
						if err := c.chatCompletionLimiter.WaitN(ctx, int(delta)); err != nil {
							outputChan <- llm.NewErrorStreamChunk(errors.WithStack(err))
							return
						}
						lastCompletionTokens = currentTokens
					}
				}
			}

			select {
			case <-ctx.Done():
				return
			case outputChan <- chunk:
			}
		}
	}()

	return outputChan
}

// Embeddings implements llm.Client.
func (c *Client) Embeddings(ctx context.Context, inputs []string, funcs ...llm.EmbeddingsOptionFunc) (llm.EmbeddingsResponse, error) {
	response, err := c.client.Embeddings(ctx, inputs, funcs...)
	if err != nil {
		return nil, errors.WithStack(err)
	}

	if c.embeddingsLimiter != nil && response.Usage() != nil && response.Usage().TotalTokens() > 0 {
		if err := c.embeddingsLimiter.WaitN(ctx, int(response.Usage().TotalTokens())); err != nil {
			return nil, errors.WithStack(err)
		}
	}

	return response, nil
}

func NewClient(client llm.Client, funcs ...OptionFunc) *Client {
	opts := NewOptions()
	return &Client{
		chatCompletionLimiter: opts.ChatCompletionLimiter,
		embeddingsLimiter:     opts.EmbeddingsLimiter,
		client:                client,
	}
}

var _ llm.Client = &Client{}
